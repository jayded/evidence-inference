from os.path import join, dirname, abspath
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))
import copy
import random

import numpy as np
from scipy import stats

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from gensim.models import KeyedVectors

import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

USE_CUDA = True

from evidence_inference.preprocess.preprocessor import SimpleInferenceVectorizer as SimpleInferenceVectorizer
from evidence_inference.models.utils import PaddedSequence
from evidence_inference.models.attention_distributions import TokenAttention, evaluate_model_attention_distribution


class CBoWEncoder(nn.Module):
    """Bag of words encoder for Intervention (also Comparator, Outcome) token sequences.

    Note that ordering information is discarded here, and our words are represented by continuous vectors.
    """

    def __init__(self, vocab_size, embeddings: nn.Embedding=None, embedding_dim=200, use_attention=False, condition_attention=False, tokenwise_attention=False, query_dims=None):
        super(CBoWEncoder, self).__init__()

        self.vocab_size = vocab_size

        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = embeddings
            self.embedding_dim = embeddings.embedding_dim

        self.use_attention = use_attention
        if self.use_attention:
            self.attention_mechanism = TokenAttention(self.embedding_dim, self.query_dims, condition_attention, tokenwise_attention)

    def forward(self, word_inputs: PaddedSequence, query_v_for_attention: torch.Tensor=None, normalize_attention_distribution=True):
        if isinstance(word_inputs, PaddedSequence):
            embedded = self.embedding(word_inputs.data)
        else:
            raise ValueError("Got an unexpected type {} for word_inputs {}".format(type(word_inputs), word_inputs))
        if self.use_attention:
            a = self.attention_mechanism(word_inputs, embedded, query_v_for_attention, normalize=normalize_attention_distribution)
            output = torch.sum(a * embedded, dim=1)
            return None, output, a
        else:
            output = torch.sum(embedded, dim=1) / word_inputs.batch_sizes.unsqueeze(-1).to(torch.float)
            return output


class GRUEncoder(nn.Module):
    """ GRU encoder for Intervention (also Comparator, Outcome) token sequences.

    Also contains attention mechanisms for use with this particular encoder
    """

    def __init__(self, vocab_size, n_layers=1, hidden_size=32, embeddings: nn.Embedding=None,
                 use_attention=False, condition_attention=False, tokenwise_attention=False, query_dims=None, bidirectional=False):
        """ Prepares a GRU encoder for the Intervention, Comparator, or outcome token sequences.

        Either initializes embedding layer from existing embeddings or creates a random one of size vocab X hidden_size.

        When using attention we either:
        * condition on a hidden unit from the encoder and some query vector of size query_dims, which passes a linear
          combination of the two through a non-linearity (Tanh) and then compresses this to a final number
        * or we use a linear function from the output of the encoder.

        In both cases, we use a softmax over the possible outputs to impose a final attention distribution.
        """
        super(GRUEncoder, self).__init__()
        if condition_attention and not use_attention:
            raise ValueError("Cannot condition attention when there is no attention mechanism! Try setting "
                             "use_attention to true or condition_attention to false, ")
        if tokenwise_attention and not use_attention:
            raise ValueError("Cannot have element-wise attention when there is no attention mechanism! Try setting "
                             "use_attention to true or condition_attention to false, ")

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.use_attention = use_attention
        self.condition_attention = condition_attention
        self.tokenwise_attention = tokenwise_attention
        self.query_dims = query_dims
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size

        if embeddings is None:
            self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
            self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=self.bidirectional)
        else:
            self.embedding = embeddings
            self.gru = nn.GRU(input_size=embeddings.embedding_dim, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=self.bidirectional)

        if self.use_attention:
            encoding_size = self.hidden_size + int(self.bidirectional) * self.hidden_size
            self.attention_mechanism = TokenAttention(encoding_size, self.query_dims, condition_attention, tokenwise_attention)

    def forward(self, word_inputs: PaddedSequence, init_hidden: torch.Tensor=None, query_v_for_attention: torch.Tensor=None, normalize_attention_distribution=True) -> (torch.Tensor, torch.Tensor):
        if isinstance(word_inputs, PaddedSequence):
            embedded = self.embedding(word_inputs.data)
            as_padded = word_inputs.pack_other(embedded)
            output, hidden = self.gru(as_padded, init_hidden)
            output = PaddedSequence.from_packed_sequence(output, batch_first=True)
        else:
            raise ValueError("Unknown input type {} for word_inputs: {}, try a PaddedSequence or a Tensor".format(type(word_inputs), word_inputs))

        # concatenate the hidden representations
        if self.bidirectional:
            if self.n_layers > 1:
                raise ValueError("Implement me!")
            hidden = torch.cat([hidden[0], hidden[1]], dim=1)

        if self.use_attention:
            # note that these hidden_input_states are masked to zeros (when appropriate) already when this is called.
            hidden_input_states = output
            a = self.attention_mechanism(hidden_input_states, query_v_for_attention, normalize=normalize_attention_distribution)

            # note this is an element-wise multiplication, so each of the hidden states is weighted by the attention vector
            weighted_hidden = torch.sum(a * output.data, dim=1)
            return output, weighted_hidden, a

        return output, hidden


class InferenceNet(nn.Module):
    """ Predicts the relative (statistical) benefits of a pair of medical interventions with respect to an outcome.

    The input to the model is:
    * an array of article tokens
    * an array of medical intervention tokens
    * an array of "comparator" tokens (i.e. an alternate intervention)
    * an array of outcome tokens

    The output is a distribution over whether or not the text of the particular article supports the intervention being
    statistically better (p=0.05), neutral, or worse than the comparator for the outcome.

    This model works via:
    * encoding the article via a gated recurrent unit
    * encoding the intervention, comparator, and outcome via either a gated recurrent unit or a continuous bag of words encoder
    * optionally allowing a separate attention mechanism within each of these units to either:
        * learn a distribution over article tokens
        * learn a distribution over article tokens conditioned on the intervention, comparator, and outcome encodings
    * passing the encoded result through a linear layer and then a softmax
    """

    def __init__(self, vectorizer, h_size=32,
                 init_embeddings=None,
                 init_wvs_path="embeddings/PubMed-w2v.bin",
                 weight_tying=False,
                 ICO_encoder="CBoW",
                 article_encoder="GRU",
                 attention_over_article_tokens=True,
                 condition_attention=True,
                 tokenwise_attention=False,
                 tune_embeddings=False,
                 h_dropout_rate=0.2):
        super(InferenceNet, self).__init__()
        if condition_attention and not attention_over_article_tokens:
            raise ValueError("Must have attention in order to have conditional attention!")

        self.vectorizer = vectorizer
        vocab_size = len(self.vectorizer.idx_to_str)
        
        if init_embeddings is None:
            print("loading pre-trained embeddings...")
            init_embedding_weights = InferenceNet.init_word_vectors(init_wvs_path, vectorizer)
            print("done.")
        else:
            print("Using provided embeddings")
            init_embedding_weights = init_embeddings

        self.ICO_encoder = ICO_encoder

        # this is the size of the concatenated <abstract, I, C, O> representations,
        # which will depend on the encoder variant being used.
        self.ICO_dims = None

        if ICO_encoder == "CBoW":
            self.intervention_encoder = CBoWEncoder(vocab_size=vocab_size, embeddings=init_embedding_weights)
            self.comparator_encoder = CBoWEncoder(vocab_size=vocab_size, embeddings=init_embedding_weights)
            self.outcome_encoder = CBoWEncoder(vocab_size=vocab_size, embeddings=init_embedding_weights)
            if article_encoder == 'CBoW':
                self.ICO_dims = init_embedding_weights.embedding_dim * 3
                MLP_input_size = self.ICO_dims + init_embedding_weights.embedding_dim
                if h_size:
                    print("Warning: ignoring the hidden size as the article encoder is CBoW and emits a fixed output")
            elif article_encoder == 'GRU' or article_encoder == 'biGRU':
                self.ICO_dims = init_embedding_weights.embedding_dim * 3
                MLP_input_size = self.ICO_dims + h_size
            else:
                raise ValueError("Unknown article_encoder type {}".format(article_encoder))
        elif ICO_encoder == "GRU" or ICO_encoder == 'biGRU':
            bidirectional = ICO_encoder == 'biGRU'
            # then use an RNN encoder for I, C, O elements.
            self.intervention_encoder = GRUEncoder(vocab_size=vocab_size, hidden_size=h_size,
                                                   embeddings=init_embedding_weights, bidirectional=bidirectional)
            self.comparator_encoder = GRUEncoder(vocab_size=vocab_size, hidden_size=h_size,
                                                 embeddings=init_embedding_weights, bidirectional=bidirectional)
            self.outcome_encoder = GRUEncoder(vocab_size=vocab_size, hidden_size=h_size,
                                              embeddings=init_embedding_weights, bidirectional=bidirectional)
            self.ICO_dims = h_size * 3 
            if article_encoder == 'CBoW':
                # note that the CBoW encoder ignores the h_size here
                MLP_input_size = self.ICO_dims + init_embedding_weights.embedding_dim
            elif article_encoder == 'GRU' or article_encoder == 'biGRU':
                MLP_input_size = self.ICO_dims + h_size  # the input to the MLP is the concatentation of the ICO hidden states and the article hidden states.
            else:
                raise ValueError("Unknown article_encoder type {}".format(article_encoder))
        else:
            raise ValueError("No such encoder: {}".format(ICO_encoder))

        self.article_encoder_type = article_encoder
        if article_encoder == 'GRU' or article_encoder == 'biGRU':
            bidirectional = article_encoder == 'biGRU'
            self.article_encoder = GRUEncoder(vocab_size=vocab_size, hidden_size=h_size,
                                              embeddings=init_embedding_weights,
                                              use_attention=attention_over_article_tokens,
                                              condition_attention=condition_attention,
                                              tokenwise_attention=tokenwise_attention,
                                              query_dims=self.ICO_dims,
                                              bidirectional=bidirectional)
        elif article_encoder == 'CBoW':
            self.article_encoder = CBoWEncoder(vocab_size=vocab_size,
                                               embeddings=init_embedding_weights,
                                               use_attention=attention_over_article_tokens,
                                               condition_attention=condition_attention,
                                               tokenwise_attention=tokenwise_attention,
                                               query_dims=self.ICO_dims)
        else:
            raise ValueError("Unknown article encoder type: {}".format(article_encoder))

        if not tune_embeddings:
            print("freezing word embedding layer!")
            for layer in (
                    self.article_encoder, self.intervention_encoder, self.comparator_encoder, self.outcome_encoder):
                # note: we are relying on the fact that all encoders will have a
                # "embedding" layer (nn.Embedding). 
                layer.embedding.requires_grad = False
                layer.embedding.weight.requires_grad = False

        # weight tying (optional)
        # note that this is not meaningful (or, rather, does nothing) when embeddings are
        # frozen.
        # TODO note that weights are currently tied because all the ICOEncoders use the same underlying objects.
        if weight_tying:
            print("tying word embedding layers")
            self.intervention_encoder.embedding.weight = self.comparator_encoder.embedding.weight = \
                self.outcome_encoder.embedding.weight = self.article_encoder.embedding.weight
        self.batch_first = True

        self.MLP_hidden = nn.Linear(MLP_input_size, 16)
        self.out = nn.Linear(16, 3)
        self.dropout = nn.Dropout(p=h_dropout_rate)

    def _encode(self, I_tokens, C_tokens, O_tokens):
        if self.ICO_encoder == "CBoW":
            # simpler case of a CBoW encoder.
            I_v = self.intervention_encoder(I_tokens)
            C_v = self.comparator_encoder(C_tokens)
            O_v = self.outcome_encoder(O_tokens)
        elif self.ICO_encoder == 'GRU' or self.ICO_encoder == 'biGRU':
            # then we have an RNN encoder. Hidden layers are automatically initialized
            _, I_v = self.intervention_encoder(I_tokens)
            _, C_v = self.comparator_encoder(C_tokens)
            _, O_v = self.outcome_encoder(O_tokens)
        else:
            raise ValueError("No such encoder: {}".format(self.ICO_encoder))
        return I_v, C_v, O_v

    def forward(self, article_tokens: PaddedSequence, I_tokens: PaddedSequence, C_tokens: PaddedSequence, O_tokens: PaddedSequence,
                batch_size, debug_attn=False, verbose_attn=False):
        if isinstance(article_tokens, PaddedSequence):
            assert all([isinstance(x, PaddedSequence) for x in [I_tokens, C_tokens, O_tokens]])
        elif isinstance(article_tokens, torch.Tensor):
            # TODO test this codepath
            assert all([isinstance(x, torch.Tensor) for x in [I_tokens, C_tokens, O_tokens]]) and all([x.shape[0] == 1 for x in [article_tokens, I_tokens, C_tokens, O_tokens]])
        else:
            raise ValueError("Got an unexpected type for our input tensor: {}".format(type(article_tokens)))

        ##################################################
        # First encode the I, C, O frame (the query)     #
        ##################################################
        # the output of each of these should be of shape (batch x word_embedding_size)
        I_v, C_v, O_v = self._encode(I_tokens, C_tokens, O_tokens)

        if self.article_encoder.use_attention:

            query_v = None
            if self.article_encoder.condition_attention:
                query_v = torch.cat([I_v, C_v, O_v], dim=1)

            _, a_v, attn_weights = self.article_encoder(article_tokens, query_v_for_attention=query_v)

            # @TODO return to debugging/inspecting attention
            if verbose_attn:
                attn_weights = attn_weights.data.cpu().numpy()
                for i in range(batch_size):
                    attn_weights_slice = attn_weights[i][:article_tokens.batch_sizes[i].item()].squeeze()
                    sorted_idx = np.argsort(attn_weights_slice)
                    # hack
                    if sorted_idx.size == 1:
                        continue
                    length = len(attn_weights_slice)
                    top_words = [self.vectorizer.idx_to_str[article_tokens.data[i][idx]] for idx in sorted_idx[max(-20, -1 * length):]]
                    top_words.reverse()
                    top_words_weights = [attn_weights_slice[idx] for idx in sorted_idx[max(-20, -1 * length):]]
                    top_words_weights.reverse()
                    bottom_words = [self.vectorizer.idx_to_str[article_tokens.data[i][idx]] for idx in sorted_idx[:min(20, length)]]
                    bottom_words.reverse()
                    bottom_words_weights = [attn_weights_slice[idx] for idx in sorted_idx[:min(20, length)]]
                    bottom_words_weights.reverse()

                    def tokens_to_str(tokens):
                        return ", ".join([self.vectorizer.idx_to_str[x.item()] for x in tokens])
                    print("I, C, O frame:",
                          tokens_to_str(I_tokens.data[i][:I_tokens.batch_sizes[i]]), ";",
                          tokens_to_str(C_tokens.data[i][:C_tokens.batch_sizes[i]]), ":",
                          tokens_to_str(O_tokens.data[i][:O_tokens.batch_sizes[i]]))
                    print("top words:", ", ".join(top_words))
                    print("weights:", ", ".join(str(x) for x in top_words_weights))
                    print("bottom words:", ", ".join(bottom_words))
                    print("weights:", ", ".join(str(x) for x in bottom_words_weights))

        else:
            if self.article_encoder_type == 'CBoW':
                # TODO implement attention for the CBoW model
                a_v = self.article_encoder(article_tokens)
            elif self.article_encoder_type == 'GRU' or self.article_encoder_type == 'biGRU':
                _, a_v = self.article_encoder(article_tokens)
            else:
                raise ValueError("Unknown article encoder type {}".format(self.article_encoder_type))

        # TODO document this
        if len(a_v.size()) == 3:
            a_v = a_v.squeeze(0)
        h = torch.cat([a_v, I_v, C_v, O_v], dim=1)
        h = self.dropout(h)
        raw_out = self.out(self.MLP_hidden(h))

        return F.softmax(raw_out, dim=1)

    @classmethod
    def init_word_vectors(cls, path_to_wvs, vectorizer, use_cuda=USE_CUDA) -> nn.Embedding:
        WVs = KeyedVectors.load_word2vec_format(path_to_wvs, binary=True)

        E = np.zeros((len(vectorizer.str_to_idx), WVs.vector_size))
        WV_matrix = np.matrix([WVs[v] for v in WVs.vocab.keys()])
        mean_vector = np.mean(WV_matrix, axis=0)

        for idx, token in enumerate(vectorizer.idx_to_str):
            if token in WVs:
                E[idx] = WVs[token]
            else:
                E[idx] = mean_vector
        # TODO make this cleaner
        padding_idx = int(vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
        E[padding_idx] = torch.zeros(E.shape[1])
        embedding = nn.Embedding(E.shape[0], E.shape[1], padding_idx=padding_idx)
        embedding.weight.data.copy_(torch.from_numpy(E))
        embedding.weight.requires_grad = False
        if use_cuda:
            embedding = embedding.cuda()
        return embedding


def _get_y_vec(y_dict, as_vec=True, majority_lbl=True) -> torch.LongTensor:
    # +1 because raw labels are -1, 0, 1 -> 0, 1, 2
    # for indexing reasons that appear in the loss function
    # (cross-entropy loss wants the index of the highest value, and we index at 0)
    all_labels = [y_j[0] + 1 for y_j in y_dict]
    if majority_lbl:
        y_collapsed = int(stats.mode(all_labels)[0][0])
    else:
        y_collapsed = random.choice(all_labels)

    if as_vec:
        y_vec = np.zeros(3)
        y_vec[y_collapsed] = 1.0
        ret = torch.LongTensor(y_vec)
    else:
        ret = torch.LongTensor([y_collapsed])
    if USE_CUDA:
        ret = ret.cuda()
    return ret


def _to_torch_var(x):
    var_x = Variable(torch.LongTensor(x))
    if USE_CUDA:
        var_x = var_x.cuda()
    return var_x


def predict_for_inst(nnet, inst, verbose_attn=False):
    abstract = _to_torch_var(inst["article"]).unsqueeze(0)
    I, C, O = _to_torch_var(inst["I"]).unsqueeze(0), _to_torch_var(inst["C"]).unsqueeze(0), _to_torch_var(inst["O"]).unsqueeze(0)
    print("sizes:", abstract.size(), I.size(), C.size(), O.size())
    y_hat = nnet(abstract, I, C, O, batch_size=1, verbose_attn=verbose_attn)
    return y_hat


'''
def conf_matrix(nnet, instances):
    M = np.zeros((3,3))
    for inst in instances:
        y = _get_y_vec(inst['y'], as_vec=False)
        y_hat = np.argmax(predict_for_inst(nnet, inst))
        M[y, y_hat] += 1.0
    return M
'''


def make_preds(nnet, instances, batch_size, inference_vectorizer, verbose_attn_to_batches=False, cuda=USE_CUDA):
    # TODO consider removing the inference_vectorizer since all we need is an unk_idx from it
    y_vec = torch.cat([_get_y_vec(inst['y'], as_vec=False) for inst in instances]).squeeze()
    unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
    y_hat_vec = []
    # we batch this so the GPU doesn't run out of memory
    nnet.eval()
    for i in range(0, len(instances), batch_size):
        batch_instances = instances[i:i+batch_size]
        articles, Is, Cs, Os = [PaddedSequence.autopad([torch.LongTensor(inst[x]) for inst in batch_instances], batch_first=True, padding_value=unk_idx) for x in ['article', 'I', 'C', 'O']]
        if cuda:
            articles, Is, Cs, Os = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda()
        verbose_attn = verbose_attn_to_batches and i in verbose_attn_to_batches
        y_hat_batch = nnet(articles, Is, Cs, Os, batch_size=len(batch_instances), verbose_attn=verbose_attn)
        y_hat_vec.append(y_hat_batch)
    nnet.train()
    return y_vec, torch.cat(y_hat_vec, dim=0)


def to_int_preds(y):
    # the cast to int is necessary as this gets passed to sklearn packages that don't understand numpy.int64, which is the default return type here.
    return [int(np.argmax(y_i)) for y_i in y.cpu()]


def _loss_for_inst(inst, nnet, criterion):
    y = _get_y_vec(inst['y'], as_vec=False).squeeze()
    y_hat = predict_for_inst(nnet, inst)
    ####
    # as per https://github.com/pytorch/pytorch/issues/5554, 
    # output needs to have dims (N, C), so we add an extra
    # dim for N here (just 1).
    y_hat = torch.unsqueeze(y_hat, dim=0)
    if USE_CUDA:
        y_hat = y_hat.cuda()
        y = y.cuda()

    return criterion(y_hat, y)


def _get_majority_label(inst):
    all_lbls = [y[0] + 1 for y in inst['y']]
    return stats.mode(all_lbls)[0][0]


def train(ev_inf: InferenceNet, train_Xy, val_Xy, test_Xy, inference_vectorizer, epochs=10, batch_size=16, shuffle=True):
    # we sort these so batches all have approximately the same length (ish), which decreases the 
    # average amount of padding needed, and thus total number of steps in training.
    if not shuffle:
        train_Xy.sort(key=lambda x: len(x['article']))
        val_Xy.sort(key=lambda x: len(x['article']))
        test_Xy.sort(key=lambda x: len(x['article']))
    print("Using {} training examples, {} validation examples, {} testing examples".format(len(train_Xy), len(val_Xy), len(test_Xy)))
    most_common = stats.mode([_get_majority_label(inst) for inst in train_Xy])[0][0]

    best_val_model = None
    best_val_f1 = float('-inf')
    if USE_CUDA:
        ev_inf = ev_inf.cuda()

    optimizer = optim.Adam(ev_inf.parameters())
    criterion = nn.CrossEntropyLoss(reduction='sum')  # sum (not average) of the batch losses.

    # TODO add epoch timing information here
    epochs_since_improvement = 0
    val_metrics = {
        "val_acc": [],
        "val_p": [],
        "val_r": [],
        "val_f1": [],
        "val_loss": [],
        'train_loss': [],
        'val_aucs': [],
        'train_aucs': [],
        'val_entropies': [],
        'val_evidence_token_mass': [],
        'val_evidence_token_err': [],
        'train_entropies': [],
        'train_evidence_token_mass': [],
        'train_evidence_token_err': []
    }
    for epoch in range(epochs):
        if epochs_since_improvement > 10:
            print("Exiting early due to no improvement on validation after 10 epochs.")
            break
        if shuffle:
            random.shuffle(train_Xy)

        epoch_loss = 0
        for i in range(0, len(train_Xy), batch_size):
            instances = train_Xy[i:i+batch_size]
            ys = torch.cat([_get_y_vec(inst['y'], as_vec=False) for inst in instances], dim=0)
            # TODO explain the use of padding here
            unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
            articles, Is, Cs, Os = [PaddedSequence.autopad([torch.LongTensor(inst[x]) for inst in instances], batch_first=True, padding_value=unk_idx) for x in ['article', 'I', 'C', 'O']]
            optimizer.zero_grad()
            if USE_CUDA:
                articles, Is, Cs, Os = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda()
                ys = ys.cuda()
            verbose_attn = (epoch == epochs - 1 and i == 0) or (epoch == 0 and i == 0)
            if verbose_attn:
                print("Training attentions:")
            tags = ev_inf(articles, Is, Cs, Os, batch_size=len(instances), verbose_attn=verbose_attn)
            loss = criterion(tags, ys)
            #if loss.item() != loss.item():
            #    import pdb; pdb.set_trace()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        val_metrics['train_loss'].append(epoch_loss)

        with torch.no_grad():
            verbose_attn_to_batches = set([0,1,2,3,4]) if epoch == epochs - 1 or epoch == 0 else False
            if verbose_attn_to_batches:
                print("Validation attention:")
            # make_preds runs in eval mode
            val_y, val_y_hat = make_preds(ev_inf, val_Xy, batch_size, inference_vectorizer, verbose_attn_to_batches=verbose_attn_to_batches)
            val_loss = criterion(val_y_hat, val_y.squeeze())
            y_hat = to_int_preds(val_y_hat)

            if epoch == 0:
                dummy_preds = [most_common] * len(val_y)
                dummy_acc = accuracy_score(val_y.cpu(), dummy_preds)
                val_metrics["baseline_val_acc"] = dummy_acc
                p, r, f1, _ = precision_recall_fscore_support(val_y.cpu(), dummy_preds, labels=None, beta=1, average='macro', pos_label=1, warn_for=('f-score',), sample_weight=None)
                val_metrics['p_dummy'] = p
                val_metrics['r_dummy'] = r
                val_metrics['f_dummy'] = f1

                print("val dummy accuracy: {:.3f}".format(dummy_acc))
                print("classification report for dummy on val: ")
                print(classification_report(val_y.cpu(), dummy_preds))
                print("\n\n")

            acc = accuracy_score(val_y.cpu(), y_hat)
            val_metrics["val_acc"].append(acc)
            val_loss = val_loss.cpu().item()
            val_metrics["val_loss"].append(val_loss)
           
            # f1 = f1_score(val_y, y_hat, average="macro")
            p, r, f1, _ = precision_recall_fscore_support(val_y.cpu(), y_hat, labels=None, beta=1, average='macro', pos_label=1, warn_for=('f-score',), sample_weight=None)
            val_metrics["val_f1"].append(f1)
            val_metrics["val_p"].append(p)
            val_metrics["val_r"].append(r)

            if ev_inf.article_encoder.use_attention:
                train_auc, train_entropies, train_evidence_token_masses, train_evidence_token_err = evaluate_model_attention_distribution(ev_inf, train_Xy, cuda=USE_CUDA, compute_attention_diagnostics=True)
                val_auc, val_entropies, val_evidence_token_masses, val_evidence_token_err = evaluate_model_attention_distribution(ev_inf, val_Xy, cuda=USE_CUDA, compute_attention_diagnostics=True)
                print("train auc: {:.3f}, entropy: {:.3f}, evidence mass: {:.3f}, err: {:.3f}".format(train_auc, train_entropies, train_evidence_token_masses, train_evidence_token_err))
                print("val auc: {:.3f}, entropy: {:.3f}, evidence mass: {:.3f}, err: {:.3f}".format(val_auc, val_entropies, val_evidence_token_masses, val_evidence_token_err))
            else:
                train_auc, train_entropies, train_evidence_token_masses, train_evidence_token_err = "", "", "", ""
                val_auc, val_entropies, val_evidence_token_masses, val_evidence_token_err = "", "", "", ""
            val_metrics['train_aucs'].append(train_auc)
            val_metrics['train_entropies'].append(train_entropies)
            val_metrics['train_evidence_token_mass'].append(train_evidence_token_masses)
            val_metrics['train_evidence_token_err'].append(train_evidence_token_err)
            val_metrics['val_aucs'].append(val_auc)
            val_metrics['val_entropies'].append(val_entropies)
            val_metrics['val_evidence_token_mass'].append(val_evidence_token_masses)
            val_metrics['val_evidence_token_err'].append(val_evidence_token_err)
            if f1 > best_val_f1:
                print("New best model at {} with val f1 {:.3f}".format(epoch, f1))
                best_val_f1 = f1
                best_val_model = copy.deepcopy(ev_inf)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            #if val_loss != val_loss or epoch_loss != epoch_loss:
            #    import pdb; pdb.set_trace()

            print("epoch {}. train loss: {}; val loss: {}; val acc: {:.3f}".format(
                epoch, epoch_loss, val_loss, acc))
       
            print(classification_report(val_y.cpu(), y_hat))
            print("val macro f1: {0:.3f}".format(f1))
            print("\n\n")

    val_metrics['best_val_f1'] = best_val_f1
    with torch.no_grad():
        print("Test attentions:")
        verbose_attn_to_batches = set([0,1,2,3,4])
        # make_preds runs in eval mode
        test_y, test_y_hat = make_preds(best_val_model, test_Xy, batch_size, inference_vectorizer, verbose_attn_to_batches=verbose_attn_to_batches)
        test_loss = criterion(test_y_hat, test_y.squeeze())
        y_hat = to_int_preds(test_y_hat)
        final_test_preds = zip([t['a_id'] for t in test_Xy], [t['p_id'] for t in test_Xy], y_hat)

        acc = accuracy_score(test_y.cpu(), y_hat)
        val_metrics["test_acc"] = acc
        test_loss = test_loss.cpu().item()
        val_metrics["test_loss"] = test_loss

        # f1 = f1_score(test_y, y_hat, average="macro")
        p, r, f1, _ = precision_recall_fscore_support(test_y.cpu(), y_hat, labels=None, beta=1, average='macro', pos_label=1, warn_for=('f-score',), sample_weight=None)
        val_metrics["test_f1"] = f1
        val_metrics["test_p"] = p
        val_metrics["test_r"] = r
        if ev_inf.article_encoder.use_attention:
            test_auc, test_entropies, test_evidence_token_masses, test_evidence_token_err = evaluate_model_attention_distribution(best_val_model, test_Xy, cuda=USE_CUDA, compute_attention_diagnostics=True)
            print("test auc: {:.3f}, , entropy: {:.3f}, kl_to_uniform {:.3f}".format(test_auc, test_entropies, test_evidence_token_masses))
        else:
            test_auc, test_entropies, test_evidence_token_masses, test_evidence_token_err = "", "", "", ""
        val_metrics['test_auc'] = test_auc
        val_metrics['test_entropy'] = test_entropies
        val_metrics['test_evidence_token_mass'] = test_evidence_token_masses
        val_metrics['test_evidence_token_err'] = test_evidence_token_err

        print("test loss: {}; test acc: {:.3f}".format(test_loss, acc))

        print(classification_report(test_y.cpu(), y_hat))
        print("test macro f1: {}".format(f1))
        print("\n\n")

    return best_val_model, inference_vectorizer, train_Xy, val_Xy, val_metrics, final_test_preds
