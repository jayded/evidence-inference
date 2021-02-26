from collections import defaultdict
from os.path import join, dirname, abspath

import copy
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

import random

import torch
import torch.nn as nn

import numpy as np
from scipy.stats import entropy, sem
from sklearn.metrics import roc_auc_score

from evidence_inference.models.utils import PaddedSequence
from evidence_inference.preprocess.preprocessor import SimpleInferenceVectorizer

USE_CUDA = True

class TokenAttention(nn.Module):

    def __init__(self, encoding_size, query_dims=0, condition_attention=False, tokenwise_attention=False):
        super(TokenAttention, self).__init__()
        self.condition_attention = condition_attention
        if condition_attention:
            self.attn_MLP_hidden_dims = 32
            self.attn_input_dims = encoding_size + query_dims
            self.token_attention_F = nn.Sequential(
                nn.Linear(self.attn_input_dims, self.attn_MLP_hidden_dims),
                nn.Tanh(),
                nn.Linear(self.attn_MLP_hidden_dims, 1))
        else:
            self.token_attention_F = nn.Linear(encoding_size, 1)
            # the code below allows for an approximate comparison of the impact of the *number* of dimensions allowed for input to the MLP. We do this my concatenating two linear layers (equivalent to a single linear transform, but it has more parameters), the first one upscaling the number of input parameters to be comparable to the conditional case. We found that this didn't have a real impact in pretraining attention, which is the result you'd expect.
            #self.attn_MLP_hidden_dims = 32
            #self.attn_input_dims = encoding_size

            #self.token_attention_F = nn.Sequential(
            #    nn.Linear(self.attn_input_dims, self.attn_input_dims + self.query_dims),
            #    nn.Linear(self.attn_input_dims + self.query_dims, self.attn_MLP_hidden_dims),
            #    nn.Tanh(),
            #    nn.Linear(self.attn_MLP_hidden_dims, 1))

            # the code below allows for an approximate comparison of the impact of the *number* of dimensions allowed for input to the MLP. since we do this by allocating more to the hidden states, we split the difference in terms of weights/biases to being before + after the non-linearity. We found that, in the best case, this did closely to the pretraining condition.
            #self.attn_MLP_hidden_dims = 32 + int(0.5 * query_dims)
            #self.attn_input_dims = encoding_size

            #self.token_attention_F = nn.Sequential(
            #    nn.Linear(self.attn_input_dims, self.attn_MLP_hidden_dims),
            #    nn.Tanh(),
            #    nn.Linear(self.attn_MLP_hidden_dims, 1))

        if tokenwise_attention:
            self.attn_sm = nn.Sigmoid()
        else:
            self.attn_sm = nn.Softmax(dim=1)

    def forward(self, hidden_input_states: PaddedSequence, query_v_for_attention, normalize=True):
        if not isinstance(hidden_input_states, PaddedSequence):
            raise TypeError("Expected an input of type PaddedSequence but got {}".format(type(hidden_input_states)))
        if self.condition_attention:
            # the code below concatenates the query_v_for_attention (for a unit in the batch to each of the hidden states in the encoder)
            # expand the query vector used for attention by making it |batch|x1x|query_vector_size|
            query_v_for_attention = query_v_for_attention.unsqueeze(dim=1)
            # duplicate it to be the same number of (max) tokens in the batch
            query_v_for_attention = torch.cat(hidden_input_states.data.size()[1] * [query_v_for_attention], dim=1)
            # finally, concatenate this vector to every "final" element of the input tensor
            attention_inputs = torch.cat([hidden_input_states.data, query_v_for_attention], dim=2)
        else:
            attention_inputs = hidden_input_states.data
        raw_word_scores = self.token_attention_F(attention_inputs)
        raw_word_scores = raw_word_scores * hidden_input_states.mask(on=1.0, off=0.0, size=raw_word_scores.size(), device=raw_word_scores.device)
        # TODO this should probably become a logsumexp depending on condition
        a = self.attn_sm(raw_word_scores)

        # since we need to handle masking, we have to kill any support out of the softmax
        masked_attention = a * hidden_input_states.mask(on=1.0, off=0.0, size=a.size(), device=a.device)
        if normalize:
            # divide by the batch length here so we reduce the variance of the input to the next layer. this is only necessary for the tokenwise attention because its sum isn't constrained
            # a = masked_attention / word_inputs.batch_sizes.unsqueeze(-1).unsqueeze(-1).float()
            weights = torch.sum(masked_attention, dim=1).unsqueeze(1)
            a = masked_attention / weights
        else:
            a = masked_attention

        return a


# noinspection PyPep8Naming
def prepare_article_attention_target(model, batch_instances, cuda):
    unk_idx = int(model.vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
    articles, Is, Cs, Os = [PaddedSequence.autopad([torch.LongTensor(inst[x]) for inst in batch_instances], batch_first=True, padding_value=unk_idx) for x in ['article', 'I', 'C', 'O']]
    target_spans = [inst['evidence_spans'] for inst in batch_instances]
    target = [torch.zeros(len(x['article'])) for x in batch_instances]
    for tgt, spans in zip(target, target_spans):
        for start, end in spans:
            tgt[start:end] = 1
    target = PaddedSequence.autopad(target, batch_first=True, padding_value=0)
    if cuda:
        articles, Is, Cs, Os, target = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda(), target.cuda()
    return articles, Is, Cs, Os, target


def _fetch_random_span(banned_start, banned_end, upper_range, length):
    start = random.randint(0, upper_range)
    end = start + length
    while (banned_start - start < 0 < banned_end - start) or (banned_start - end < 0 < banned_end - end) or end > upper_range:
        start = random.randint(0, upper_range)
        end = start + length
    return start, end


# noinspection PyPep8Naming
def prepare_article_attention_target_balanced(model, batch_instances, cuda):
    unk_idx = int(model.vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
    Is = []
    Cs = []
    Os = []
    articles = []
    target = []
    for inst in batch_instances:
        i = torch.LongTensor(inst['I'])
        c = torch.LongTensor(inst['C'])
        o = torch.LongTensor(inst['O'])
        article = torch.LongTensor(inst['article'])
        target_spans = set([tuple(x) for x in inst['evidence_spans']])
        for start, end in target_spans:
            # positive example
            Is.append(i)
            Cs.append(c)
            Os.append(o)
            articles.append(article[start:end])
            target.append(torch.ones(end - start))

            # negative example
            neg_start, neg_end = _fetch_random_span(start, end, len(article), end - start)
            Is.append(i)
            Cs.append(c)
            Os.append(o)
            articles.append(article[neg_start:neg_end])
            target.append(torch.zeros(neg_end - neg_start))

    Is, Cs, Os, articles = [PaddedSequence.autopad(x, batch_first=True, padding_value=unk_idx) for x in [Is, Cs, Os, articles]]
    target = PaddedSequence.autopad(target, batch_first=True, padding_value=0)
    if cuda:
        articles, Is, Cs, Os, target = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda(), target.cuda()
    return articles, Is, Cs, Os, target


# noinspection PyPep8Naming
def _prepare_article_attention_distribution(model, batch_instances, cuda):
    unk_idx = int(model.vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
    articles, Is, Cs, Os = [PaddedSequence.autopad([torch.LongTensor(inst[x]) for inst in batch_instances], batch_first=True, padding_value=unk_idx) for x in ['article', 'I', 'C', 'O']]
    target_spans = [inst['evidence_spans'] for inst in batch_instances]
    target = [torch.zeros(len(x['article'])) for x in batch_instances]
    for tgt, spans in zip(target, target_spans):
        for start, end in spans:
            tgt[start:end] = 1
        tgt /= torch.sum(tgt)
    target = PaddedSequence.autopad(target, batch_first=True, padding_value=0)
    if cuda:
        articles, Is, Cs, Os, target = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda(), target.cuda()
    return articles, Is, Cs, Os, target


# noinspection PyPep8Naming
def get_article_attention_weights(model, article_tokens, Is, Cs, Os, tokenwise_attention=False):
    query_v = None
    if model.article_encoder.condition_attention:
        query_v = torch.cat(model._encode(Is, Cs, Os), dim=1)

    _, _, attn_weights = model.article_encoder(article_tokens, query_v_for_attention=query_v, normalize_attention_distribution=(not tokenwise_attention))
    return attn_weights


def remove_empty_evidence_spans(instance):
    if 'evidence_spans' in instance:
        spans = set(instance['evidence_spans'])
        result_spans = []
        for start, end in spans:
            if not (start == end or end <= start):
                result_spans.append((start, end))
        instance = dict(instance)
        if len(result_spans) > 0:
            instance['evidence_spans'] = result_spans
        else:
            del instance['evidence_spans']
        return instance
    else:
        return instance


# TODO document this monster
# TODO add some padding for the document
# noinspection PyPep8Naming
def _prepare_random_concatenated_spans(model, batch_instances, cuda):
    unk_idx = int(model.vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
    target_spans = [inst['evidence_spans'] for inst in batch_instances]
    Is = []
    Os = []
    Cs = []
    target = []
    articles = []
    for instance, evidence_spans in zip(batch_instances, target_spans):
        article = instance['article']
        article = torch.LongTensor(article)
        tgt = torch.zeros(len(article))
        Is.append(instance['I'])
        Os.append(instance['O'])
        Cs.append(instance['C'])
        for start, end in evidence_spans:
            tgt[start:end] = 1
        start, end = random.choice(evidence_spans)
        unacceptable_start = start - (end - start)
        unacceptable_end = end + (end - start)
        random_matched_span_start = random.randint(0, len(article))
        # rejection sample until we find an acceptable span start either inside or outside the document
        while unacceptable_start - random_matched_span_start < 0 and 0 < unacceptable_end - random_matched_span_start:
            random_matched_span_start = random.randint(0, len(article))
        random_matched_span = (random_matched_span_start, random_matched_span_start + end - start)
        if random.random() > 0.5:
            tgt = torch.cat([tgt[start:end], tgt[random_matched_span[0]:random_matched_span[1]]]).contiguous()
            article = torch.cat([article[start:end], article[random_matched_span[0]:random_matched_span[1]]]).contiguous()
        else:
            tgt = torch.cat([tgt[random_matched_span[0]:random_matched_span[1]], tgt[start:end]]).contiguous()
            article = torch.cat([article[random_matched_span[0]:random_matched_span[1]], article[start:end]]).contiguous()
        tgt /= torch.sum(tgt)
        target.append(tgt)
        articles.append(article)

    Is, Cs, Os = [PaddedSequence.autopad([torch.LongTensor(elem) for elem in cond], batch_first=True, padding_value=unk_idx) for cond in [Is, Cs, Os]]
    target = PaddedSequence.autopad(target, batch_first=True, padding_value=0)
    articles = PaddedSequence.autopad(articles, batch_first=True, padding_value=unk_idx)
    if cuda:
        articles, Is, Cs, Os, target = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda(), target.cuda()
    return articles, Is, Cs, Os, target

# noinspection PyPep8Naming
def pretrain_max_evidence_attention(train_Xy, val_Xy, model, epochs=10, batch_size=16, cuda=True, tokenwise_attention=False, attention_acceptance='auc'):
    """ A pretraining variant for attention to maximize total mass on evidence tokens.

    This method uses the output of the attention distributions from InferenceNet
    and attempts to maximize the mass on the evidence span (minimize the mass on 
    anywhere else in the document). Therefore the output of the attention
    mechanism must be normalized.
    """
    if tokenwise_attention:
        raise ValueError("This attention distribution doesn't make sense when run in a tokenwise mode!")
    def max_evidence_loss(target, predicted):
        total_evidence_mass = torch.sum(target * predicted, dim=1)
        missing_mass = 1 - total_evidence_mass
        averaged_missing_mass = torch.sum(total_evidence_mass, dim=0) / target.size()[0]
        return averaged_missing_mass
    return pretrain_attention(train_Xy,
                              val_Xy,
                              model,
                              prepare=prepare_article_attention_target,
                              get_attention_weights=get_article_attention_weights,
                              criterion=max_evidence_loss,
                              epochs=epochs,
                              batch_size=batch_size,
                              cuda=cuda,
                              tokenwise_attention=tokenwise_attention,
                              attention_acceptance=attention_acceptance)

# noinspection PyPep8Naming
def pretrain_tokenwise_attention(train_Xy, val_Xy, model, epochs=10, batch_size=16, cuda=True, tokenwise_attention=False, attention_acceptance='auc'):
    """ A pretraining variant for attention to maximize the score of each evidence token, in its context in the document.

    This method uses the unnormalized output of the attention distributions from
    InferenceNet and attempts to maximize the score (sigmoid) of evidence tokens
    and minimize the score of non-evidence tokens.
    """
    if not tokenwise_attention:
        raise ValueError("This attention distribution doesn't make sense without tokenwise mode!")
    return pretrain_attention(train_Xy,
                              val_Xy,
                              model,
                              prepare=prepare_article_attention_target,
                              get_attention_weights=get_article_attention_weights,
                              criterion=torch.nn.BCELoss(reduction='sum'),
                              epochs=epochs,
                              batch_size=batch_size,
                              cuda=cuda,
                              tokenwise_attention=tokenwise_attention,
                              attention_acceptance=attention_acceptance)


# noinspection PyPep8Naming
def pretrain_tokenwise_attention_balanced(train_Xy, val_Xy, model, epochs=10, batch_size=16, cuda=True, tokenwise_attention=False, attention_acceptance='auc'):
    """ Similar to pretrain_tokenwise_attention, but takes a balanced approach for positive/negative classes.
     
    This method is identical to pretrain_tokenwise_attention except that it 
    randomly samples negative spans from the document and uses them as negative
    examples. These spans: (1) do not overlap with the evidence span, and (2)
    are the same length as the evidence span.
    """
    return pretrain_attention(train_Xy,
                              val_Xy,
                              model,
                              prepare=prepare_article_attention_target_balanced,
                              get_attention_weights=get_article_attention_weights,
                              criterion=torch.nn.BCELoss(reduction='sum'),
                              epochs=epochs,
                              batch_size=batch_size,
                              cuda=cuda,
                              tokenwise_attention=tokenwise_attention,
                              attention_acceptance=attention_acceptance)


# noinspection PyPep8Naming
def pretrain_attention_to_match_span(train_Xy, val_Xy, model, epochs=10, batch_size=16, cuda=True, tokenwise_attention=False, attention_acceptance='auc'):
    """ An attention pretraining variant similar to pretrain_tokenwise_attention

    This method is identical to pretrain_tokenwise_attention except that it
    attempts to work on a distribution level, i.e. the loss is applied on the
    normalized distribution.

    Note that this objective is rather weird and it's not actually possible to
    achieve zero loss on it.
    """
    return pretrain_attention(train_Xy,
                              val_Xy,
                              model,
                              prepare=_prepare_article_attention_distribution,
                              get_attention_weights=get_article_attention_weights,
                              criterion=torch.nn.BCELoss(reduction='sum'),
                              epochs=epochs,
                              batch_size=batch_size,
                              cuda=cuda,
                              tokenwise_attention=tokenwise_attention,
                              attention_acceptance=attention_acceptance)


# noinspection PyPep8Naming
def pretrain_attention_with_concatenated_spans(train_Xy, val_Xy, model, epochs=10, batch_size=16, cuda=True, tokenwise_attention=False, attention_acceptance='auc'):
    """ An attention pretraining variant similar to pretrain_attention_with_random_spans

    The main difference here is that we sample random spans of the same size
    of the evidence, and then concatenate this with the evidence (ordering is
    random).
    """
    return pretrain_attention(train_Xy,
                              val_Xy,
                              model,
                              prepare=_prepare_random_concatenated_spans,
                              get_attention_weights=get_article_attention_weights,
                              criterion=torch.nn.BCELoss(reduction='sum'),
                              epochs=epochs,
                              batch_size=batch_size,
                              cuda=cuda,
                              tokenwise_attention=tokenwise_attention,
                              attention_acceptance=attention_acceptance)


# noinspection PyPep8Naming
def pretrain_attention_with_random_spans(train_Xy, val_Xy, model, epochs=10, batch_size=16, cuda=True, tokenwise_attention=False, attention_acceptance='auc'):
    """ A pretraining variants that is balanced. This is similar to pretrain_tokenwise_attention_balanced
    
    The primary difference between this and pretrain_tokenwise_attention_balanced
    is that the loss function here is an MSELoss, so it uses the squared error
    between the attention mechanism's output (unnormalized if tokenwise) instead
    of using a BCELoss.
    """
    def _prepare_random_matched_spans(model, batch_instances, cuda):
        unk_idx = int(model.vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
        Is, Cs, Os = [PaddedSequence.autopad([torch.LongTensor(inst[x]) for inst in batch_instances], batch_first=True, padding_value=unk_idx) for x in ['I', 'C', 'O']]
        target_spans = [inst['evidence_spans'] for inst in batch_instances]
        target = []
        articles = []
        for article, evidence_spans in zip((x['article'] for x in batch_instances), target_spans):
            tgt = torch.zeros(len(article))
            for start, end in evidence_spans:
                tgt[start:end] = 1
            (start, end) = random.choice(evidence_spans)
            # select a random span of the same length
            random_matched_span_start = random.randint(0, len(article))
            random_matched_span_end = random_matched_span_start + end - start
            tgt_pos = tgt[start:end]
            tgt_neg = tgt[random_matched_span_start:random_matched_span_end]
            article_pos = torch.LongTensor(article[start:end])
            article_neg = torch.LongTensor(article[random_matched_span_start:random_matched_span_end])
            if random.random() > 0.5:
                articles.append(torch.cat([article_pos, article_neg]))
                target.append(torch.cat([tgt_pos, tgt_neg]))
            else:
                articles.append(torch.cat([article_neg, article_pos]))
                target.append(torch.cat([tgt_neg, tgt_pos]))

        target = PaddedSequence.autopad(target, batch_first=True, padding_value=0)
        articles = PaddedSequence.autopad(articles, batch_first=True, padding_value=unk_idx)
        if cuda:
            articles, Is, Cs, Os, target = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda(), target.cuda()
        return articles, Is, Cs, Os, target
    return pretrain_attention(train_Xy,
                              val_Xy,
                              model,
                              prepare=_prepare_random_matched_spans,
                              get_attention_weights=get_article_attention_weights,
                              criterion=torch.nn.MSELoss(reduction='sum'),
                              epochs=epochs,
                              batch_size=batch_size,
                              cuda=cuda,
                              tokenwise_attention=tokenwise_attention,
                              attention_acceptance=attention_acceptance)


# noinspection PyPep8Naming
def pretrain_attention(train_Xy, val_Xy, model, prepare, get_attention_weights, criterion, epochs=100, batch_size=16, cuda=True, tokenwise_attention=False, patience=10, attention_acceptance='auc'):
    if not model.article_encoder.use_attention:
        raise ValueError("Cannot pretrain attention for a model that doesn't use it!")

    if attention_acceptance == 'auc' or attention_acceptance == 'evidence_mass':
        best_score = float('-inf')
    elif attention_acceptance == 'entropy':
        best_score = float('inf')
    else:
        raise ValueError("Unknown attention acceptance metric {}".format(attention_acceptance))
    best_model = None

    epochs_since_improvement = 0
    metrics = {
        # note this loss is computed throughout the epoch, but the val loss is computed at the end of the epoch
        'attn_train_losses': [],
        'pretrain_attn_train_auc': [],
        'pretrain_attn_train_token_masses': [],
        'pretrain_attn_train_token_masses_err': [],
        'pretrain_attn_train_entropies': [],
        'attn_val_losses': [],  # TODO eventually rename this for consistency with the other metrics
        'pretrain_attn_val_auc_all': [],  # TODO eventually remove _all from this metric name
        'pretrain_attn_val_token_masses': [],
        'pretrain_attn_val_token_masses_err': [],
        'pretrain_attn_val_entropies': []
    }

    train_Xy = map(remove_empty_evidence_spans, train_Xy)
    train_Xy = list(filter(lambda x: 'evidence_spans' in x, train_Xy))
    train_Xy.sort(key=lambda x: len(x['article']), reverse=True)
    val_Xy = map(remove_empty_evidence_spans, val_Xy)
    val_Xy = list(filter(lambda x: 'evidence_spans' in x, val_Xy))
    val_Xy.sort(key=lambda x: len(x['article']), reverse=True)
    print("Pre-training attention distribution with {} training examples, {} validation examples".format(len(train_Xy), len(val_Xy)))

    # import pdb; pdb.set_trace()
    # TODO is this the right gradient function?
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(train_Xy), batch_size):

            instances = train_Xy[i:i+batch_size]
            articles, Is, Cs, Os, target = prepare(model, instances, cuda)
            attn_weights = get_attention_weights(model, articles, Is, Cs, Os, tokenwise_attention)

            optimizer.zero_grad()
            if not (torch.min(attn_weights >= 0).item() == 1 and torch.min(attn_weights <= 1.) == 1):
                # import pdb; pdb.set_trace()
                print("Error in weights")
            if not (torch.min(target.data >= 0).item() == 1 and torch.min(target.data <= 1.) == 1):
                # import pdb; pdb.set_trace()
                print("Error in weights")
            loss = criterion(attn_weights.squeeze(), target.data) / torch.sum(target.batch_sizes).float()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch {} Loss for pretrained attention (training set):".format(epoch), epoch_loss)
        metrics['attn_train_losses'].append(epoch_loss)
        with torch.no_grad():
            print("Epoch {} for pretraining attention:".format(epoch))
            train_auc_score, train_h, train_evidence_token_mass, train_etm_stderr = evaluate_model_attention_distribution(model, train_Xy, criterion=None, compute_attention_diagnostics=True)
            metrics['pretrain_attn_train_auc'].append(train_auc_score)
            metrics['pretrain_attn_train_entropies'].append(train_h)
            metrics['pretrain_attn_train_token_masses'].append(train_evidence_token_mass)
            metrics['pretrain_attn_train_token_masses_err'].append(train_etm_stderr)
            print("Pretraining attention training loss: {:.3F}, auc: {:.3F}, entropy: {:.3F}, token mass: {:.3F}, token mass err: {:.3F}".format(epoch_loss, train_auc_score, train_h, train_evidence_token_mass, train_etm_stderr))

            overall_val_auc, val_h, val_evidence_token_mass, val_etm_stderr, val_loss = evaluate_model_attention_distribution(model, val_Xy, criterion=criterion, compute_attention_diagnostics=True)
            metrics['pretrain_attn_val_auc_all'].append(overall_val_auc)
            metrics['pretrain_attn_val_entropies'].append(val_h)
            metrics['pretrain_attn_val_token_masses'].append(val_evidence_token_mass)
            metrics['pretrain_attn_val_token_masses_err'].append(val_etm_stderr)
            metrics['attn_val_losses'].append(val_loss)
            print("Pretraining attention validation loss: {:.3F}, auc: {:.3F}, entropy: {:.3F}, token mass: {:.3F}, token mass err: {:.3F}".format(val_loss, overall_val_auc, val_h, val_evidence_token_mass, val_etm_stderr))

            epochs_since_improvement += 1
            if attention_acceptance == 'auc':
                if overall_val_auc > best_score:
                    print("new best model at epoch {}".format(epoch))
                    best_score = overall_val_auc
                    best_model = copy.deepcopy(model)
                    epochs_since_improvement = 0
            elif attention_acceptance == 'evidence_mass':
                if val_evidence_token_mass > best_score:
                    print("new best model at epoch {}".format(epoch))
                    best_score = val_evidence_token_mass
                    best_model = copy.deepcopy(model)
                    epochs_since_improvement = 0
            elif attention_acceptance == 'entropy':
                if val_h < best_score:
                    print("new best model at epoch {}".format(epoch))
                    best_score = val_h
                    best_model = copy.deepcopy(model)
                    epochs_since_improvement = 0
            else:
                raise ValueError("Unknown attention acceptance criterion {}".format(attention_acceptance))

            if epochs_since_improvement > patience:
                print("Exiting early due to no improvement on validation after {} epochs.".format(patience))
                break

    return best_model, metrics


def evaluate_model_attention_distribution(model, Xy, batch_size=16, criterion=None, cuda=USE_CUDA, compute_attention_diagnostics=False):
    Xy = map(remove_empty_evidence_spans, Xy)
    Xy = list(filter(lambda x: 'evidence_spans' in x, Xy))
    Xy.sort(key=lambda x: len(x['article']), reverse=True)
    # generate a fixed set of validation instances
    Xy_batches = list()
    for i in range(0, len(Xy), batch_size):
        instances = Xy[i:min(i+batch_size, len(Xy))]
        articles, Is, Cs, Os, target = prepare_article_attention_target(model, instances, cuda=False)
        Xy_batches.append((articles, Is, Cs, Os, target))
    with torch.no_grad():
        all_tagged = []
        all_true = []
        entropies = []
        evidence_token_masses = []
        loss = 0
        for (articles, Is, Cs, Os, target) in Xy_batches:
            if cuda:
                articles, Is, Cs, Os, target = articles.cuda(), Is.cuda(), Cs.cuda(), Os.cuda(), target.cuda()
            attn_weights = get_article_attention_weights(model, articles, Is, Cs, Os)
            unpadded_weights = [x.cpu() for x in articles.unpad(attn_weights.squeeze(dim=2))]
            unpadded_truth = [x.cpu() for x in target.unpad(target.data)]

            all_tagged.extend(unpadded_weights)
            all_true.extend(unpadded_truth)
            if compute_attention_diagnostics:
                entropies.extend([entropy(x) for x in unpadded_weights])
                evidence_token_masses.extend([sum(x * y) for (x,y) in zip(unpadded_weights, unpadded_truth)])
                if criterion:
                    loss += (criterion(attn_weights.squeeze(), target.data) / torch.sum(target.batch_sizes).float()).item()
                if entropies[-1] != entropies[-1] or evidence_token_masses[-1] != evidence_token_masses[-1]:
                    import pdb; pdb.set_trace()

        all_tagged = torch.cat(all_tagged).view(-1, 1).squeeze().numpy()
        all_true = torch.cat(all_true).view(-1, 1).squeeze().numpy()
        try:
            # we can't measure AUC scores in the cheating case because everything is in class!
            auc_score = roc_auc_score(all_true, all_tagged)
        except:
            auc_score = float('nan')
        if compute_attention_diagnostics:
            h, etm, stderr = np.average(entropies), np.average(evidence_token_masses), sem(evidence_token_masses)
            if criterion:
                return auc_score, h, etm, stderr, loss
            else:
                return auc_score, h, etm, stderr
        else:
            return auc_score
