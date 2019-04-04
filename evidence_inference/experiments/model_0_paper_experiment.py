import argparse
import copy
import os
import random
import sys
from collections import namedtuple, defaultdict

import torch
from torch import nn
import numpy as np
import pandas as pd
from scipy import stats

import dill  

from os.path import abspath, dirname, join
# this monstrosity produces the module directory in an environment where this is unpacked

sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

from evidence_inference.models.model_0 import train, InferenceNet, PaddedSequence, USE_CUDA
from evidence_inference.models.model_scan import ScanNet
from evidence_inference.models.attention_distributions import pretrain_attention_to_match_span, pretrain_attention_with_concatenated_spans, pretrain_attention_with_random_spans, pretrain_tokenwise_attention, pretrain_tokenwise_attention_balanced, evaluate_model_attention_distribution, pretrain_max_evidence_attention
import evidence_inference.preprocess.preprocessor as preprocessor
from evidence_inference.preprocess.preprocessor import SimpleInferenceVectorizer as SimpleInferenceVectorizer
from evidence_inference.models.model_ico_scan import ScanNet as ScanNetICO


def identity(train_Xy, val_Xy, test_Xy, inference_vectorizer):
    return train_Xy, val_Xy, test_Xy

def scan_net_ICO_preprocess_create(loc):
    """ Function that returns a mangler type function. """
    
    def load_model_scan_ICO(inference_vectorizer):
        """ Load in the model (with proper weights). """
        sn = ScanNetICO(inference_vectorizer, use_attention=False)
        state = sn.state_dict()
        partial = torch.load(loc)
        state.update(partial)
        sn.load_state_dict(state)
        sn = sn.cuda()
        sn.eval()
        return sn
    
    def get_preds_ICO(model, span, I, C, O, inference_vectorizer):
        """ Get a prediction from the model for a single span. """
        if len(span) == 0:
            # if we happen to get an empty span, predict 0.
            return 0 
        unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
        sentences = [torch.LongTensor(span)]
        I = [torch.LongTensor(I)]
        C = [torch.LongTensor(C)]
        O = [torch.LongTensor(O)]
        sens, I, C, O = [PaddedSequence.autopad(to_enc, batch_first=True, padding_value=unk_idx) for to_enc in [sentences, I, C, O]]
        sens, I, C, O = sens.cuda(), I.cuda(), C.cuda(), O.cuda()
        preds = model(sens, I, C, O)
        pred = preds[0].data.tolist()[0]
        return pred
    
    def reformat_ICO(Xy, model, inference_vectorizer, sent_out_prefix=""):
        """ Given an Xy, parse through and reset the article. """

        all_str = []
        for prompt in Xy:
            sen = prompt["sentence_span"]
            I   = prompt["I"]
            C   = prompt["C"]
            O   = prompt["O"]
            new_article = []
            best_pred = 0
            back_up_sentence = []
            for s in sen:
                pred = get_preds_ICO(model, s[0], I, C, O, inference_vectorizer)
                
                # if no preds > .5
                if best_pred < pred:
                    best_pred = pred
                    back_up_sentence = s[0]
                    
                # if it is greater than .5, mark as evidence.
                if (pred > .5):
                    str_v = " ".join([inference_vectorizer.idx_to_str[word] for word in s[0]])
                    all_str.append([prompt["p_id"], pred, str_v])
                    new_article = np.append(new_article, s[0])
                    
            if len(new_article) == 0:
                new_article = back_up_sentence
                
            prompt['article'] = new_article    
        
    def scan_net_preprocess_ICO(train_Xy, val_Xy, test_Xy, inference_vectorizer):
        """ Takes the unprocessed data and reformats it. """
        sn = load_model_scan_ICO(inference_vectorizer)
        reformat_ICO(train_Xy, sn, inference_vectorizer, sent_out_prefix="train")
        reformat_ICO(val_Xy, sn, inference_vectorizer, sent_out_prefix="val")
        reformat_ICO(test_Xy, sn, inference_vectorizer, sent_out_prefix="test")
        
        return train_Xy, val_Xy, test_Xy
        
    return scan_net_preprocess_ICO


def scan_net_preprocess_create(loc):
    """ Function that returns a mangler type function. """

    def load_model_scan(inference_vectorizer):
        """ Load in the model (with proper weights). """
        sn = ScanNet(inference_vectorizer, use_attention=False)
        state = sn.state_dict()
        partial = torch.load(loc)
        state.update(partial)
        sn.load_state_dict(state)
        sn = sn.cuda()
        sn.eval()
        return sn

    def get_preds(model, span, inference_vectorizer):
        """ Get a prediction from the model for a single span. """
        if len(span) == 0:
            # if we happen to get an empty span, predict 0.
            return 0
        batch_instances = [span]
        unk_idx = int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])
        sentences = [torch.LongTensor(inst) for inst in batch_instances]
        sens, = [PaddedSequence.autopad(sentences, batch_first=True, padding_value=unk_idx)]
        sens = sens.cuda()
        preds = model(sens, batch_size=len(sentences))
        pred = preds[0].data.tolist()[0]
        return pred

    def reformat(Xy, model, inference_vectorizer, sent_out_prefix=""):
        """ 
        Given an array of pairs (Xs and ys), reset the Xs to reflect the predictions
        of the first part of the scan_net predictions and reset the article. 
        """

        all_str = []
        for prompt in Xy:
            sen = prompt["sentence_span"]
            new_article = []
            for s in sen:
                pred = get_preds(model, s[0], inference_vectorizer)

                # if it is greater than .5, mark as evidence.
                if pred > .5:
                    str_v = " ".join([inference_vectorizer.idx_to_str[word] for word in s[0]])
                    all_str.append([prompt["a_id"], pred, str_v])
                    new_article = np.append(new_article, s[0])

            prompt['new_article'] = new_article

        df_a = pd.DataFrame(data=all_str, columns=["Article id", "Prediction value", "Sentences"])
        df_a.fillna("")
        df_a.to_csv('./' + sent_out_prefix + '_evidence_sentences.csv', index = False, encoding = 'utf-8')

    def scan_net_preprocess(train_Xy, val_Xy, test_Xy, inference_vectorizer):
        """ Takes the unprocessed data and reformats it. """
        sn = load_model_scan(inference_vectorizer)
        reformat(train_Xy, sn, inference_vectorizer, sent_out_prefix="train")
        reformat(val_Xy, sn, inference_vectorizer, sent_out_prefix="val")
        reformat(test_Xy, sn, inference_vectorizer, sent_out_prefix="test")

        return train_Xy, val_Xy, test_Xy

    return scan_net_preprocess


def replace_articles_with_evidence_spans(train_Xy, val_Xy, test_Xy, inference_vectorizer):
    def replace(instances):
        ret = []
        for inst in instances:
            inst = copy.deepcopy(inst)
            all_evidences = list(filter(lambda x: type(x) is str and len(x) > 0, set([x[1] for x in inst['y']])))
            if len(all_evidences) == 0:
                continue
            evidence = stats.mode(all_evidences)[0][0]
            inst['article'] = inference_vectorizer.string_to_seq(evidence)
            if len(evidence) == 0 or len(inst['article']) == 0:
                continue
            ret.append(inst)
        return ret
    return replace(train_Xy), replace(val_Xy), replace(test_Xy)


def replace_articles_with_empty(train_Xy, val_Xy, test_Xy, inference_vectorizer):
    def replace(instances):
        ret = []
        for inst in instances:
            inst = copy.deepcopy(inst)
            inst['article'] = [int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])]
            ret.append(inst)
        return ret
    return replace(train_Xy), replace(val_Xy), replace(test_Xy)


def replace_prompts_with_empty(train_Xy, val_Xy, test_Xy, inference_vectorizer):
    def replace(instances):
        ret = []
        for inst in instances:
            inst = copy.deepcopy(inst)
            inst['I'] = [int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])]
            inst['O'] = [int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])]
            inst['C'] = [int(inference_vectorizer.str_to_idx[SimpleInferenceVectorizer.PAD])]
            ret.append(inst)
        return ret
    return replace(train_Xy), replace(val_Xy), replace(test_Xy)


def double_training_trick(train_Xy, val_Xy, test_Xy, inference_vectorizer):
    def double(instances):
        ret = []
        for inst in instances:
            original = inst
            ret.append(original)
            inst = copy.deepcopy(inst)
            inst['I'], inst['C'] = original['C'], original['I']
            new_ys = []
            for y in original['y']:
                # we flip the sign here since at this point the inputs are -1, 0, 1; 0 stays the same, the others flip
                new_y = (y[0] * -1, *y[1:])
                new_ys.append(new_y)
            inst['y'] = new_ys
            ret.append(inst)
        return ret
    return double(train_Xy), val_Xy, test_Xy


def get_data(sections_of_interest=None, mode='experiment', include_sentence_span_splits = False):
    random.seed(177)
    if mode == 'experiment':
        # raise ValueError('implement me!')
        train_docs = list(preprocessor.train_document_ids())
        random.shuffle(train_docs)
        split_index = int(len(train_docs) * .9)
        real_train_docs = train_docs[:split_index]
        real_val_docs = train_docs[split_index:]
        parent_path = abspath(os.path.join(dirname(abspath(__file__)), '..', '..'))
        vocab_f = os.path.join(parent_path, "annotations", "vocab.txt")
        real_train_Xy, inference_vectorizer = preprocessor.get_train_Xy(set(real_train_docs), sections_of_interest=sections_of_interest, vocabulary_file=vocab_f, include_sentence_span_splits = include_sentence_span_splits)
        real_val_Xy = preprocessor.get_Xy(set(real_val_docs), inference_vectorizer, sections_of_interest=sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)

        # in development, our "test" set is our validation ids so we don't cheat.
        real_test_Xy = preprocessor.get_Xy(preprocessor.validation_document_ids(), inference_vectorizer, sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)
        return real_train_Xy, real_val_Xy, real_test_Xy, inference_vectorizer
    elif mode == 'paper':
        parent_path = abspath(os.path.join(dirname(abspath(__file__)), '..', '..'))
        vocab_f = os.path.join(parent_path, "annotations", "vocab.txt")
        train_docs = preprocessor.train_document_ids()
        train_Xy, inference_vectorizer = preprocessor.get_train_Xy(train_docs, sections_of_interest=sections_of_interest, vocabulary_file=vocab_f, include_sentence_span_splits = include_sentence_span_splits)
        val_Xy = preprocessor.get_Xy(preprocessor.validation_document_ids(), inference_vectorizer, sections_of_interest=sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)
        test_Xy = preprocessor.get_Xy(preprocessor.test_document_ids(), inference_vectorizer, sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)
        return train_Xy, val_Xy, test_Xy, inference_vectorizer
    elif mode == 'minimal':
        parent_path = abspath(os.path.join(dirname(abspath(__file__)), '..', '..'))
        vocab_f = os.path.join(parent_path, "annotations", "vocab.txt")
        train_docs = list(preprocessor.train_document_ids())[:5]
        train_Xy, inference_vectorizer = preprocessor.get_train_Xy(train_docs, sections_of_interest=sections_of_interest, vocabulary_file=vocab_f, include_sentence_span_splits = include_sentence_span_splits)
        val_Xy = preprocessor.get_Xy(list(preprocessor.validation_document_ids())[:5], inference_vectorizer, sections_of_interest=sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)
        test_Xy = preprocessor.get_Xy(list(preprocessor.validation_document_ids())[5:10], inference_vectorizer, sections_of_interest, include_sentence_span_splits = include_sentence_span_splits)
        return train_Xy, val_Xy, test_Xy, inference_vectorizer
    else:
        raise ValueError('implement me!')


# noinspection PyPep8Naming
def run(real_train_Xy, real_val_Xy, real_test_Xy, inference_vectorizer, mangle_method, config, cuda=USE_CUDA, determinize=False):
    random.seed(177)
    if determinize:
        torch.manual_seed(360)
        torch.backends.cudnn.deterministic = True
        np.random.seed(2115)
    shuffle = False
    print("Running config {}".format(config))
    if config.no_pretrained_word_embeddings:
        num_embeddings = len(inference_vectorizer.idx_to_str)
        embedding_dim = 200
        init_word_embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=inference_vectorizer.str_to_idx[inference_vectorizer.PAD], _weight=torch.FloatTensor((num_embeddings, embedding_dim)))
    else:
        init_word_embeddings = None
    initial_model = InferenceNet(inference_vectorizer, ICO_encoder=config.ico_encoder, article_encoder=config.article_encoder, attention_over_article_tokens=config.attn, condition_attention=config.cond_attn, tokenwise_attention=config.tokenwise_attention, tune_embeddings=config.tune_embeddings, init_embeddings=init_word_embeddings)
    if cuda:
        initial_model = initial_model.cuda()
    attn_metrics = {
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
    train_Xy, val_Xy, test_Xy = mangle_method(real_train_Xy, real_val_Xy, real_test_Xy, inference_vectorizer)
    if config.attn and config.pretrain_attention:
        print("pre-training attention")
        if config.pretrain_attention == 'pretrain_attention_to_match_span':
            ev_inf, attn_metrics = pretrain_attention_to_match_span(train_Xy, val_Xy, initial_model, epochs=config.attn_epochs, batch_size=config.attn_batch_size, tokenwise_attention=config.tokenwise_attention, cuda=cuda, attention_acceptance=config.attention_acceptance)
        elif config.pretrain_attention == 'pretrain_attention_with_concatenated_spans':
            ev_inf, attn_metrics = pretrain_attention_with_concatenated_spans(train_Xy, val_Xy, initial_model, epochs=config.attn_epochs, batch_size=config.attn_batch_size, tokenwise_attention=config.tokenwise_attention, cuda=cuda, attention_acceptance=config.attention_acceptance)
        elif config.pretrain_attention == 'pretrain_attention_with_random_spans':
            ev_inf, attn_metrics = pretrain_attention_with_random_spans(train_Xy, val_Xy, initial_model, epochs=config.attn_epochs, batch_size=config.attn_batch_size, tokenwise_attention=config.tokenwise_attention, cuda=cuda, attention_acceptance=config.attention_acceptance)
        elif config.pretrain_attention == 'pretrain_tokenwise_attention':
            ev_inf, attn_metrics = pretrain_tokenwise_attention(train_Xy, val_Xy, initial_model, epochs=config.attn_epochs, batch_size=config.attn_batch_size, tokenwise_attention=config.tokenwise_attention, cuda=cuda, attention_acceptance=config.attention_acceptance)
        elif config.pretrain_attention == 'pretrain_tokenwise_attention_balanced':
            ev_inf, attn_metrics = pretrain_tokenwise_attention_balanced(train_Xy, val_Xy, initial_model, epochs=config.attn_epochs, batch_size=config.attn_batch_size, tokenwise_attention=config.tokenwise_attention, cuda=cuda, attention_acceptance=config.attention_acceptance)
        elif config.pretrain_attention == 'pretrain_max_evidence_attention':
            ev_inf, attn_metrics = pretrain_max_evidence_attention(train_Xy, val_Xy, initial_model, epochs=config.attn_epochs, batch_size=config.attn_batch_size, tokenwise_attention=config.tokenwise_attention, cuda=cuda, attention_acceptance=config.attention_acceptance)
        else:
            raise ValueError("Unknown pre-training configuration {}".format(config.pretrain_attention))
    else:
        ev_inf = initial_model

    best_model, _, _, _, val_metrics, final_test_preds = train(ev_inf, train_Xy, val_Xy, test_Xy, inference_vectorizer, batch_size=config.batch_size, epochs=config.epochs, shuffle=shuffle)
    if config.attn and config.article_sections == 'all' and config.data_config == 'vanilla':
        final_train_auc = evaluate_model_attention_distribution(ev_inf, train_Xy, cuda=cuda)
        final_val_auc = evaluate_model_attention_distribution(ev_inf, val_Xy, cuda=cuda)
        final_test_auc = evaluate_model_attention_distribution(ev_inf, test_Xy, cuda=cuda)
    else:
        final_train_auc = ""
        final_val_auc = ""
        final_test_auc = ""

    val_metrics['final_train_auc'] = final_train_auc
    val_metrics['final_val_auc'] = final_val_auc
    val_metrics['final_test_auc'] = final_test_auc

    return best_model, val_metrics, attn_metrics, final_test_preds


Config = namedtuple('Config', ['article_sections', 'ico_encoder', 'article_encoder', 'attn', 'cond_attn', 'tokenwise_attention', 'batch_size', 'attn_batch_size', 'epochs', 'attn_epochs', 'data_config', 'pretrain_attention', 'tune_embeddings', 'no_pretrained_word_embeddings', 'attention_acceptance'])


def generate_paper_results(configurations, mode='experiment', save_dir=None, determinize=False):
    results_list = []
    data_dict = {}
    for mangle_method, article_section_set, current_config in configurations:
        if current_config.article_sections not in data_dict:
            data_dict[current_config.article_sections] = get_data(article_section_set, mode=mode, include_sentence_span_splits = (current_config.data_config == "scan_net" or current_config.data_config == "scan_net_ICO"))
        real_train_Xy, real_val_Xy, real_test_Xy, inference_vectorizer = data_dict[current_config.article_sections]
        print("Current configuration: ", current_config)
        best_model, val_metrics, attn_metrics, final_test_preds = run(real_train_Xy, real_val_Xy, real_test_Xy, inference_vectorizer, mangle_method, current_config, determinize=determinize)
        results_list.append((val_metrics, attn_metrics))
        if save_dir is not None:
            config_name = str(current_config).replace(' ', '').replace('(', '').replace(')', '').replace('Config', '').replace('article_sections', 'as').replace('ico_encoder', 'icoe').replace('tokenwise_attention', 'twattn').replace('cond_attn', 'ca').replace('batch_size', 'bs').replace('data_config', 'dc').replace('pretrain_attention', 'pta').replace('tune_embeddings', 'te').replace('no_pretrained_word_embeddings', 'nptwe').replace('False', 'F').replace('True', 'T').replace("'", '').replace('"', '')
            torch.save(best_model, os.path.join(save_dir, config_name + '.pkl'), pickle_module=dill)
            with open(os.path.join(save_dir, config_name + '.decoded.csv'), 'w') as of:
                for _, p_id, pred in final_test_preds:
                    of.write(",".join([str(p_id), str(pred - 1)]) + '\n')
            results_to_csv(current_config, val_metrics, attn_metrics, os.path.join(save_dir, config_name + '.results.csv'))
    return results_list


def results_to_csv(config: namedtuple, val_metrics: dict, attn_metrics: dict, output_csv=None):
    keys = set(val_metrics.keys()) | set(attn_metrics.keys())
    keys.add('epoch')

    overlapping_keys = val_metrics.keys() & attn_metrics.keys()
    if len(overlapping_keys) > 0:
        raise ValueError("Found overlapping keys {} in training and attention metrics".format(overlapping_keys))

    def find_keys_with_epoch(metrics_dict: dict) -> set:
        return set(list(filter(lambda x: type(metrics_dict[x]) is list, metrics_dict.keys())))
    val_keys_with_epoch = find_keys_with_epoch(val_metrics)
    val_keys_without_epoch = val_metrics.keys() - val_keys_with_epoch
    attn_keys_with_epoch = find_keys_with_epoch(attn_metrics)
    attn_metrics_without_epoch = attn_metrics.keys() - attn_keys_with_epoch

    keys_without_epochs = (val_metrics.keys() - val_keys_with_epoch) | (attn_metrics.keys() - attn_keys_with_epoch)

    model_parameters = dict(config._asdict())
    df = pd.DataFrame(columns=(list(model_parameters.keys()) + list(keys)))

    def add_all_keys_with_epoch(df_, keys_with_epoch, undef_keys, metrics):
        (max_epoch,) = set(len(metrics[x]) for x in keys_with_epoch)
        for epoch in range(max_epoch):
            update_dict_ = dict(model_parameters)
            update_dict_['epoch'] = epoch
            for k in keys_with_epoch:
                update_dict_[k] = metrics[k][epoch]
            for k in undef_keys:
                update_dict_[k] = ""
            df_ = df_.append(update_dict_, ignore_index=True)
        return df_

    df = add_all_keys_with_epoch(df, val_keys_with_epoch, keys_without_epochs | attn_keys_with_epoch, val_metrics)
    df = add_all_keys_with_epoch(df, attn_keys_with_epoch, keys_without_epochs | val_keys_with_epoch, attn_metrics)
    update_dict = dict(model_parameters)
    for k in attn_metrics_without_epoch:
        update_dict[k] = attn_metrics[k]
    for k in val_keys_without_epoch:
        update_dict[k] = val_metrics[k]
    for k in val_keys_with_epoch | attn_keys_with_epoch:
        update_dict[k] = ""
    update_dict['epoch'] = ""
    df = df.append(update_dict, ignore_index=True)
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False, compression=None)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment and dump it's output.")
    parser.add_argument('--article_sections', dest='article_sections', default='all', help='which article sections to load')
    parser.add_argument('--ico_encoder', dest='ico_encoder', default='GRU', help='CBoW or GRU or biGRU?')
    parser.add_argument('--article_encoder', dest='article_encoder', default='GRU', help='CBoW or GRU?')
    parser.add_argument('--attn', dest='attn', action='store_true', default=False, help='Do we want an attention function?')
    parser.add_argument('--cond_attn', dest='cond_attn', action='store_true', default=False, help='Do we want to condition our attention?')
    parser.add_argument('--tokenwise_attention', dest='tokenwise_attention', action='store_true', default=False, help='Do we want our attention distribution to be tokenize?')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--attn_batch_size', dest='attn_batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--attn_epochs', dest='attn_epochs', type=int, default=10)
    parser.add_argument('--data_config', dest='data_config', default='vanilla')
    parser.add_argument('--pretrain_attention', dest='pretrain_attention', default=False)
    parser.add_argument('--attention_acceptance', dest='attention_acceptance', default='auc', help="What is the criterion for choosing the best model")
    parser.add_argument('--tune_embeddings', dest='tune_embeddings', action='store_true', default=False)
    parser.add_argument('--no_pretrained_word_embeddings', dest='no_pretrained_word_embeddings', action='store_true', default=False)
    parser.add_argument('--scan_net_location', dest='scan_net_location', type=str, default="")
    parser.add_argument('--save_dir', dest='save_dir', default=None, help='Where do we save our models to?')
    parser.add_argument('--mode', dest='mode', default='experiment', help='paper or experiment (run with *real* test data, or run with validation data?)')
    parser.add_argument('--determinize', dest='determinize', default=False, action='store_true', help='Do we run in a deterministic mode or not?')

    args = parser.parse_args()
    config = Config(article_sections=args.article_sections,
                    ico_encoder=args.ico_encoder,
                    article_encoder=args.article_encoder,
                    attn=args.attn,
                    cond_attn=args.cond_attn,
                    tokenwise_attention=args.tokenwise_attention,
                    batch_size=args.batch_size,
                    attn_batch_size=args.attn_batch_size,
                    epochs=args.epochs,
                    attn_epochs=args.attn_epochs,
                    data_config=args.data_config,
                    pretrain_attention=args.pretrain_attention,
                    tune_embeddings=args.tune_embeddings,
                    no_pretrained_word_embeddings=args.no_pretrained_word_embeddings,
                    attention_acceptance=args.attention_acceptance)
    article_sections = {'all': None, 'abstract/results': {'abstract', 'results'}, 'results': {'results'}, 'abstracts': {'abstract'}}
    data_configs = {'vanilla': identity, 'cheating': replace_articles_with_evidence_spans, 
                    'no_prompt': replace_prompts_with_empty, 
                    'no_article': replace_articles_with_empty,
                    'double_training_trick': double_training_trick, 
                    'scan_net': scan_net_preprocess_create(args.scan_net_location), 
                    'scan_net_ICO': scan_net_ICO_preprocess_create(args.scan_net_location)}
    configs = [(data_configs[args.data_config], article_sections[args.article_sections], config)]
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    results = generate_paper_results(configs, mode=args.mode, save_dir=args.save_dir, determinize=args.determinize)
    if len(results) > 1:
        raise ValueError("Can't properly output more than result file in this setting, FIXME")
    val_metrics, attn_metrics = results[0]
    df = results_to_csv(config, val_metrics, attn_metrics)
    print("<csvsnippet>")
    df.to_csv(sys.stdout, index=False, compression=None)
    print("</csvsnippet>")


if __name__ == '__main__':
    main()
