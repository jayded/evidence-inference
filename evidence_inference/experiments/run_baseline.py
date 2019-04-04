import os
import sys
import argparse
from model_0_paper_experiment import generate_paper_results, Config, identity, replace_articles_with_evidence_spans, replace_prompts_with_empty, replace_articles_with_empty, double_training_trick, scan_net_preprocess_create, results_to_csv, scan_net_ICO_preprocess_create
from LR_pipeline import run_lr_pipeline
from scan_net_regression import run_scan_net_regression
from scan_net_redux import run_scan_net_redux
from scan_net_ICO_redux import run_scan_net_ico

from os.path import abspath, dirname, join
# this monstrosity produces the module directory in an environment where this is unpacked

sys.path.insert(0, abspath(join(dirname(abspath(__file__)), '..', '..')))

from evidence_inference.models.heuristics import main as heuristics
from evidence_inference.models.heuristics_cheating import main as heuristics_cheating
from evidence_inference.models.regression import main as lr
from evidence_inference.experiments.regression_cheating import main as lr_cheating

def run_scan_net(sn_loc, save_model, data_config = 'scan_net_ICO'):
    config = Config(article_sections='all',
                    ico_encoder='CBoW',
                    article_encoder='GRU',
                    attn=False,
                    cond_attn=False,
                    tokenwise_attention=False,
                    batch_size=32,
                    attn_batch_size=32,
                    epochs=25,
                    attn_epochs=25,
                    data_config=data_config,
                    pretrain_attention=False,
                    tune_embeddings=False,
                    no_pretrained_word_embeddings=False,
                    attention_acceptance='auc')
    article_sections = {'all': None, 'abstract/results': {'abstract', 'results'}, 'results': {'results'}, 'abstracts': {'abstract'}}
    data_configs = {'vanilla': identity, 'cheating': replace_articles_with_evidence_spans, 
                    'no_prompt': replace_prompts_with_empty, 
                    'no_article': replace_articles_with_empty,
                    'double_training_trick': double_training_trick, 
                    'scan_net': scan_net_preprocess_create(sn_loc), 
                    'scan_net_ICO': scan_net_ICO_preprocess_create(sn_loc)}
    configs = [(data_configs[data_config], article_sections['all'], config)]
    os.makedirs(save_model, exist_ok=True)
    results = generate_paper_results(configs, mode='experiment', save_dir=save_model, determinize=False)
    if len(results) > 1:
        raise ValueError("Can't properly output more than result file in this setting, FIXME")
    val_metrics, attn_metrics = results[0]
    df = results_to_csv(config, val_metrics, attn_metrics)
    print("<csvsnippet>")
    df.to_csv(sys.out, index=False, compression=None)
    print("</csvsnippet>")
    
    
def main(config, init_save, final_save):
    if config == 'LR_Pipeline':
        run_scan_net_regression(init_save)
        run_lr_pipeline(100, True, path = init_save)
    elif config == 'scan_net ICO':
        run_scan_net_redux(init_save)
        run_scan_net(init_save, final_save, data_config = 'scan_net_ICO')
    elif config == 'scan_net':
        run_scan_net_ico(init_save)
        run_scan_net(init_save, final_save, data_config = 'scan_net_ICO')
    elif config == 'heuristic':
        heuristics()
    elif config == 'heuristic_cheating':
        heuristics_cheating()
    elif config == 'lr':
        lr(100, True)
    elif config == 'lr_cheating':
        lr_cheating(100, True)
    else:
        pass

parser = argparse.ArgumentParser(description="Run a single baseline experiment.")
parser.add_argument('--config',       dest='config', help='Which experiment to run.')
parser.add_argument('--init_save',  dest='half_save',     default='tmp.pth', help='Where to save the part of the pipeline to.')
parser.add_argument('--final_save', dest='final_save',    default='tmp_f.pth', help='Where to save the second part of the pipeline to.')
args = parser.parse_args()

main(args.config, args.init_save, args.final_save)
