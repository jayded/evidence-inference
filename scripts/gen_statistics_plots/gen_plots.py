# -*- coding: utf-8 -*-
"""
Load the data, print the plots.
"""
import copy
import numpy as np
from find_best_config import find_best
from plots import plot_seaborn, plot_seaborn_bar, plot, plot_seaborn_neg
from load_data import load, filter_data, extract_plotable_data
from sklearn.utils import shuffle


np.random.seed(500)

def negative_epoch_graph(restraints, diff, y_name, y_axis1, y_axis2, config = 'avg'):   
    df, header = load()
    xs, ys, splits, dashed, colors = [], [], [], [], []
    color_order = [(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                   (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                   (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                   (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),]
    i = 0; c_order = 0
    for r in restraints:
        filt       = filter_data(df, header, r)
        x1, y1, _ = extract_plotable_data(filt, header, 'epoch', y_axis1)
        x2, y2, _ = extract_plotable_data(filt, header, 'epoch', y_axis2)

        if config == 'avg' or config == 'sample':
            if len(y1) != 0:
                
                y1, x1, max_len1 = multi_average(x1, y1, config)
                y2, x2, max_len2 = multi_average(x2, y2, config)
                
                # change value of x1 and x2
                x1 = list(np.asarray(x1) - len(x1))
                x2 = list(np.asarray(x2) - 1)
                
                # connect the lines
                x1.append(0.0)
                y1.append(y2[0])
                
                # set up data
                x = x1 + x2
                y = y1 + y2
                splits += [i] * (max_len1 + 1) + [i] * max_len2
                dashed.append(True)
                colors += [color_order[c_order]]
                i += 1; c_order += 1
              
            else:
                y, x, max_len = multi_average(x2, y2, config)
                x = list(np.asarray(x) - 1)
                splits += [i] * max_len
                dashed.append(False)
                colors += [color_order[c_order]]
                i += 1; c_order += 1
            
        # add data
        xs.append(x)
        ys.append(y)

    fx = [] # final x, final y
    fy = []
    for x, y in zip(xs, ys):
        fx.extend(x)
        fy.extend(y)
    
    plot_seaborn_neg(fx, fy, diff, splits, dashed, colors, 'epoch', y_axis1)
    return None


def multi_average(X, to_avg, config):
    """ Average over axis 0 w/ variable amount of dimensions. Give also X to pick the X that matches the max length. """
    if config == 'sample':
        X, to_avg = shuffle(X, to_avg)
        X = X[:5]
        to_avg = to_avg[:5]
    
    max_len = max([len(y) for y in to_avg])
    use_x = None
    for y, x in zip(to_avg, X):
        if len(y) == max_len:
            use_x = x
            continue 
        
        while(len(y) < max_len):
            y.append(y[-1])
            
    return list(np.mean(to_avg, axis = 0)), use_x, max_len

def graph_error_bar(restraints, x_axis, y_axis, error_bar, diff):
    df, header = load()
    xs, ys, errors = [], [], []
    for r in restraints:
        filt      = filter_data(df, header, r)
        x1, y1, _ = extract_plotable_data(filt, header, x_axis, y_axis)
        x2, y2, _ = extract_plotable_data(filt, header, x_axis, error_bar)
 
        y1, x1, max_len1 = multi_average(x1, y1, 'sample')
        y2, x2, max_len2 = multi_average(x2, y2, 'sample')
        
        # add desired data.
        xs.append(x1[-1])
        ys.append(y1[-1])
        errors.append(y2[-1])
    
    # xs, ys, errors
    plot_seaborn_bar(xs, ys, errors, x_axis, y_axis, diff)

def overlay_graph_diff(restraints, y_name, x_axis, y_axis, config = 'single'):
    """ Different restraints, same variables. Y_name should be an array (len 2) of names. """
    df, header = load()
    filts, xs, ys, hs = [], [], [], []
    for r in restraints:
        filt    = filter_data(df, header, r)
        x, y, h = extract_plotable_data(filt, header, x_axis, y_axis)
        filts.append(filt)
        xs.append(x)
        ys.append(y)
        hs.append(h)
 
    if config == 'single':
        xs = [x[0] for x in xs]
        ys = [y[0] for y in ys]
        hs = [h[0] for h in hs]
        
    elif config == 'avg':
        new_xs, new_ys = [], []
        for x, y in zip(xs, ys):
            ty, tx, max_len = multi_average(x, y)
            new_xs.append(tx)
            new_ys.append(ty)
        
        xs, ys = new_xs, new_ys
        
    diff = [item for sublist in [[y_name[i]] * len(xs[i]) for i in range(len(xs))] for item in sublist]
    diff_name = 'Experiment Variants'
    fx = [] # final x, final y
    fy = []
    for x, y in zip(xs, ys):
        fx.extend(x)
        fy.extend(y)
    
    plot_seaborn(fx, fy, diff, diff_name, x_axis, y_axis)

def overlay_graph(restraints, y_name, y_axis1, y_axis2, config = 'single'):
    """ Same restraints, different variables (i.e. token_auc + entropy). """
    df, header = load()
    filt       = filter_data(df, header, restraints)
    x1, y1, h1 = extract_plotable_data(filt, header, 'epoch', y_axis1)
    x2, y2, h2 = extract_plotable_data(filt, header, 'epoch', y_axis2)
    
    if config == 'single':
        y1 = y1[0]
        y2 = y2[0]
        x1 = x1[0]
        x2 = x2[0]
        
    elif config == 'avg':
        y1 = list(np.mean(y1, axis = 0))
        y2 = list(np.mean(y2, axis = 0))     

    diff = [y_axis1] * len(x1) + [y_axis2] * len(x2) 
    x1 = x1 + x2
    y1 = y1 + y2 
    plot_seaborn(x1, y1, diff, y_axis1 + " vs. " + y_axis2, 'epochs', 'combined')
    return None

def main(restraints, x_axis, y_axis, one_graph = False):
    df, header = load()
    filt       = filter_data(df, header, restraints)
    xs, ys, hs = extract_plotable_data(filt, header, x_axis, y_axis)
    
    if not(one_graph):
        for x, y, h in zip(xs, ys, hs):
            plot(x, y, x_axis, y_axis, h, header, one_graph)
    else:
        plot(xs, ys, x_axis, y_axis, hs, header, one_graph)
        
    return None
        
restraints = {'article_sections':   set(['all']),
              'ico_encoder':        set(['CBoW']),
              'cond_attn':          set([True]),
              'pretrain_attention': set(['pretrain_tokenwise_attention']),
              'data_config':        set(['vanilla'])}

"""
Plot 
"""
def configure(plot_type, restraints, restraints_no_pretrain):
    restraints1 = copy.deepcopy(restraints); restraints1['cond_attn'] = set([True])
    restraints2 = copy.deepcopy(restraints_no_pretrain); restraints2['cond_attn'] = set([True])
    restraints3 = copy.deepcopy(restraints); restraints3['cond_attn'] = set([False])  
    restraints4 = copy.deepcopy(restraints_no_pretrain); restraints4['cond_attn'] = set([False])  
    
    restraints = [restraints1, restraints2, restraints3, restraints4]
    
    diff = ["cond-attn\n+ pretraining", 'cond-attn', 'attn\n+ pretraining', 'attn']
    if plot_type == 'negative epochs':
        negative_epoch_graph(restraints, diff, 'Token AUC (Validation)', 'pretrain_attn_val_auc_all', 'val_aucs', 'sample')
        negative_epoch_graph(restraints, diff, 'Token AUC (Validation)', 'pretrain_attn_val_entropies', 'val_entropies', 'sample')
        negative_epoch_graph(restraints, diff, 'Token AUC (Validation)', 'pretrain_attn_val_token_masses', 'val_evidence_token_mass', 'sample')
    elif plot_type == 'token auc w/ entropy':
        overlay_graph(restraints, 'Token AUC and Entropy', 'val_aucs', 'val_entropies')
    elif plot_type == 'val_auc & val_entropy':
        overlay_graph_diff(restraints, diff, 'epoch', 'val_aucs', config = 'avg')
        overlay_graph_diff(restraints, diff, 'epoch', 'val_entropies', config = 'avg')
    elif plot_type == 'val_f1':
        main(restraints, 'epoch', 'val_f1', one_graph = True)
    elif plot_type == 'token_mass hist':
        graph_error_bar(restraints, 'epoch', 'val_evidence_token_mass', 'val_evidence_token_err', diff)
    elif plot_type == 'token_mass plot':
        overlay_graph_diff(restraints, diff, 'epoch', 'val_evidence_token_mass', config = 'avg')
    else:
        pass

restraints = {'article_sections':   set(['all']),
              'data_config':        set(['vanilla']), 
              'attention_acceptance': set(['auc']),
              'tokenwise_attention': set(['True', True]),
              'pretrain_attention': set(['pretrain_tokenwise_attention', 'pretrain_max_evidence_attention'])}

restraints_no_pretrain = {'article_sections':   set(['all']),
                          'data_config':        set(['vanilla']),
                          'pretrain_attention': set(['False', False])}

# pretraining is false vs. not false
restraints             = find_best(restraints)

restraints_no_pretrain = find_best(restraints_no_pretrain)

print(restraints)

#configure('val_auc & val_entropy', restraints, restraints_no_pretrain)
configure('negative epochs', restraints, restraints_no_pretrain)
configure('token_mass hist', restraints, restraints_no_pretrain)
#configure('token_mass plot', restraints, restraints_no_pretrain)



    
    
