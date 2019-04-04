# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:42:12 2019

@author: Eric Lehman
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

import matplotlib.pyplot as plt

SAVE = True

def new_y_name(n):
    n = ' '.join(n.split('_'))
    n = n.replace('val', 'validation')
    n = n.replace('all', '')
    n = n.replace('auc', 'token AUC')
    n =  n.capitalize()
    print(n)
    
    if n == 'Pretrain attn validation token auc ':
        return 'Evidence token AUC'
    elif n == 'Pretrain attn validation entropies':
        return 'Evidence token entropy'
    elif n == 'Validation evidence token mass':
        return 'Evidence token mass'
    elif n == 'Pretrain attn validation token masses':
        return 'Evidence token mass'
    else:
        return n

def gen_name_from_header(header_array, header_order):
    """ Take from one header, and generate a name. """
    names = ['article_sections', 'ico_encoder', 'article_encoder', 'attn', 'cond_attn', 'tokenwise_attention', 'data_config', 'pretrain_attention']
    final_name = ""
    for n in names:
        final_name += n + "=" + str(header_array[header_order[n]])
        
        if n != names[-1]:
            final_name += ","
        
    return final_name

def plot_seaborn(x, y, col_diff, diff_name, x_name, y_name):
    fig, ax = plt.subplots()
    if col_diff is None:
        d = {x_name: x, y_name: y}
        data = pd.DataFrame(data = d)
        
        sns.lineplot(x=x_name, y=y_name, data=data)
        plt.show() # SAVE TO PDF

    else:
        d = {x_name: x, y_name: y, diff_name: col_diff}
        data = pd.DataFrame(data = d)
        sns.lineplot(x=x_name, y=y_name, hue=diff_name, data=data, palette='Blues_r')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:]) #, loc = 'upper right')
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.ylabel(new_y_name(y_name))
        plt.xlabel(x_name.capitalize())
        if SAVE:
            plt.savefig("neg_auc" + x_name + "_" + y_name + ".pdf", bbox_inches='tight')
        
def plot_seaborn_neg(fx, fy, diff, splits, to_set_dashed, colors, x_name, y_name):
    d = {x_name: fx, y_name: fy, 'subgroup': splits}
    data = pd.DataFrame(data = d)
    
    #a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots() #figsize = a4_dims)
    i = 0
    for k, g in data.groupby('subgroup'):
        g.plot(ax=ax, x=x_name, y=y_name, legend=False, color=colors[i])
        i += 1
     
    lines = []  
    for i in range(len(ax.lines)):
        if to_set_dashed[i]:
            ax.lines[i].set_linestyle("--")    
        
        lines.append(ax.lines[i])
          
    ax.legend(lines, diff)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.ylabel(new_y_name(y_name))
    plt.xlabel(x_name.capitalize())
    if SAVE:
        plt.savefig("neg_auc" + x_name + "_" + y_name + ".pdf", bbox_inches='tight')
        
def plot_seaborn_bar(xs, ys, errors, x_axis, y_axis, diff):
    fig, ax = plt.subplots()
    d = {x_axis: xs, y_axis: ys, 'Model Variants': diff}
    df = pd.DataFrame(data = d)
    sns.barplot(data=df, yerr=errors, x='Model Variants', y=y_axis, palette = 'colorblind')#, hue='Model Variants', yerr= errors, data=df)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.ylabel(new_y_name(y_axis))

    if SAVE:
        plt.savefig("bar_graph_token_mass.pdf", bbox_inches='tight')
    
def plot(x, y, x_name, y_name, header, header_order, one_graph = False, label_info = []):
    ### USE SEABORN ### 
    fig, ax = plt.subplots()
    
    # PUT EVERYTHING IN ONE GRAPH
    if one_graph and len(label_info) == 0:
        
        for t1, t2, h in zip(x, y, header):
            ax.plot(t1, t2, label = gen_name_from_header(h, header_order))
            
        plt.legend(bbox_to_anchor=(1.1, 1.05))
    elif one_graph and len(label_info) > 0:
        for t1, t2, h in zip(x, y, label_info):
            ax.plot(t1, t2, label = h)
            
        plt.legend(bbox_to_anchor=(1.1, 1.05))
    else:
        ax.plot(x, y)
        
    
    ax.set(xlabel=x_name, ylabel=y_name)
    ax.grid()
    
    #fig.savefig(gen_name_from_header(header_array))
    plt.show()
    return None