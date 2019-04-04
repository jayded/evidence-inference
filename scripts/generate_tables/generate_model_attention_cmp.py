# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 00:04:24 2018

@author: Eric
"""
from tabulate import tabulate
from table_parse_helper import find_best_f1, formalize_table_neural_experiments, gen_all_attention_type

 
def table_3_comparisons():
    """ The function for getting the best model results."""
    # get configuration for best attention/no-attention + vanilla
    attn_v    = find_best_f1({"attn": 'True', 'data_config': 'vanilla', 'pretrain_attention': 'False'})
    # find all other pretraining types 
    other_pretrain = gen_all_attention_type()
        
    res = [formalize_table_neural_experiments("w/ attn, w/o pretrain", attn_v)]
    for prtn in other_pretrain:
        tmp = find_best_f1({"attn": 'True', 
                            'data_config': 'vanilla', 
                            'pretrain_attention': prtn})
        res.append(formalize_table_neural_experiments("w/ {} w/ attn".format(prtn), tmp))
    
    """        
    # calculate delta from the last col
    cmp_f1 = float(res[0][-1])     
    for r in res:
        r.append("{:.3f}".format(float(r[-1]) - cmp_f1))
    """
    return res
    

def gen_table_data():
    """ Generate a table. """
    c_header = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    table    = [c_header]
    all_exp = table_3_comparisons()
    
    for row in all_exp:
        table.append(row)
    
    return table

def gen_latex_table(table):
    """ Generate latex table based on table. """ 
    content = "\\begin{table*}\n"
    content += tabulate(table[1:], headers = table[0], tablefmt="latex")
    content += "\n\\end{table*}"
    return content


if __name__ == "__main__":
    table = gen_table_data()
    latex = gen_latex_table(table)
    print(latex)