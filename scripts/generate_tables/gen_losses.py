# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:29:24 2018

@author: Eric
"""
import matplotlib.pyplot as plt

from table_parse_helper import parse_all_experiments

def gen_losses(variants, loss_type):
    rows = parse_all_experiments(variants)
    losses = []
    for k in rows.keys():
        row = rows[k]
        if (loss_type in row):
            vl  = row[loss_type]
            losses.append(vl)
        
    return losses
  
ex_v = {}
tr_row = gen_losses(ex_v, "train_loss")
va_row = gen_losses(ex_v, "val_loss")

plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([float(e) for e in tr_row[0]])
plt.plot([float(e) for e in va_row[0]])

plt.show()


