# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:39:34 2018

@author: Eric
"""
import glob
import numpy as np
import pandas as pd


def load_data(l):
    """
    Load some file.
    
    @param l is the location to load.
    """
    df = pd.read_csv(l, engine = "python", encoding = "utf-8")
    df.fillna("")
    return np.asarray(df)

loc_ann = "..//.//annotations//annotations_merged.csv"
loc_prompt = "..//.//annotations//prompts_merged.csv"

ann = load_data(loc_ann)
pmt = load_data(loc_prompt)

def num_prompts():
    """
    Number of total prompts.
    """
    print("Number of unique prompts: {}".format(len(set([r[2] + r[3] + r[4] for r in pmt]))))

def num_annotations_unique():
    """
    Number of annotations (unique).
    """
    print("Number of prompts annotated (including duplicates): {}".format(len(set(r[1] for r in ann))))
    
def num_annotations_not_unique():
    """
    Just total number of annotations.
    """
    print("Number of total annotations: {}".format(len(ann)))

def num_prompts_per_article():
    """
    Average number of prompts per article
    """
    files   = len(set([r[1] for r in pmt]))
    num_pmt = len(set([r[2] + r[3] + r[4] for r in pmt]))
    print("Ratio of prompts per article: {} / {} = {:.3f}".format(num_pmt, files, num_pmt / files))

def num_invalid_articles():
    """
    Number of invalid articles (out of total number of articles) + percentage
    """
    files = len(glob.glob("..//.//annotations//xml_files//*.nxml"))
    inv   = len(load_data(".//extra_files//invalid.csv"))
    print("Percent of invalid files: {} / {} = {:.3f}".format(inv, files, inv / files))

def num_invalid():
    """
    Average number of invalid reasonings/answers/prompts.
    """
    num_ann     = len(ann)
    num_inv_ans = sum([1 if r[3] == False else 0 for r in ann])
    num_inv_res = sum([1 if r[4] == False else 0 for r in ann])
    print("Ratio of invalid answers:    {} / {} = {:.3f}".format(num_inv_ans, num_ann, num_inv_ans / num_ann))
    print("Ratio of invalid reasonings: {} / {} = {:.3f}".format(num_inv_res, num_ann, num_inv_res / num_ann))

def mark_invalid():
    """
    How often a prompt is marked invalid.
    """
    num_ann = sum([1 if r[0] != 0 else 0 for r in ann])
    num_inv = sum([1 if r[-4] == 3 else 0 for r in ann])
    print("Ratio of answers marked invalid: {} / {} = {:.3f}".format(num_inv, num_ann, num_inv / num_ann))

def num_per_generated(printing):
    """
    The number that each prompt generator generated
    """
    
    def load_and_count(l):
        """
        Loads the location, and counts number of unique rows.
        """
        df = pd.read_csv(l, engine = 'python', encoding = 'utf-8')
        df.fillna("", inplace = True)
        df = np.asarray(df)
        num_gen = set()
        for row in df:
            num_gen.add(row[1] + row[2] + row[3])
        
        return len(num_gen)
        
    files = glob.glob(".//additional_files//*.csv")
    data  = {}
    for f in files:
        num_gen = load_and_count(f)
        name = f.split("out_")[1].split(".csv")[0]
        data[name] = num_gen
        if (printing):
            print("{} generated {} prompts".format(name, num_gen))
        
    return data

def pmt_gen_details():
    # Name: [Total hours, total cost, pay rate]
    return {"danijel":   [4.66,   116.67,  25.00],
            "shahzad":   [70.00,  1295.01, 18.50],
            "fernando":  [136.50, 3412.51, 25.00],
            "luisana":   [5.00,   75.00,   15.00],
            "giuseppe":  [3.66,   91.67,   25.00],
            "alejandra": [9.16,   128.33,  14.00],
            "cesar":     [7.00,   147.00,  21.00],
            "andrea":    [15.66,  391.66,  25.00],
            "krystie":   [241.66, 2416.66, 10.00],
            "sergii":    [184.83, 2218.00, 12.00]}

def avg_time_pmt(threshold):
    """
    Average time for prompt generation.
    """
    # get dictionary
    data    = pmt_gen_details()
    num_gen = num_per_generated(False)
    
    # set initial values w/ time_spent in hours
    total_num_gen, total_time_spent = 0, 0 
    for k in data.keys():
        if not(threshold and data[k][1] < 500):
            indv_num_gen  = num_gen[k]
            time_spent    = data[k][0]
            total_num_gen += indv_num_gen
            total_time_spent += time_spent
        
    if (threshold):
        print("Average number of prompts per hour w/ threshold: {} / {} = {:.3f}".format(total_num_gen, total_time_spent, total_num_gen / total_time_spent))
    else:
        print("Average number of prompts per hour w/out threshold: {} / {} = {:.3f}".format(total_num_gen, total_time_spent, total_num_gen / total_time_spent))

    return None

def avg_cost_pmt(threshold):
    """
    Average time for prompt generation.
    """
    # get dictionary
    data    = pmt_gen_details()
    num_gen = num_per_generated(False)
    
    # set initial values w/ time_spent in hours
    total_num_gen, total_money_spent = 0, 0 
    for k in data.keys():
        if not(threshold and data[k][1] < 500):
            indv_num_gen  = num_gen[k]
            money_spent   = data[k][1]
            total_num_gen += indv_num_gen
            total_money_spent += money_spent
        
    if (threshold):
        print("W/ threshold, each prompt costs: {} / {} = {:.3f}".format(total_money_spent, total_num_gen, total_money_spent / total_num_gen))
    else:
        print("W/O threshold, each prompt costs: {} / {} = {:.3f}".format(total_money_spent, total_num_gen, total_money_spent / total_num_gen))

    return None

def avg_time_ann():
    """
    Average time per prompt for annotation.
    """
    money = {'edin': 1837.50, 'lidija': 851.69, 'milorad': 3849.75}
    rate  = {'edin': 15, 'lidija': 10, 'milorad': 14.50}
    total_time_spent = 0
    total_number_prompts = 0
    for key in money.keys():
        total_time_spent += money[key] / rate[key]
        total_number_prompts += len(np.loadtxt("./extra_files/ordering_list_" + key + '.txt', delimiter = " "))
        
    print("Average number of prompts annotated per hour: {} / {:.2f} = {:.3f}".format(total_number_prompts, total_time_spent, total_number_prompts / total_time_spent))
    return None

def avg_cost_ann():
    """
    Average cost per prompt for annotation.
    """
    money = {'edin': 1837.50, 'lidija': 851.69, 'milorad': 3849.75}
    total_cost = 0
    total_number_prompts = 0
    for key in money.keys():
        total_cost += money[key]
        total_number_prompts += len(np.loadtxt("./extra_files/ordering_list_" + key + '.txt', delimiter = " "))
        
    print("Average cost of annotation per prompt: {:.2f} / {:.2f} = {:.3f}".format(total_cost, total_number_prompts, total_cost / total_number_prompts))
    return None

def avg_time_ver():
    """ Find average time per prompt for an annotation. """
    money = {'ahmed': 330.67, 'hazel': 4172.50, 'daniela': 1302.00}
    rate = {'ahmed': 16, 'hazel': 15, 'daniela': 18}
    total_time_spent = 0
    total_number_prompts = 0
    for key in money.keys():
        total_time_spent += money[key] / rate[key]
        total_number_prompts += len(np.asarray(pd.read_csv("./extra_files/out_" + key + '.csv', engine = 'python')))
        
    print("Average number of prompts verified per hour: {} / {:.2f} = {:.3f}".format(total_number_prompts, total_time_spent, total_number_prompts / total_time_spent))
    return None

def avg_cost_ver(): 
    """ Find average cost per prompt for an verification. """
    money = {'ahmed': 330.67, 'hazel': 4172.50, 'daniela': 1302.00}

    total_cost = 0
    total_number_prompts = 0
    for key in money.keys():
        total_cost += money[key]
        total_number_prompts += len(np.asarray(pd.read_csv("./extra_files/out_" + key + '.csv', engine = 'python')))
        
    print("Average cost of verified per prompt: {:.2f} / {:.2f} = {:.3f}".format(total_cost, total_number_prompts, total_cost / total_number_prompts))
    return None

def total_cost():
    """
    Cost per prompt for all (+ total)
    """
    total_prompts = 10671
    money1 = {'ahmed': 330.67, 'hazel': 4172.50, 'daniela': 1302.00}
    money2 = {'edin': 1837.50, 'lidija': 851.69, 'milorad': 3849.75}
    money3 = {"danijel": 116.67, "shahzad": 1295.01, "fernando": 3412.51, "luisana": 75.00,
              "giuseppe": 91.67, "alejandra": 128.33, "cesar": 147.00,
              "andrea": 391.66, "krystie": 2416.66, "sergii": 2218.00}
    
    money = [money1, money2, money3]
    total_cost = 0
    for m in money:
        for p in m.keys():
            total_cost += m[p]
            
    print("Average total cost per prompt: {:.2f} / {:.2f} = {:.3f}".format(total_cost, total_prompts, total_cost / total_prompts))
    return None



if __name__ == '__main__':
    num_prompts()
    num_annotations_unique()
    num_annotations_not_unique()
    num_prompts_per_article()
    num_invalid_articles()
    num_invalid()
    mark_invalid()
    avg_time_pmt(True)
    avg_time_pmt(False)
    avg_cost_pmt(True)
    avg_cost_pmt(False)
    avg_time_ann()
    avg_cost_ann()
    avg_time_ver()
    avg_cost_ver()
    total_cost()