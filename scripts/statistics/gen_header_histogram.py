# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:01:10 2018

@author: Eric

Generate the histograms for the file headers (Abstract/Methods-Materials/... etc.).
"""
import glob
import string
import pandas as pd
import xml.etree.ElementTree as ET

# Location of the files.
xml_loc = '..\\.\\annotations\\xml_files\\'
table = str.maketrans({key: None for key in string.punctuation})

"""
Remove punctuation and numbers.
"""
def transform(t):
    return ''.join([i for i in t if not i.isdigit()]).translate(table).strip(" ").lower()

"""
Return the article split into sections. It will return an array of pairs, 
with a given pair having a first entry of the title, and the second entry
containing the actual text of that section.

@param body represents the whole article.
"""
def _get_sections(body):
    if (body is None):
        return []
    arr = []
    title = ""
    paragraph = ""
    children = body.getchildren()
    for i in range(len(children)):
        child = children[i]
        if (child.tag == 'sec'):
            sub_sec = _get_sections(child)
            if (sub_sec is None):
                continue
            else:
                arr.extend(sub_sec)
        elif (child.tag == 'title'):
            title = ET.tostring(child, method = 'text', encoding = 'utf8').decode('utf-8')
        else:
            paragraph += ET.tostring(child).decode('utf-8')
            
    if (title == '' and len(arr) > 0):
        return arr
    elif (len(arr) > 0):
        return [title].extend(arr)
    else:
        return [title]

all_loc = glob.glob(xml_loc + '*.nxml')
res = {}
arr = {}
for x in all_loc:
    e = ET.parse(x).getroot()
    body = e.find("body")
    titles_bd = _get_sections(body)
    
    if not(e.find('front').find('article-meta').find('abstract') is None): 
        titles_bd.append('abstract')
    
    titles = titles_bd
    for t in titles:
        t = transform(t)
        if (t == '' or t is None):
            continue
        elif (t in res):
            res[t] += 1
            if (res[t] >= 15):
                arr[t] = res[t]    
        else:
            res[t] = 1


df = pd.DataFrame.from_dict(arr, orient='index')
df.plot(kind = 'bar')