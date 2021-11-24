#!/usr/bin/env python'3'
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:1'4':09 2021

@author: mac
"""
import networkx as nx
import pickle
import pandas as pd
from Louvain_model import DirectedGraph 
from collections import Counter, defaultdict

def read_csv(file_name):
    G = nx.DiGraph()
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0 : continue
            edge = [tuple(line.replace('\n', '').split(',') )]
            G.add_edges_from(edge)
    return G

def read_label(file_name):
    label_list = defaultdict(lambda: 'UNK')
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0 : continue
            id, label = line.replace('\n', '').split(',')
            label_list[id] = label
    return label_list

def toy():
     G = nx.DiGraph()
     G.add_edges_from( 
         [('1', '2'), ('2','3'), ('3','4'), ('4','1'), ('2','1'), ('3','2'), ('4','3'), ('1','4'),
        ('4','9'), ('9','5') , ('5','6'), ('9','4'), ('5','9'), ('6','5'),
        ('6','7'), ('7','8') , ('8','6')]
     )
     return G

def voting(res, gt):
    voting_list = {i:Counter() for i in res.keys()}
    for name, community in res.items():
        for node in community:
            voting_list[name].update([gt[node]])
        voting_list[name].pop('UNK')
    
    final_ = { class_id:[] for class_id in gt.values()}
    for key in voting_list:
        sort = sorted(voting_list[key], key=voting_list[key].get, reverse=True)
        final_[sort[0]] += res[key]
        print(key, '\t', sort)

    return final_

def eval(res, gt):
    correct = 0.
    for name, community in res.items(): 
        for node in community:
            correct += 1 if gt[node] == name else 0

    return correct / 300 

def to_csv(res, output_file):
    with open (output_file, 'w') as f:
        f.write('id,category\n')
        for name, community in res.items():
            for node in community:
                f.write(node + ',' + name + '\n')

if __name__ == '__main__':
    G = read_csv('dataset/edges_update.csv')
    #G = toy()
    DG = DirectedGraph(G)
    res = DG.louvain(9, 9)

    gt = read_label('dataset/ground_truth.csv')
    print(len(gt))
    voting_res = voting(res, gt)
    print('1st acc= ', eval(voting_res, gt))

    # Next is some trick, cuz we find that class 0 and class 1 are not well seperated
    # So we do clustering again.
    for commu in [ '2', '3', '4']:
        G.remove_nodes_from(voting_res[commu])
    new_DG = DirectedGraph(G)
    new_res = new_DG.louvain(2, 10)
    voting2 = voting(new_res, gt)

    final_res = {'0' : voting2['0'], '1': voting2['1'], 
                 '2' : voting_res['2'], '3': voting_res['3'], '4': voting_res['4'] }

    with open("trial3.txt", "wb") as fp:   #Pickling
        pickle.dump(final_res, fp)

    to_csv(final_res, 'trial3.csv')
    print('2nd acc= ', eval(final_res,gt))
