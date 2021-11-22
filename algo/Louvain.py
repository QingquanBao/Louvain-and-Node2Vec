#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:14:09 2021

@author: mac
"""
import networkx as nx
import pandas as pd
from Louvain_model import DirectedGraph 

def read_csv(file_name):
    G = nx.DiGraph()
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0 : continue
            edge = [tuple(line.replace('\n', '').split(',') )]
            G.add_edges_from(edge)
    return G

if __name__ == '__main__':
    G = read_csv('edges_update.csv')
    DG = DirectedGraph(G)
    DG.louvain()
