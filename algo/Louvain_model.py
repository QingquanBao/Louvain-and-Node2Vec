#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:14:09 2021

@author: mac
"""
from collections import Counter
from tqdm import tqdm


class CommunitySuperNode:
    def __init__(self, name, edges:list, in_edges):
        self.name = name
        self.node_list = []
        self.community = []

        self.in_neighbors = Counter()
        self.out_neighbors = Counter()
        #self.in_neighbors = []
        #self.in_weight = []
        #self.out_neighbors = []
        #self.out_weight = []

        self.out_total_degree = 0
        self.in_total_degree = 0
        self.to_commu = self.name
        #self.internals = 0
        self.__init_graph__(name, edges, in_edges)
    
    def __init_graph__(self, name, edges, in_edges):
        '''
        When the raw graph is created, we need to call this func
        Input:  
            edges: (list) list of the outward neighbors
        '''
        self.node_list += [self.name]
        self.community.append(name)
        self.in_neighbors.update({e: 1 for e in in_edges})
        self.out_neighbors.update({e: 1 for e in edges})
        #self.in_neighbors += [e[0] for e in in_edges]
        #self.in_weight = [1 for i in range(len(self.in_neighbors))]
        #self.out_neighbors += edges
        #self.out_weight = [1 for i in range(len(self.out_neighbors))]
        
        self.in_total_degree = len(self.in_neighbors)
        self.out_total_degree = len(self.out_neighbors)
        self.in_d = self.in_total_degree
        self.out_d = self.out_total_degree

    def cal_delta_modu(self, m, nodex, graph):
        assert isinstance(nodex, CommunitySuperNode)
        if len(self.community) == 0:
            return -1000000
        else:
            #k_i_in  = set(nodex.out_neighbors)  &  set(self.community)
            #k_i_out = set(nodex.in_neighbors) &  set(self.community)
            #d_i_in = sum([ nodex.out_weight[nodex.out_neighbors.index(n)] for n in k_i_in ])
            #d_i_out = sum([ nodex.in_weight[nodex.in_neighbors.index(n)] for n in k_i_out ])
            d_i_in = sum( [nodex.out_neighbors[key] if  graph.community_list[key].to_commu == self.name else 0 \
                           for key in nodex.out_neighbors.keys() ] )
            d_i_out = sum( [nodex.in_neighbors[key] if  graph.community_list[key].to_commu == self.name else 0 \
                           for key in nodex.in_neighbors.keys() ] )

            modu = (d_i_in + d_i_out ) / m 
            modu -= (nodex.out_d * self.in_total_degree + nodex.in_d * self.out_total_degree ) / (m**2)
            return modu

    def remove(self, m, nodex, graph ):
        self.community.remove(nodex.name)
        self.out_total_degree -= nodex.out_d
        self.in_total_degree  -= nodex.in_d
        if len(self.community) == 0:
            return 0
        else:
            return self.cal_delta_modu(m, nodex, graph)
            #self.internals -= 

    def append(self, nodex):
        # only use for phase1, so we do not need to append node_list 
        self.community.append(nodex.name)
        self.out_total_degree += nodex.out_d
        self.in_total_degree  += nodex.in_d

    def merge(self, graph):
        assert len(self.community) > 0, 'Oooops, len of community = {} , seems to be wrong'.format(len(self.community))
        self.new_node_list = []
        for node in self.community:
            member = graph.community_list[node]
            self.new_node_list += member.node_list

            self.in_neighbors.update(member.in_neighbors)
            self.out_neighbors.update(member.out_neighbors)

        self.name =self.community[0]
        community_ = {old: self.name for old in self.community}
        self.community = [self.name]
        self.to_commu = self.name

        return community_

    def update(self, old2new):
        new_in_neighbors  = Counter()
        new_out_neighbors = Counter()

        for neigh, weight in self.in_neighbors.items():
            new_in_neighbors.update( {old2new[neigh]: weight})
        for neigh, weight in self.out_neighbors.items():
            new_out_neighbors.update({old2new[neigh]: weight})

        self.in_neighbors = new_in_neighbors
        self.out_neighbors = new_out_neighbors

        self.in_d =  sum(self.in_neighbors.values())
        self.out_d = sum(self.out_neighbors.values())
            
        # To avoid recursive update, We must modify node_list HERE!
        self.node_list = self.new_node_list
        self.new_node_list = []

class DirectedGraph:
    def __init__(self, graph):
        '''
        Turn the graph with the form list of dictionaries into CommunitySuperNode 
        Input:
            graph: networkx.DiGraph
        '''
        self.community_list = {}
        for node in graph:
            in_neighbor = graph.in_edges(node)
            self.community_list[node] = CommunitySuperNode(node, dict(graph[node]).keys(), 
                                                        [ neigh for (neigh, i) in in_neighbor])

        self.m = graph.number_of_edges()

    def louvain(self, community_num=9, max_iter=30):
        num = 100
        while (num > community_num):
            self._phase1(max_iter=max_iter)
            num = self._phase2()
            print('Community num: ', num)
            if (num < 50): max_iter = 1
            if (num < 10):
                for name, community in self.community_list.items():
                    print(name, '\t', len(community.node_list))
            
        res_ = {}
        for name, community in self.community_list.items():
            res_[name] = community.node_list
        return res_

    def _phase1(self, max_iter=6):
        stop_flag = False
        iter = 0
        # When Modularity converges, stop
        while not stop_flag and iter < max_iter:
            stop_flag = True
            iter += 1
            # iterate very node
            for node in tqdm(self.community_list.values(), desc='Iteration={}'.format(iter)):
                cur_community = self.community_list[node.to_commu]

                gain_M = {}
                gain_M[cur_community.name] = cur_community.remove(self.m, node, self)
                # test every gain in Modularity
                neighbors = node.out_neighbors + node.in_neighbors
                for neighbor in neighbors:
                    neighbor_commu = self.community_list[neighbor].to_commu
                    gain_M[neighbor_commu] = self.community_list[neighbor_commu].cal_delta_modu(self.m, node, self)
                
                best_com = max(gain_M, key=gain_M.get)
                self.community_list[best_com].append(node)
                if best_com != node.to_commu: stop_flag = False
                node.to_commu = best_com

    def _phase2(self):
        new_community = {}
        old2new = {}
        # merge the super node
        for node in self.community_list.values():
            if len(node.community) == 0 :
                continue
            old2new.update(node.merge(self))
            new_community[node.name] = node

        # Rewire the edge
        for node in new_community:
            new_community[node].update(old2new)

        #new = {key: self.community_list[key] for key in new_community}
        self.community_list = new_community
        return len(self.community_list)
