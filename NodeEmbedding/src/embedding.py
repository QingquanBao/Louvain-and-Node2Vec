import torch
from torch import nn
import random
import numpy as np
from torch.autograd import grad
from loss import LogLikelihood
from multiprocessing import Pool
random.seed(10086)

class RandomWalker:
    def __init__(self, graph, p=1, q=1, sample_len=20, k=10):
        self.graph = graph
        self.p = p
        self.q = q
        self.sample_len = sample_len
        self.k = k

    def sample(self, u:int):
        neighlist = torch.zeros( self.sample_len, dtype=torch.int64)

        parent_neighbors = []
        parent = None
        cur_node = u
        for i in range(self.sample_len):
            neighbors = list(self.graph[u].keys())
            neighbor_prob = self._weight_map(cur_node, parent, parent_neighbors)
            neighlist[i] = random.choices(neighbors, weights=neighbor_prob, k=1)[0]

            parent = cur_node
            parent_neighbors = neighbors
        
        return neighlist
        #return torch.tensor(neighlist, dtype=torch.int64)

    def neg_samples(self , batch=1):
        size = self.k * self.sample_len
        neg_samples = np.random.choice(list(self.graph.nodes), size=(batch, size), replace=True)
        #samples = random.sample(list(self.graph.nodes), k=size)
        return torch.tensor(neg_samples, dtype=torch.int64)

    def _weight_map(self, u, parent, parent_neighbors):
        weight = []        
        for v in self.graph[u].keys():
            if v == parent:
                weight.append(1 / self.p)
            elif v in parent_neighbors:
                weight.append(1)
            else:
                weight.append(1 / self.q)
        
        return np.array(weight) / sum(weight)


class Node2VecEmbedding(nn.Module):
    def __init__(self, node_size, embedding_size, graph, p, q, sample_len=20, k=10):
        super(Node2VecEmbedding, self).__init__()
        self.embedding = nn.Embedding(node_size, embedding_size, max_norm=7)
        self.random_walker = RandomWalker(graph, p, q, sample_len, k)
        self.embedding_size = embedding_size

    def forward(self, node_id):
        return self.embedding(node_id)

    def train(self, batch:list, optimizer, scheduler=None):
        self.embedding.train()
        '''
        with Pool(4) as p:
            neighbor_list = p.map(self.random_walker.sample, batch)
            neg_sample_list = p.map(self.random_walker.neg_samples , batch)

        '''
        neighbor_list = []
        for i in batch:
            neighbor = self.random_walker.sample(i)
            neighbor_list.append(neighbor)

        with torch.enable_grad():
            neighbor_emb = self.forward(torch.stack(neighbor_list))
            negative_samples = self.random_walker.neg_samples(len(batch))
            neg_samples_emb = self.forward(negative_samples)

            emb = self.forward(torch.LongTensor(batch))

            loss = LogLikelihood()(emb, neighbor_emb, neg_samples_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None: 
                scheduler.step()

        return loss
    
    def eval(self):
        self.embedding.eval()


class DeepWalk(Node2VecEmbedding):
    def __init__(self, node_size, embedding_size, graph, sample_len):
        super(DeepWalk, self).__init__(node_size, embedding_size, graph, 1, 1, sample_len)


