import networkx as nx
import random
import torch
import torch.nn.functional as F
import pandas as pd 
from embedding import Node2VecEmbedding
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def construct_graph(file_name):
    df = pd.read_csv(file_name)
    G = nx.from_pandas_edgelist(df)
    return G

class Dataloader:
    def __init__(self, graph, test_csv, data_ratio=0.8):
        self.graph = graph
        self.data_ratio = data_ratio

        df = pd.read_csv(test_csv, usecols=[1,2])
        self.test_pos = df.values.tolist()
    
    def train_val_split(self):
        T = nx.minimum_spanning_tree(self.graph)
        self.T = self.graph.copy()
        self.train_edge_num = int(len(self.graph.edges) * self.data_ratio)
        val_edge_num = len(self.graph.edges) - self.train_edge_num
        self.val_edge = []

        edge_list = list(self.graph.edges)
        spanning_edge = list(T.edges)
        random.shuffle(edge_list)
        for e in edge_list:
            if e in spanning_edge: continue
            else:
                self.val_edge.append(e)
                self.T.remove_edge(*e)
                if (len(self.val_edge) >= val_edge_num):
                    break
        
        print('Split Done')
        
        return self.T

    def train_data(self, train_val_ratio=1.0 ):
        if train_val_ratio != 1.0:
            print('Train_val_ratio is deprecated, u d better set 1.0')

        pos = list(self.T.edges)
        inuse_data_num = int(len(pos) * self.data_ratio)
        pos_num = int(train_val_ratio * inuse_data_num)

        random.shuffle(pos)
        neg = self.get_neg_link(inuse_data_num)

        pos_ = pos[0: pos_num]
        neg_ = neg[0: pos_num] 
        x = torch.tensor( pos_ + neg_ , dtype=torch.int64)
        y = torch.concat((torch.ones(pos_num), torch.zeros(pos_num)))

        return x, y
    
    def get_neg_link(self, pos_num, data_domain='train'):
        if data_domain == 'train':
            graph = self.T
        elif data_domain == 'val':
            graph = self.graph

        neg_ = []
        graph_size = len(graph)
        print('Preparing ' + data_domain + ' data')
        with tqdm(total=pos_num) as pbar:
            while (len(neg_) < pos_num):
                for u in graph:
                    v = random.randint(0, graph_size)
                    if v not in graph[u].keys():
                        neg_.append((u, v))
                        pbar.update(1)
                    if (len(neg_) >= pos_num):
                        break
        return neg_

    def val_data(self):
        val_pos_ = self.val_edge
        val_num = len(val_pos_)
        val_neg_ = self.get_neg_link(val_num, 'val')

        val_x = torch.tensor( val_pos_ + val_neg_ , dtype=torch.int64)
        val_y = torch.concat((torch.ones(val_num), torch.zeros(val_num)))
        return val_x, val_y

    def test_data(self, emb):
        x_ = torch.tensor(self.test_pos)
        return x_

def save_csv(res, outpath):
    with open(outpath, 'w') as f:
        for i, pred in enumerate(res):
            f.write('{}, {} \n'.format(i, pred)) 

def emb_pred(emb, labels):
    src, tgt = labels[:, 0], labels[:, 1]
    src_emb = emb(torch.LongTensor(src))
    tgt_emb = emb(torch.LongTensor(tgt))
    pred_ = F.cosine_similarity(src_emb, tgt_emb)
    return (pred_ + 1. ) / 2.

def evaluate(pred, gt, is_vis=False, model_name='Emb cosine sim', title=''):
    fpr, tpr, _ = roc_curve(gt.detach().numpy(), pred.detach().numpy())
    roc_auc = auc(fpr, tpr)

    if is_vis:
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Model " + model_name)
        plt.legend(loc="lower right")
        plt.savefig(title +'ROC.png')
    return roc_auc

def main(pretrained=None):
    train_val_ratio = 1.0

    p = 0.5
    q = 2
    sample_len = 30
    k = 10
    embedding_size = 50
    batch_size = 64
    lr = 0.3
    train_iter = 4000

    G = construct_graph('data/lab2_edge.csv')
    loader = Dataloader(G, 'data/lab2_test.csv', data_ratio=0.95)
    T = loader.train_val_split()
    emb = Node2VecEmbedding(max(G)+1, embedding_size, T, p=p, q=q, sample_len=sample_len, k=k)
    #optimizer = torch.optim.Adam(emb.parameters(), lr=lr)
    optimizer = torch.optim.SGD(emb.parameters(), lr=lr, momentum=0.98)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 2000, 5000], gamma=0.5)

    if pretrained is None:
        # train the embedding
        node_list = list(G.nodes)
        loss_list = []
        for i in tqdm(range(train_iter)):
            batch = random.sample(node_list, k=batch_size)
            #batch = torch.LongTensor(random.sample(node_list, k=batch_size))
            loss = emb.train(batch, optimizer, scheduler)
            if i % 30 == 0 : 
                loss_list.append(loss.detach().numpy())
                #print(loss)
        
        plt.plot(loss_list)
        plt.savefig('loss_embsize{}_lr{}_momentum0.98_mile500.png'.format(embedding_size, lr))
        torch.save(emb.state_dict(), 'trial.pt')
    else:
        emb.load_state_dict(torch.load(pretrained))

    emb.eval()

    # load Link da
    train_x_label, train_y = loader.train_data(train_val_ratio=train_val_ratio)
    val_x_label, val_y = loader.val_data()
    # Test Embedding cosine similarity
    pred = emb_pred(emb, train_x_label)
    auc_  = evaluate(pred, train_y, True, title='Train data')
    print('Training', auc_)

    val_pred = emb_pred(emb, val_x_label)
    save_csv(val_pred, 'Valtion.csv')
    val_auc_  = evaluate(val_pred, val_y, True, title='Val data')
    print('Validation', val_auc_)

    # Generate test
    test_x = loader.test_data(emb)
    test_pred = emb_pred(emb, test_x)
    save_csv(test_pred, 'Prediction.csv')

    '''
    # train the classifier
    from sklearn import svm
    from classifier import NNClf
    clf = svm.SVC()
    #clf = NNClf(embedding_size*2)
    train_x_label, train_y, val_x_label, val_y= loader.train_data(train_val_ratio=train_val_ratio)
    train_x = torch.cat((emb(train_x_label[:,0]), emb(train_x_label[:,1])), dim=1)

    #clf.fit(train_x.detach().numpy(), train_y)
    clf.fit(train_x.detach(), train_y)

    # predict
    y = clf.predict(train_x.detach())
    print( "Train acc = ", (y == train_y.numpy()).sum() / len(y))
    #print( "Train acc = ", ((y > 0.5).squeeze() == train_y).sum() / len(y))

    # val
    val_x = torch.cat((emb(val_x_label[:,0]), emb(val_x_label[:,1])), dim=1)
    y = clf.predict(val_x.detach())
    print( "Validation acc = ", (y == val_y.numpy()).sum() / len(y))
    #print( "Validation acc = ", ((y > 0.5).squeeze() == val_y).sum() / len(y))

    # Test 
    #test_x = loader.test_data(emb)
    #test_y = clf.predict(test_x.detach())
    '''


if __name__ == '__main__':
    #main('final.pt')    
    main()
