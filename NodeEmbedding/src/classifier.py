import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

class NNClf(nn.Module):
    def __init__(self, embedding_size):
        super(NNClf, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(embedding_size, 64),
                        nn.ReLU(),
                        nn.Linear(64,64),
                        nn.ReLU(),
                        nn.Linear(64,1),
                        nn.Sigmoid()
        )
    
    def forward(self, emb):
        return self.model(emb)
    
    def fit(self, emb, label, epoch=15, batch_size=256, lr = 0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for e in tqdm(range(epoch)):
            batch_num = int(emb.shape[0] / batch_size )
            for b in range(batch_num):
                if (b+1) * batch_size > emb.shape[0]: break
                x = emb[b * batch_size: (b+1) * batch_size, :]
                y = label[b * batch_size: (b+1) * batch_size]
                loss = F.binary_cross_entropy(self.model(x).squeeze(), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (b % 50 == 0):
                    print('epoch={}, Loss = {}'.format(e, loss))

    def predict(self, emb):
        self.model.eval()
        return self.forward(emb)

        
