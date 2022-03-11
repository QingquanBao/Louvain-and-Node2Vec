import torch
from torch import nn

class LogLikelihood(nn.Module):
    def __init__(self):
        super(LogLikelihood, self).__init__()
    
    def forward(self, u:torch.Tensor, v:torch.Tensor, neighbor:torch.Tensor):
        if len(u.shape) == 1:
            _out = torch.log10( torch.sigmoid(torch.matmul(u.unsqueeze(0) , v.transpose(1,0)))).sum()
            _out -= torch.log10( torch.sigmoid(torch.matmul(u.unsqueeze(0) , neighbor.T))).sum()
        else:
            _out = torch.log10( torch.sigmoid(1. * torch.matmul(u.unsqueeze(1) , v.transpose(1,2)))).sum()
            _out += torch.log10( torch.sigmoid(-1. * torch.matmul(u.unsqueeze(1) , neighbor.transpose(1,2)))).sum()
            #_out = torch.log10( torch.sigmoid(torch.matmul( u , v.transpose(1,0)))).sum()
            #_out -= torch.log10( torch.sigmoid(torch.matmul(u , neighbor.transpose(1,0)))).sum()

        return   -1. * _out / u.shape[0]