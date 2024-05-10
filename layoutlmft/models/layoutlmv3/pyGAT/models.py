import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer, SpGraphAttentionLayer
from pdb import set_trace as bp

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, attention_mask = None):
        x_out = []
        for i in range(x.shape[0]):
            x_ = F.dropout(x[i,:, :], self.dropout, training=self.training)
            adj_ = adj[i,:,:]*attention_mask[i]
            adj_.fill_diagonal_(1)
            # adj_ = adj[i,:,:]
            x_ = torch.cat([att(x_, adj_) for att in self.attentions], dim=1)
            x_ = F.dropout(x_, self.dropout, training=self.training)
            x_ = F.elu(x_)
            # x_ = F.elu(self.out_att(x_, adj[i,:,:]))
            x_out.append(x_)
            
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(x)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return torch.stack(x_out)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
