from layers import *
import torch.nn.init as init
import scipy.sparse as sp




class GCN(nn.Module):
    def __init__(self, node_count, node_dim, n_representation,dropout=0.3):
        super(GCN, self).__init__()
        self.n_feature = node_dim
        self.n_hidden = 128
        self.n_representation = n_representation

        #self.embedding = nn.Embedding(node_count, self.n_feature)
        self.gc1 = GraphConvolution(self.n_feature, self.n_hidden)
        self.gc2 = GraphConvolution(self.n_hidden, self.n_representation)
        self.dropout = nn.Dropout(p=0.2)
        self.tmp_linear = nn.Linear(self.n_feature, self.n_representation)
        self.init_weights()

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = self.gc2(x1, adj)
        out = x1

        return out

    def init_weights(self):
        init.xavier_uniform_(self.tmp_linear.weight)
        init.xavier_uniform_(self.gc1.weight)
        init.xavier_uniform_(self.gc2.weight)


        
class Link_Prediction(nn.Module):
    def __init__(self, n_representation, hidden_dims=[128, 32], dropout=0.3):
        super(Link_Prediction, self).__init__()
        self.n_representation = n_representation
        self.linear1 = nn.Linear(2*self.n_representation, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, x1, x2):
        x = torch.cat((x1,x2),1) # N * (2 node_dim)
        
        x = F.relu(self.linear1(x)) # N * hidden1_dim
        x = self.dropout(x)
        x = F.relu(self.linear2(x)) # N * hidden2_dim
        x = self.dropout(x)
        x = self.linear3(x) # N * 2
        x = self.sigmoid(x) # N * ( probility of each event )
        return x

    def init_weights(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)

