import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBF(nn.Module):
    def __init__(self, num_features, betas=2.0, use_gpu=True):
        super(RBF, self).__init__()
        if use_gpu:
            self.betas = nn.Parameter(torch.randn(num_features).cuda())
            self.center = nn.Parameter(torch.randn(num_features, num_features).cuda())
            self.A = nn.Parameter(betas*torch.eye(num_features, num_features).cuda())
        
            self.weight = nn.Parameter(torch.randn(num_features, num_features).cuda())
            self.bias = nn.Parameter(torch.randn(num_features).cuda())
        else:
            self.betas = nn.Parameter(torch.randn(num_features))
            self.center = nn.Parameter(torch.randn(num_features, num_features))
            self.A = nn.Parameter(betas*torch.eye(num_features, num_features))
        
            self.weight = nn.Parameter(torch.randn(num_features, num_features))
            self.bias = nn.Parameter(torch.randn(num_features))

        ## Parameter initialization
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.betas, val=2.0)
        nn.init.uniform_(self.center, a=0.0, b=1.0)
        
    def forward(self, x):
        expanded_center = self.center[None,None,:,:]
        A = self.A + sys.float_info.epsilon
        psi = A.t() * A
        s = x[:,:,:,None].repeat(1,1,1,x.size(2)) - expanded_center
        dist = torch.sqrt(torch.sum(torch.tensordot(s, psi, dims=1) * s, dim=-1))
        mahalanobis = torch.exp(-self.betas * dist)
        
        out = torch.tensordot(mahalanobis, self.weight, dims=1)
        g = out + self.bias
        return g
    
class Kernel_trick(nn.Module):
    def __init__(self, num_classes):
        super(Kernel_trick, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=3)
        size = 14*14
        self.rbf1 = RBF(size)
        
        self.conv2 = nn.Conv2d(16*2, 32, kernel_size=6, stride=2)
        size = 5*5
        self.rbf2 = RBF(size)
        
        self.conv3 = nn.Conv2d(32*2, 32, kernel_size=5, stride=1)
        size = 1*1
        self.rbf3 = RBF(size)
        
        self.classifier = nn.Linear(32*2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        b = x.size(0)
        h = self.relu(self.conv1(x))
        g = self.rbf1(h.flatten(start_dim=2)).view(b,-1,h.size(2),h.size(3))
        
        h_cat = torch.cat((h, g), dim=1)
        h = self.relu(self.conv2(h_cat))
        g = self.rbf2(h.flatten(start_dim=2)).view(b,-1,h.size(2),h.size(3))
        
        h_cat = torch.cat((h, g), dim=1)
        h = self.relu(self.conv3(h_cat))
        g = self.rbf3(h.flatten(start_dim=2)).view(b,-1,h.size(2),h.size(3))
        
        h_cat = torch.cat((h, g), dim=1)
        out = self.classifier(h_cat.flatten(start_dim=1))
        return out, g
        