import os
import sys
sys.path.append('..')
import  numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score, mutual_info_score, homogeneity_score, completeness_score
from bhtsne import tsne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import lenet, vgg, resnet, wideresnet, rbf_cnn

def get_network(model_type, num_classes, use_pc=False, is_train=True):
    # ResNet-18/34/50/101/152
    if model_type == 'resnet18':
        model = resnet.resnet18(num_classes=num_classes, use_pc=use_pc)
    elif model_type == 'resnet34':
        model = resnet.resnet34(num_classes=num_classes, use_pc=use_pc)
    elif model_type == 'resnet50':
        model = resnet.resnet50(num_classes=num_classes, use_pc=use_pc)
    elif model_type == 'resnet101':
        model = resnet.resnet101(num_classes=num_classes, use_pc=use_pc)
    elif model_type == 'resnet152':
        model = resnet.resnet152(num_classes=num_classes, use_pc=use_pc)
    # Wide ResNet34-10
    elif model_type == 'wrn34-10':
        model = wideresnet.wrn34_10(num_classes=num_classes, use_pc=use_pc, is_train=is_train)
    # LeNet5
    elif model_type == 'lenet5':
        model = lenet.Lenet5(num_classes=num_classes)
    # VGG-11/13/16/19 (w/ or w/o bn)
    elif model_type == 'vgg11':
        model = vgg.vgg11(num_classes=num_classes)
    elif model_type == 'vgg11_bn':
        model = vgg.vgg11_bn(num_classes=num_classes)
    elif model_type == 'vgg13':
        model = vgg.vgg13(num_classes=num_classes)
    elif model_type == 'vgg13_bn':
        model = vgg.vgg13_bn(num_classes=num_classes)
    elif model_type == 'vgg16':
        model = vgg.vgg16(num_classes=num_classes)
    elif model_type == 'vgg16_bn':
        model = vgg.vgg16_bn(num_classes=num_classes)
    elif model_type == 'vgg19':
        model = vgg.vgg19(num_classes=num_classes)
    elif model_type == 'vgg19_bn':
        model = vgg.vgg19_bn(num_classes=num_classes)
    # RBF kernel cnn
    elif model_type == 'rbf_cnn':
        model = rbf_cnn.Kernel_trick(num_classes=num_classes)
    else:
        assert 0, 'Error: %s is not supported.'
    
    return model

def get_dataloader(dataset, batch_size, tiny_imagenet_path=None, image_size=32):
    if dataset == 'svhn' or dataset == 'mnist':
        training_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor()])
    else:
        training_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        
    if dataset == 'svhn':
        train_dataset = datasets.__dict__[dataset.upper()]('./data', split='train', download=True, transform=training_transforms)
        test_dataset = datasets.__dict__[dataset.upper()]('./data', split='test', download=True, transform=transforms.ToTensor())
    elif dataset == 'tiny-imagenet':
        train_dataset = datasets.ImageFolder(os.path.join(tiny_imagenet_path, 'train'), training_transforms)
        test_dataset = datasets.ImageFolder(os.path.join(tiny_imagenet_path, 'val'), transforms.ToTensor())
    else:
        train_dataset = datasets.__dict__[dataset.upper()]('./data', train=True, download=True, transform=training_transforms)
        test_dataset = datasets.__dict__[dataset.upper()]('./data', train=False, download=True, transform=transforms.ToTensor())
        
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_dataloader


#################################
# Loss functions
#  - center loss
#  - pc loss
#################################
class Proximity(nn.Module):

    def __init__(self, num_classes=100, feat_dim=1024, use_gpu=True):
        super(Proximity, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
            
        dist = (distmat*mask).clamp(min=1e-12, max=1e+12)
        loss = torch.sum(dist) / batch_size
        return loss
    
class Con_Proximity(nn.Module):

    def __init__(self, num_classes=100, feat_dim=1024, use_gpu=True):
        super(Con_Proximity, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())#100 x feats- for 100 centers
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        mask_inv = 1 - mask.float()
        dist = (distmat*mask_inv).clamp(min=1e-12, max=1e+12)
        loss = torch.sum(dist) / (batch_size*(self.num_classes - 1))
        
        ### ??????
        #cc = mask.float().unsqueeze(2).repeat(1,1,self.feat_dim) * self.centers.unsqueeze(0).repeat(batch_size,1,1)
        #nc = mask_inv.unsqueeze(2).repeat(1,1,self.feat_dim) * self.centers.unsqueeze(0).repeat(batch_size,1,1)
        #dist_centroid = torch.pow(cc - nc, 2).clamp(min=1e-12, max=1e+12)
        #loss = (torch.sum(dist) + torch.sum(dist_centroid)) / (batch_size*(self.num_classes - 1))
        return loss
    
#################################################
# Evaluation clustering
#  - Silhouette coefficient
#  - Calinski and Harabasz index
#  - Homogeneity score
#  - Completeness score
#
# Visualizing low level features with t-SEN
#################################################
def extract_features(model, target, inputs):
    feature = None
    
    def forward_hook(module, inputs, outputs):
        global features
        features = outputs.detach().clone()
        
    handle = target.register_forward_hook(forward_hook)
    model.eval()
    model(inputs)
    
    handle.remove()
    return features

class Eval_cluster:
    def __init__(self, model, dataloader, target_module, num_features):
        self.model = model
        self.dataloader = dataloader
        self.target_module = target_module
        self.num_features = num_features
        
    def _cal_score(self, features, labels):
        """
        Calinski and Harabasz index: T. Calinski and J. Harabasz, 1974.
        
        s = \frac{tr(B_{k})}{tr(W_{k})}\times \frac{n_{E} - k}{k - 1}
        Inter-class variance: W_{k} = \sum_{q=1}^{k}\sum_{x\in C_{q}} (x - c_{q})(x - c_{q})^{\top}
        Intra-class variance: B_{k} = \sum_{q=1}^{k}n_{q}(c_{q} - c_{E})(c_{q} - c_{E})^{\top}
        """
        cal_score = calinski_harabasz_score(features, labels)
        return cal_score
    
    def _sil_score(self, features, labels):
        """
        Silhouette coefficient: P. J. Rousseeuw, 1987.
        
        s^{(i)} = \frac{b^{(i)} - a^{(i)}}{\max(a^{(i)}, b^{(i)})}
        Compactness of intra-class: a^{(i)} = \frac{1}{|C_{in}| - 1}\sum_{x^{(j)}\in C_{in}}\left\|x^{(i)} - x^{(j)}\right\|
        Inter-class variance: b^{(i)} = \frac{1}{|C_{near}| - 1}\sum_{x^{(j)}\in C_{near}}\left\|x^{(i)} - x^{(j)}\right\|
        """
        sil_score = silhouette_score(features, labels)
        return sil_score
    
    def _homo_score(self, ground_truth, predicted):
        """
        Homogeneity score
        
        h =
          \begin{cases}
            1 & \mathrm{if~} H(C,K) = 1\\
            1 - \frac{H(C|K)}{H(C)} & \mathrm{else}
          \end{cases}
        H(C|K) = -\sum_{k=1}^{|K|}\sum_{c=1}^{|C|}\frac{a_{c,k}}{N}\log \frac{a_{c,k}}{\sum_{c=1}^{|C|}a_{c,k}}
        H(C) = -\sum_{c=1}^{|C|}\frac{\sum_{k=1}^{|K|}a_{c,k}}{n}\log \frac{\sum_{k=1}^{|K|}a_{c,k}}{n}
        """
        homo_score = homogeneity_score(ground_truth, predicted)
        return homo_score
    
    def _comp_score(self, ground_truth, predicted):
        """
        Completeness score
        
        c =
          \begin{cases}
            1 & \mathrm{if~} H(K,C) = 0\\
            1 - \frac{H(K|C)}{H(K)} & \mathrm{else}
          \end{cases}
        H(K|C) = -\sum_{c=1}^{|C|}\sum_{k=1}^{|K|}\frac{a_{c,k}}{N}\log \frac{a_{c,k}}{\sum_{k=1}^{|K|}a_{c,k}}
        H(K) = -\sum_{k=1}^{|K|}\frac{\sum_{c=1}^{|C|}a_{c,k}}{n}\log \frac{\sum_{c=1}^{|C|}a_{c,k}}{n}
        """
        comp_score = completeness_score(ground_truth, predicted)
        return comp_score
        
    def computing_score(self):
        predicted_labels = torch.zeros(len(self.dataloader.dataset))
        ground_truth = torch.zeros(len(self.dataloader.dataset))
        low_level_features = torch.zeros(len(self.dataloader.dataset), self.num_features).to(torch.float64)
        
        cnt = 0
        for idx, (inputs, target) in enumerate(self.dataloader):
            inputs, target = inputs.cuda(), target.cuda()
            batch = inputs.size(0)
            
            with torch.no_grad():
                pred = self.model(inputs).softmax(dim=1).argmax(dim=1).data.cpu()
            predicted_labels[cnt:cnt+batch] = pred
            ground_truth[cnt:cnt+batch] = target.data.cpu()
            
            features = extract_features(self.model, self.target_module, inputs).view(batch, -1)
            low_level_features[cnt:cnt+batch] = features.data.cpu()
            cnt += batch
        
        sil_score = self._sil_score(low_level_features.numpy(), ground_truth.numpy())
        cal_score = self._cal_score(low_level_features.numpy(), ground_truth.numpy())
        homo_score = self._homo_score(ground_truth.numpy(), predicted_labels.numpy())
        comp_score = self._comp_score(ground_truth.numpy(), predicted_labels.numpy())
        
        print('Sil. score: %.6f' % sil_score)
        print('Cal. score: %.6f' % cal_score)
        print('Homo score: %.6f' % homo_score)
        print('Comp score: %.6f' % comp_score)
        
    def visualize_low_level_features(self, save_name='figures', random_seed=-1, plot_size=0.5):
        os.makedirs(save_name, exist_ok=True)
        ground_truth = torch.zeros(len(self.dataloader.dataset))
        low_level_features = torch.zeros(len(self.dataloader.dataset), self.num_features).to(torch.float64)
        
        cnt = 0
        for idx, (inputs, target) in enumerate(self.dataloader):
            inputs, target = inputs.cuda(), target.cuda()
            batch = inputs.size(0)
            
            ground_truth[cnt:cnt+batch] = target.data.cpu()
            features = extract_features(self.model, self.target_module, inputs).view(batch, -1)
            low_level_features[cnt:cnt+batch] = features.data.cpu().to(torch.float64)
            cnt += batch
        
        X = tsne(low_level_features.numpy(), dimensions=2, perplexity=30, rand_seed=random_seed)
        xcoords, ycoords = X[:,0], X[:,1]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colorlist = [colors[int(index.item())] for index in ground_truth]
        
        plt.scatter(xcoords, ycoords, color=colorlist, s=plot_size)
        plt.savefig((os.path.join(save_name, 't-SNE-scatter.png')))
        plt.savefig((os.path.join(save_name, 't-SNE-scatter.pdf')))
        
#################################################
# Computing confision matrix
#   - standard confision matrix
#   - adversarial confision matrix
#################################################
class Confision_matrix:
    def __init__(self, model, dataloader, epsilon, alpha, num_classes, classes=None):
        self.model = model
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_classes = num_classes
        
        if classes is not None:
            self.classes = classes
        else:
            self.classes = np.arange(num_classes)
        
    def _make_heatmap(self, X, save_dir='figures', save_name='standard_conf_mat', title=None):
        os.makedirs(save_dir, exist_ok=True)
        data = pd.DataFrame(data=X.data.cpu().numpy(), index=self.classes, columns=self.classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(data, square=True, cbar=True, annot=True, cmap='turbo')
        
        if title is not None:
            plt.title(title)
        plt.ylabel('Ground truth', size=13)
        plt.xlabel('Predicted label', size=13)
        plt.savefig(os.path.join(save_dir, save_name+'.png'))
        plt.savefig(os.path.join(save_dir, save_name+'.pdf'))
        
    def standard(self):
        conf_mat = torch.zeros(self.num_classes, self.num_classes).cuda()
        for (inputs, targets) in tqdm(self.dataloader):
            inputs, targets = inputs.cuda(), targets
            
            with torch.no_grad():
                logits = self.model(inputs)
            pred_idx = logits.softmax(dim=1).argmax(dim=1)
            org_onehot = torch.eye(self.num_classes)[targets].cuda()
            pre_onehot = torch.eye(self.num_classes)[pred_idx].cuda()
            conf_mat += (pre_onehot.t()@org_onehot)
            
        print('Standard confision matrix')
        print(conf_mat.data.cpu())
        self._make_heatmap(conf_mat, save_name='standard_conf_mat', title='standard confision matrix')
        
    def _fgsm_attk(self, inputs, t, is_noise=True, norm='linf'):
        xent = nn.CrossEntropyLoss()
        if is_noise:
            noise = torch.FloatTensor(inputs.shape).uniform_(-self.epsilon, self.epsilon).cuda()
            x = torch.clamp(inputs+noise, min=0, max=1)
        else:
            x = inputs.clone()
        x.requires_grad_()
        logits = self.model(x)
        loss = xent(logits, t)
        loss.backward()
        grads = x.grad.data
        if norm == 'l2':
            norm_grads = grads/torch.norm(grads, p=2)
            x = inputs.detach() + self.epsilon*norm_grads.detach()
        elif norm == 'linf':
            x = inputs.detach() + self.epsilon*torch.sign(grads).detach()
        return x.clamp(min=0, max=1)
    
    def _pgd_attk(self, inputs, t, n_steps, is_noise=True, norm='linf'):
        xent = nn.CrossEntropyLoss()
        if is_noise:
            noise = torch.FloatTensor(inputs.shape).uniform_(-self.epsilon, self.epsilon).cuda()
            x = torch.clamp(inputs+noise, min=0, max=1)
        else:
            x = inputs.clone()
        
        for _ in range(n_steps):
            x.requires_grad_()
            logits = self.model(x)
            loss = xent(logits, t)
            loss.backward()
            grads = x.grad.data
            if norm == 'l2':
                norm_grads = grads/torch.norm(grads, p=2)
                x = x.detach() + self.epsilon*norm_grads.detach()
            elif norm == 'linf':
                x = x.detach() + self.alpha*torch.sign(grads).detach()
                x = torch.min(torch.max(x, inputs-self.epsilon), inputs+self.epsilon)
        return x.clamp(min=0, max=1)
        
    def adversarial(self, attk='pgd', norm='linf'):
        conf_mat = torch.zeros(self.num_classes, self.num_classes).cuda()
        for (inputs, targets) in tqdm(self.dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            if attk == 'fgsm':
                x = self._fgsm_attk(inputs, targets, is_noise=True, norm=norm)
            elif attk == 'pgd':
                x = self._pgd_attk(inputs, targets, n_steps=10, is_noise=True, norm=norm)
            else:
                assert 0, '%s is not supported.'
            
            with torch.no_grad():
                logits = self.model(x)
            pred_idx = logits.softmax(dim=1).argmax(dim=1)
            org_onehot = torch.eye(self.num_classes)[targets].cuda()
            pre_onehot = torch.eye(self.num_classes)[pred_idx].cuda()
            conf_mat += (pre_onehot.t()@org_onehot)
            
        print('Adversarial confision matrix (%s)' % attk)
        print(conf_mat.data.cpu())
        self._make_heatmap(conf_mat, save_name='standard_conf_mat_%s' % attk, 
                           title='adversarial confision matrix (%s)' % attk)
        
        
        