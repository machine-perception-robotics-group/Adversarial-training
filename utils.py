import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import lenet, vgg, resnet, wideresnet, rbf_cnn

def get_network(model_type, num_classes):
    # ResNet-18/34/50/101/152
    if model_type == 'resnet18':
        model = resnet.resnet18(num_classes=num_classes)
    elif model_type == 'resnet34':
        model = resnet.resnet34(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = resnet.resnet50(num_classes=num_classes)
    elif model_type == 'resnet101':
        model = resnet.resnet101(num_classes=num_classes)
    elif model_type == 'resnet152':
        model = resnet.resnet152(num_classes=num_classes)
    # Wide ResNet34-10
    elif model_type == 'wrn34-10':
        model = wideresnet.wrn34_10(num_classes=num_classes)
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
