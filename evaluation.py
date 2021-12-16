import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import get_network
from torchvision import transforms, datasets
from advertorch.utils import CarliniWagnerLoss
from advertorch.attacks import CarliniWagnerL2Attack

def validation(model, dataloader, epsilon, alpha, num_repeats, lower, upper):
    model.eval()
    total_correct_nat = 0
    total_correct_rob = 0
    xent = nn.CrossEntropyLoss()
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        noise = torch.FloatTensor(inputs.shape).uniform_(-1, 1).cuda()
        x = torch.clamp(inputs + noise, min=lower, max=upper)
        
        # PGD attack with 10 steps
        for _ in range(num_repeats):
            x.requires_grad_()
            with torch.enable_grad():
                logits = model(x)[0]
            loss = xent(logits, targets)
            loss.backward()
            grads = x.grad.data
            x = x.detach() + alpha*torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs-epsilon), inputs+epsilon).clamp(min=lower, max=upper)
        
        with torch.no_grad():
            logits_nat = model(inputs)[0]
            logits_adv = model(x)[0]
            
        total_correct_nat += logits_nat.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        total_correct_rob += logits_adv.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        if idx % 10 == 0:
            print('[%d/%d] nat: %.4f, : rob: %.4f'\
                  % (idx, len(dataloader.dataset),
                     logits_nat.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()/inputs.size(0),
                     logits_adv.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()/inputs.size(0)))
    
    avg_acc_nat = total_correct_nat/len(dataloader.dataset)
    avg_acc_rob = total_correct_rob/len(dataloader.dataset)
    print('Avg acc (nat): %.4f' % avg_acc_nat)
    print('Avg acc (rob): %.4f' % avg_acc_rob)
    return avg_acc_nat, avg_acc_rob

def standard_acc(model, dataloader):
    total_correct = 0
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
            logits = model(inputs)[0]
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (nat): %.4f' % avg_acc)

def robustness_fgsm(model, dataloader, epsilon, lower, upper, is_noise=True, norm='linf'):
    total_correct = 0
    xent = nn.CrossEntropyLoss()
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        if is_noise:
            noise = torch.FloatTensor(inputs.shape).uniform_(-1, 1).cuda()
            x = torch.clamp(inputs+noise, min=lower, max=upper)
        else:
            x = inputs.clone()
            
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)[0]
        loss = xent(logits, targets)
        loss.backward()
        grads = x.grad.data
        if norm == 'l2':
            norm_grads = grads/torch.norm(grads, p=2)
            x = x.detach() + epsilon*norm_grads.detach()
        elif norm == 'linf':
            x = x.detach() + epsilon*torch.sign(grads).detach()
        x = torch.clamp(x, min=0, max=1)
        
        with torch.no_grad():
            logits = model(x)[0]
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (fgsm): %.4f' % avg_acc)
    
def robustness_pgd(model, dataloader, epsilon, alpha, num_repeats, lower, upper, is_noise=True, norm='linf'):
    total_correct = 0
    xent = nn.CrossEntropyLoss()
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        if is_noise:
            noise = torch.FloatTensor(inputs.shape).uniform_(-1, 1).cuda()
            x = torch.clamp(inputs+noise, min=lower, max=upper)
        else:
            x = inputs.clone()
        
        for _ in range(num_repeats):
            x.requires_grad_()
            with torch.enable_grad():
                logits = model(x)[0]
            loss = xent(logits, targets)
            loss.backward()
            grads = x.grad.data
            if norm == 'l2':
                norm_grads = grads/torch.norm(grads, p=2)
                x = x.detach() + alpha*norm_grads.detach()
            elif norm == 'linf':
                x = x.detach() + alpha*torch.sign(grads).detach()
                x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, min=0, max=1)
        
        with torch.no_grad():
            logits = model(x)[0]
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (pgd): %.4f' % avg_acc)
    
def robustness_pgd_with_cw_loss(model, dataloader, epsilon, alpha, num_repeats, lower, upper, is_noise=True, norm='linf'):
    total_correct = 0
    cw_loss = CarliniWagnerLoss()
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        if is_noise:
            noise = torch.FloatTensor(inputs.shape).uniform_(-1, 1).cuda()
            x = torch.clamp(inputs+noise, min=lower, max=upper)
        else:
            x = inputs.clone()
        
        for _ in range(num_repeats):
            x.requires_grad_()
            with torch.enable_grad():
                logits = model(x)[0]
            loss = cw_loss(logits, targets)
            loss.backward()
            grads = x.grad.data
            if norm == 'l2':
                norm_grads = grads/torch.norm(grads, p=2)
                x = x.detach() + alpha*norm_grads.detach()
            elif norm == 'linf':
                x = x.detach() + alpha*torch.sign(grads).detach()
                x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, min=0, max=1)
        
        with torch.no_grad():
            logits = model(x)[0]
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (pgd w/ cw loss): %.4f' % avg_acc)
    
def robustness_cw_l2(model, dataloader, num_classes, lower, upper):
    total_correct = 0
    cw = CarliniWagnerL2Attack(predict=model, 
                               num_classes=num_classes,
                               confidence=0, 
                               targeted=False, 
                               learning_rate=0.01,
                               binary_search_steps=9, 
                               max_iterations=10000, 
                               abort_early=True, 
                               initial_const=1e-3, 
                               clip_min=lower, 
                               clip_max=upper, 
                               loss_fn=None)
    
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        x = cw.perturb(inputs, targets)
        
        with torch.no_grad():
            logits = model(x)
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (cw): %.4f' % avg_acc)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--norm', type=str, default='linf')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--eps', type=int, default=8)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    try:
        dataset = datasets.__dict__[args.dataset.upper()]('./data', download=True, train=False, transform=transforms.ToTensor())
    except:
        dataset = datasets.__dict__[args.dataset.upper()]('./data', download=True, split='test', transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, drop_last=False, shuffle=False)
    
    try:
        state_dict = torch.load(args.checkpoint)['state_dict']
    except:
        state_dict = torch.load(args.checkpoint)['state_dict_rob']
    model = nn.DataParallel(get_network(args.model_type, args.num_classes).cuda())
    model.load_state_dict(state_dict)
    model.eval()
    
    epsilon = args.eps/255
    alpah = epsilon/4
    num_steps_list = [10, 20, 30, 50]
    
    #standard_acc(mdoel, dataloader)
    #robustness_fgsm(model, dataloader, epsilon, lower=0, upper=1, is_noise=True, norm=args.norm)
    #for n in range(num_steps_list):
    #    print('num PGD steps: %d' % n)
    #    robustness_pgd(model, dataloader, epsilon, alpha, n, lower=0, upper=1, is_noise=True, norm=args.norm)
    robustness_pgd_with_cw_loss(model, dataloader, epsilon, alpha, num_repeats=20, lower=0, upper=1, is_noise=True, norm=args.norm)
    robustness_cw_l2(model, dataloader, args.num_classes, lower=0, upper=1)
    

            