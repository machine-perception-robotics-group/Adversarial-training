import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from advertorch.utils import CarliniWagnerLoss
from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.attacks import FGSM, LinfPGDAttack, L2PGDAttack
from autoattack import AutoAttack
from utils import get_network

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
                logits = model(x)
            loss = xent(logits, targets)
            loss.backward()
            grads = x.grad.data
            x = x.detach() + alpha*torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs-epsilon), inputs+epsilon).clamp(min=lower, max=upper)
        
        with torch.no_grad():
            logits_nat = model(inputs)
            logits_adv = model(x)
            
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
            logits = model(inputs)
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (nat): %.4f' % avg_acc)

def robustness_fgsm(model, dataloader, epsilon, lower, upper):
    total_correct = 0
    fgsm = FGSM(predict=model, 
                loss_fn=nn.CrossEntropyLoss(), 
                eps=epsilon, 
                clip_min=lower, 
                clip_max=upper, 
                targeted=False)
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        x = fgsm(inputs, targets)
        
        with torch.no_grad():
            model.eval()
            logits = model(x)
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (fgsm): %.4f' % avg_acc)
    
def robustness_pgd(model, dataloader, epsilon, alpha, num_repeats, lower, upper, norm='Linf'):
    total_correct = 0
    xent = nn.CrossEntropyLoss()
    if norm == 'Linf':
        pgd = LinfPGDAttack(predict=model,
                            loss_fn=xent,
                            eps=epsilon,
                            nb_iter=num_repeats,
                            eps_iter=alpha,
                            rand_init=True,
                            clip_min=lower,
                            clip_max=upper,
                            targeted=False)
    elif norm == 'L2':
        pgd = L2PGDAttack(predict=model,
                          loss_fn=xent,
                          eps=epsilon,
                          nb_iter=num_repeats,
                          eps_iter=alpha,
                          rand_init=True,
                          clip_min=lower,
                          clip_max=upper,
                          targeted=False)
    else:
        assert 0, "Error"
    
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        x = pgd(inputs, targets)
        
        with torch.no_grad():
            model.eval()
            logits = model(x)
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (pgd): %.4f' % avg_acc)
    
def robustness_pgd_with_cw_loss(model, dataloader, epsilon, alpha, num_repeats, lower, upper, norm='Linf'):
    total_correct = 0
    cw_loss = CarliniWagnerLoss()
    if norm == 'Linf':
        pgd = LinfPGDAttack(predict=model,
                            loss_fn=cw_loss,
                            eps=epsilon,
                            nb_iter=num_repeats,
                            eps_iter=alpha,
                            rand_init=True,
                            clip_min=lower,
                            clip_max=upper,
                            targeted=False)
    elif norm == 'L2':
        pgd = L2PGDAttack(predict=model,
                          loss_fn=cw_loss,
                          eps=epsilon,
                          nb_iter=num_repeats,
                          eps_iter=alpha,
                          rand_init=True,
                          clip_min=lower,
                          clip_max=upper,
                          targeted=False)
    else:
        assert 0, "Error"
        
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        x = pgd(inputs, targets)
        
        with torch.no_grad():
            model.eval()
            logits = model(x)
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (pgd w/ cw loss): %.4f' % avg_acc)
    
def robustness_sa(model, dataloader, epsilon, alpha, num_repeats, lower, upper, tau, norm='Linf'):
    total_correct = 0
    xent = nn.CrossEntropyLoss()
    if norm == 'Linf':
        pgd = LinfPGDAttack(predict=model,
                            loss_fn=xent,
                            eps=epsilon,
                            nb_iter=num_repeats,
                            eps_iter=alpha,
                            rand_init=True,
                            clip_min=lower,
                            clip_max=upper,
                            targeted=False)
    elif norm == 'L2':
        pgd = L2PGDAttack(predict=model,
                          loss_fn=xent,
                          eps=epsilon,
                          nb_iter=num_repeats,
                          eps_iter=alpha,
                          rand_init=True,
                          clip_min=lower,
                          clip_max=upper,
                          targeted=False)
    else:
        assert 0, "Error"
    
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        x = pgd(inputs, targets)
        
        batch = x.size(0)
        with torch.no_grad():
            model.eval()
            logits = model(x)
            
        probs = torch.softmax(logits, dim=1)
        classes = torch.arange(logits.size(1))[None,:].repeat(batch,1).cuda()
        wrong_probs = probs[classes!=targets[:,None]].view(batch, logits.size(1)-1)
        correct_probs = probs[classes==targets[:,None]].unsqueeze(1)
        top2_probs = torch.topk(wrong_probs, k=1, dim=1).values
        margin = correct_probs - top2_probs
        s = torch.exp(-tau*margin)
        s = s/torch.sum(s)
            
        total_correct += torch.sum(s*logits.softmax(dim=1).argmax(dim=1).eq(targets)).item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (pgd sa): %.4f' % avg_acc)
    
def robustness_tr(model, dataloader, epsilon, alpha, num_repeats, lower, upper, tau, is_noise=True, norm='linf'):
    total_correct = 0
    xent = nn.CrossEntropyLoss()
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch = inputs.size(0)
        if is_noise:
            noise = torch.FloatTensor(inputs.shape).uniform_(-1, 1).cuda()
            x = torch.clamp(inputs+noise, min=lower, max=upper)
        else:
            x = inputs.clone()
        
        for _ in range(num_repeats):
            x.requires_grad_()
            with torch.enable_grad():
                model.train()
                logits = model(x)
            
            classes = torch.arange(logits.size(1))[None,:].repeat(batch,1).cuda()
            probs = torch.softmax(logits, dim=1)
            wrong_probs = probs[classes!=targets[:,None]].view(batch, logits.size(1)-1)
            correct_probs = probs[classes==targets[:,None]].unsqueeze(1)
            top2_probs = torch.topk(wrong_probs, k=1, dim=1).values
            margin = correct_probs - top2_probs
            s = torch.exp(-tau*margin)
            loss = torch.sum(-s*torch.log_softmax(logits, dim=1)[classes==targets[:,None]])/batch
            loss.backward()
            grads = x.grad.data
            if norm == 'l2':
                norm_grads = grads/torch.norm(grads, p=2)
                x = x.detach() + alpha*norm_grads.detach()
            elif norm == 'linf':
                x = x.detach() + alpha*torch.sign(grads).detach()
                x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            delta = x.data - inputs.data
            x = Variable(torch.clamp(inputs.data+delta, min=lower, max=upper), requires_grad=True)
        
        with torch.no_grad():
            model.eval()
            logits = model(x)
            
        probs = torch.softmax(logits, dim=1)
        classes = torch.arange(logits.size(1))[None,:].repeat(batch,1).cuda()
        wrong_probs = probs[classes!=targets[:,None]].view(batch, logits.size(1)-1)
        correct_probs = probs[classes==targets[:,None]].unsqueeze(1)
        top2_probs = torch.topk(wrong_probs, k=1, dim=1).values
        margin = correct_probs - top2_probs
        s = torch.exp(-tau*margin)
        s = s/torch.sum(s)
            
        total_correct += torch.sum(s*logits.softmax(dim=1).argmax(dim=1).eq(targets)).item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (pgd tr): %.4f' % avg_acc)
    
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
        delta = x.data - inputs.data
        x = Variable(torch.clamp(inputs.data+delta, min=lower, max=upper), requires_grad=True)
        
        with torch.no_grad():
            logits = model(x)
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (cw): %.4f' % avg_acc)
    
def robustness_aa(model, dataloader, num_classes, epsilon, lower, upper, norm='Linf'):
    total_correct = 0
    AA = AutoAttack(model, norm=norm, eps=epsilon, version='standard')
    
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch = inputs.size(0)
        x = AA.run_standard_evaluation(inputs, targets, bs=batch)
        
        with torch.no_grad():
            logits = model(x)
            
        total_correct += logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    avg_acc = total_correct/len(dataloader.dataset)
    print('Avg acc (aa): %.4f' % avg_acc)
    
def robustness_aa_individual(model, dataloader, num_classes, epsilon, lower, upper, norm='Linf'):
    total_correct_apgd_ce = 0
    total_correct_apgd_t = 0
    total_correct_fab = 0
    total_correct_square = 0
    AA = AutoAttack(model, norm=norm, eps=epsilon, version='standard')
    
    for idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch = inputs.size(0)
        
        x = AA.run_standard_evaluation_individual(inputs, targets, bs=batch)
        
        with torch.no_grad():
            logits_apgd_ce = model(x['apgd-ce'])
            logits_apgd_t = model(x['apgd-t'])
            logits_fab = model(x['fab-t'])
            logits_square = model(x['square'])
            
        total_correct_apgd_ce += logits_apgd_ce.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        total_correct_apgd_t += logits_apgd_t.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        total_correct_fab += logits_fab.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        total_correct_square += logits_square.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
    
    avg_acc_apgd_ce = total_correct_apgd_ce/len(dataloader.dataset)
    avg_acc_apgd_t = total_correct_apgd_t/len(dataloader.dataset)
    avg_acc_fab = total_correct_fab/len(dataloader.dataset)
    avg_acc_square = total_correct_square/len(dataloader.dataset)
    
    print('Avg acc (APGD-CE): %.4f' % avg_acc_apgd_ce)
    print('Avg acc (APGD-T): %.4f' % avg_acc_apgd_t)
    print('Avg acc (FAB): %.4f' % avg_acc_fab)
    print('Avg acc (Square): %.4f' % avg_acc_square)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--eps', type=int, default=8)
    parser.add_argument('--pc', action='store_true')
    parser.add_argument('--rob_sa', action='store_true')
    parser.add_argument('--rob_tr', action='store_true')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    checkpoint = torch.load(args.checkpoint)
    try:
        state_dict = checkpoint['state_dict']
    except:
        state_dict = checkpoint['state_dict_rob']
    dataset = checkpoint['dataset']
    model_type = checkpoint['model_type']
    try:
        num_classes = checkpoint['num_classes']
    except:
        num_classes = 10
    model = nn.DataParallel(get_network(model_type, num_classes, use_pc=args.pc, is_train=False).cuda())
    model.load_state_dict(state_dict)
    model.eval()
    
    try:
        dataset = datasets.__dict__[dataset.upper()]('./data', download=True, train=False, transform=transforms.ToTensor())
    except:
        dataset = datasets.__dict__[dataset.upper()]('./data', download=True, split='test', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=100, drop_last=False, shuffle=False)
    
    lower, upper = 0, 1
    epsilon = args.eps/255
    alpha = epsilon/4
    num_steps_list = [10, 20]
    
    standard_acc(model, dataloader)
    robustness_fgsm(model, dataloader, epsilon, lower=0, upper=1)
    for n in num_steps_list:
        print('num PGD steps: %d' % n)
        robustness_pgd(model, dataloader, epsilon, alpha, n, lower=0, upper=1, norm=args.norm)

    if args.rob_sa:
        robustness_sa(model, dataloader, epsilon, alpha, 20, lower, upper, tau=1.5, norm=args.norm)
    if args.rob_tr:
        robustness_tr(model, dataloader, epsilon, alpha, 20, lower, upper, tau=1.5, is_noise=True, norm='linf')
    robustness_pgd_with_cw_loss(model, dataloader, epsilon, alpha, 20, lower, upper, norm=args.norm)
    #robustness_aa(model, dataloader, num_classes, epsilon, lower, upper, norm='Linf')
    #robustness_aa_individual(model, dataloader, num_classes, epsilon, lower, upper, norm='Linf')
    #robustness_cw_l2(model, dataloader, args.num_classes, lower=0, upper=1)
    

            