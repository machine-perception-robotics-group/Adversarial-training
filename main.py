import os
import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader

from utils import *
from trainer import trainer
from evaluation import validation
from inner_maximization import Inner_maximize_selector
from outer_minimization import Outer_minimize_selector
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_pytorch', type=int, default=np.random.randint(4294967295))
    parser.add_argument('--seed_numpy', type=int, default=np.random.randint(4294967295))
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--training_type', type=str,
                        choices=['standard', 'alp', 'avmixup', 'clp', 'mail', 'gairat', 'prototype',
                                 'wmmr', 'trades', 'gat', 'mart', 'prob_compact', 'lbgat', 'kernel_trick'])
    args = parser.parse_args()
    
    path_chekpoint = os.path.join(args.checkpoint, args.training_type)
    np.random.seed(args.seed_numpy)
    torch.manual_seed(args.seed_pytorch)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(path_chekpoint, exist_ok=True)
    
    #######################################
    # Load parameters
    #######################################
    yaml_path = './configs/config_%s.yml' % args.training_type
    with open(yaml_path) as yf:
        configs = yaml.safe_load(yf.read())['training']
    print(configs)
    
    dataset = configs['dataset']
    model_type = configs['model_type']
    total_epochs = configs['total_epochs']
    batch_size = configs['batch_size']
    lr = configs['lr']
    lr_pc = configs['lr_pc']
    lr_center = configs['lr_center']
    momentum = configs['momentum']
    weight_decay = configs['weight_decay']
    scheduler = configs['scheduler']
    num_classes = configs['num_classes']
    lower = configs['lower']
    upper = configs['upper']
    alpha = configs['alpha']/255
    epsilon = configs['epsilon']/255
    num_repeats = configs['num_repeats']
    tau = configs['tau']
    lam = configs['lam']
    lambda_av = configs['lambda_av']
    gamma = configs['gamma']
    beta = configs['beta']
    pm_type = configs['pm_type']
    warm_up = configs['warm_up']
    xi = configs['xi']
    wc = configs['wc']
    wp = configs['wp']
    
    if dataset == 'mnist':
        image_size = 28
    else:
        image_size = 32
    #######################################
        
    train_dataset, test_dataloader = get_dataloader(dataset=dataset, batch_size=batch_size, image_size=image_size)
    num_samples = len(train_dataset)
    num_samples_for_train = int(num_samples * 0.98)
    num_samples_for_valid = num_samples - num_samples_for_train
    train_set, valid_set = random_split(train_dataset, [num_samples_for_train, num_samples_for_valid])
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_set, batch_size=1, shuffle=True, drop_last=False)
    
    model_nat = nn.DataParallel(get_network(model_type=model_type, num_classes=num_classes).cuda())
    optimizer = optim.SGD(model_nat.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model_rob = None
    center, pc = None, None
    optimizer_center, optimizer_pc = None, None
    if args.training_type == 'lbgat' or args.training_type == 'bgat':
        model_rob = nn.DataParallel(get_network(model_type=model_type, num_classes=num_classes).cuda())
        optimizer = optim.SGD([{'params': model_nat.parameters()},
                               {'params': model_rob.parameters()}], lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif args.training_type == 'prototype':
        center = Proximity(num_classes=num_classes, feat_dim=512)
        pc = Con_Proximity(num_classes=num_classes, feat_dim=512)
        optimizer_center = optim.SGD(center.parameters(), lr=lr_center)
        optimizer_pc = optim.SGD(pc.parameters(), lr=lr_pc)
    
    adjust_learning_rate = lr_scheduler.MultiStepLR(optimizer, scheduler, gamma=0.1)
    
    inner_max = Inner_maximize_selector(training_type=args.training_type,
                                        alpha=alpha,
                                        epsilon=epsilon,
                                        num_repeats=num_repeats,
                                        num_classes=num_classes,
                                        lower=lower,
                                        upper=upper,
                                        tau=tau,
                                        lam=lam)
    outer_min = Outer_minimize_selector(training_type=args.training_type,
                                        lower=lower,
                                        upper=upper,
                                        num_repeats=num_repeats,
                                        num_classes=num_classes,
                                        lam1=lambda_av[0],
                                        lam2=lambda_av[1],
                                        gamma=gamma,
                                        beta=beta,
                                        pm_type=pm_type,
                                        warm_up=warm_up,
                                        lam=lam,
                                        tau=tau,
                                        xi=xi,
                                        center=center,
                                        pc=pc,
                                        wc=wc,
                                        wp=wp)

    best_acc_nat = 0
    best_acc_rob = 0
    for epoch in range(total_epochs):
        print('************* Training *************')
        trainer(epoch, model_nat, model_rob, optimizer, train_dataloader, inner_max, outer_min, optimizer_center, optimizer_pc)
        print('************ Validation ************')
        avg_acc_nat, avg_acc_rob = validation(model_nat, valid_dataloader, epsilon, alpha, num_repeats, lower, upper)
        
        is_best = best_acc_nat < avg_acc_nat and best_acc_rob < avg_acc_rob
        best_acc_nat = max(best_acc_nat, avg_acc_nat)
        best_acc_rob = max(best_acc_rob, avg_acc_rob)
        save_checkpoints = {'state_dict': model_nat.state_dict(),
                            'best_acc_nat': best_acc_nat,
                            'best_acc_rob': best_acc_rob,
                            'optimizer': optimizer.state_dict(),
                            'torch_seed': args.seed_pytorch,
                            'numpy_seed': args.seed_numpy}
        if args.training_type == 'lbgat':
            save_checkpoints['state_dict_rob'] = model_rob.state_dict()
        elif args.training_type == 'prototype':
            save_checkpoints['optimizer_center'] = optimizer_center.state_dict()
            save_checkpoints['optimizer_pc'] = optimizer_pc.state_dict()
            
        torch.save(save_checkpoints, os.path.join(path_chekpoint, 'model'))
        if is_best:
            print('Updating best model')
            torch.save(save_checkpoints, os.path.join(path_chekpoint, 'best_model'))
        adjust_learning_rate.step()
    
    