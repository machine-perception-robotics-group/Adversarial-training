import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Outer_minimize_selector:
    def __init__(self, training_type, lower, upper, num_classes, num_repeats, 
                 lam1=None, lam2=None, gamma=None, beta=None, pm_type=None, 
                 warm_up=None, lam=None, tau=None, xi=None, center=None, pc=None, wc=None, wp=None):
        
        self.lower = lower
        self.upper = upper
        self.num_repeats = num_repeats
        self.num_classes = num_classes
        self.training_type = training_type
        
        ## Loss functions
        self.mse = nn.MSELoss()
        self.xent = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='none')
        self.center = center
        self.pc = pc
        
        ## Activation functions
        self.tanh = nn.Tanh()
        self.hinge = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        ## Hyper parameters
        self.lam1 = lam1
        self.lam2 = lam2
        self.gamma = gamma
        self.beta = beta
        self.pm_type = pm_type
        self.warm_up = warm_up
        self.tau = tau
        self.lam = lam
        self.xi = xi
        self.wc = wc
        self.wp = wp
        
            
    def __call__(self, model, inputs, targets, optimizer, **kwargs):
        if self.training_type == 'standard':
            return self.standard_at(model, inputs, targets, optimizer)
        elif self.training_type == 'kernel_trick':
            return self.kernel_trick(model, inputs, targets, optimizer)
        elif self.training_type == 'alp':
            return self.alp(model, inputs, targets, optimizer)
        elif self.training_type == 'avmixup':
            return self.avmixup(model, inputs, targets, optimizer)
        elif self.training_type == 'clp':
            return self.clp(model, inputs, targets, optimizer)
        elif self.training_type == 'mail':
            return self.mail(model, inputs, targets, optimizer, epoch=kwargs.get('epoch'))
        elif self.training_type == 'gairat':
            return self.gairat(model, inputs, targets, optimizer, epoch=kwargs.get('epoch'), kappa=kwargs.get('kappa'))
        elif self.training_type == 'wmmr':
            return self.wmmr(model, inputs, targets, optimizer)
        elif self.training_type == 'trades':
            return self.trades(model, inputs, targets, optimizer)
        elif self.training_type == 'gat':
            return self.gat(model, inputs, targets, optimizer)
        elif self.training_type == 'mart':
            return self.mart(model, inputs, targets, optimizer)
        elif self.training_type == 'prob_compact':
            return self.prob_compact(model, inputs, targets, optimizer, epoch=kwargs.get('epoch'))
        elif self.training_type == 'prototype':
            return self.prototype(model, inputs, targets, optimizer, 
                                  optimizer_center=kwargs.get('optimizer_center'),
                                  optimizer_pc=kwargs.get('optimizer_pc'))
        elif self.training_type == 'lbgat':
            return self.lbgat(model, inputs, targets, optimizer, model_rob=kwargs.get('model_rob'))
        else:
            assert 0, 'Error: %d is not supported.'

    def standard_at(self, model, inputs, targets, optimizer):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits = model(x_adv)
        loss = self.xent(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def kernel_trick(self, model, inputs, targets, optimizer):
        logits = model(inputs)
        loss = self.xent(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def alp(self, model, inputs, targets, optimizer):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits_nat = model(x_nat)
        logits_adv = model(x_adv)
        loss_xent = self.xent(logits_adv, targets)
        loss_lp = torch.sqrt(torch.sum(torch.pow(logits_nat - logits_adv, 2)))/x_nat.size(0)
        loss = loss_xent + self.lam*loss_lp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits_adv.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def clp(self, model, inputs, targets, optimizer):
        logits = model(inputs)
        loss_xent = self.xent(logits, targets)
        logits1, logits2 = torch.chunk(logits, 2, dim=0)
        loss_lp = torch.sqrt(torch.sum((logits1 - logits2)**2))/inputs.size(0)
        loss = loss_xent + self.lam*loss_lp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def avmixup(self, model, inputs, targets, optimizer):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        xw = torch.from_numpy(np.random.beta(1,1,[x_nat.size(0),1,1,1])).float().cuda()
        delta = (x_adv - x_nat) * self.gamma
        x_av = torch.clamp(x_nat+delta, min=0, max=1)
        x = xw * x_nat + (1 - xw) * x_av
        
        yw = xw.view(x_nat.size(0),-1)
        onehot = torch.eye(self.num_classes)[targets].cuda()
        y_nat = onehot * self.lam1 + (onehot - 1) * ((self.lam1 - 1)/(self.num_classes - 1))
        y_adv = onehot * self.lam2 + (onehot - 1) * ((self.lam2 - 1)/(self.num_classes - 1))
        y = yw * y_nat + (1 - yw) * y_adv
        
        logits = model(x)
        loss = torch.sum(-y * F.log_softmax(logits, dim=1))/x.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def mail(self, model, inputs, targets, optimizer, **kwargs):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits = model(x_adv)
        if kwargs.get('epoch') > self.warm_up:
            class_index = torch.arange(self.num_classes)[None,:].repeat(x_adv.size(0),1).cuda()
            if self.pm_type == 'nat':
                probs = torch.softmax(model(x_nat), dim=1)
            elif self.pm_type == 'adv':
                probs = torch.softmax(logits, dim=1)
                
            wrong_probs = probs[class_index!=targets[:,None]].view(x_adv.size(0), self.num_classes-1)
            correct_probs = probs[class_index==targets[:,None]]
            top2_probs = torch.topk(wrong_probs, k=1)[0]
            pm = correct_probs - top2_probs
            s = self.sigmoid(-self.gamma*(pm - self.beta))
            s = s/torch.sum(s)
            loss = torch.sum(-s*F.log_softmax(logits, dim=1)[class_index==targets[:,None]])/x_nat.size(0)
        else:
            loss = self.xent(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def gairat(self, model, inputs, targets, optimizer, **kwargs):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits = model(x_adv)
        kappa = kwargs.get('kappa')
        if kwargs.get('epoch') > self.warm_up:
            s = (1 + self.tanh(self.lam + 5*(1 - 2*kappa/self.num_repeats)))/2
            s = s/torch.sum(s)
            class_index = torch.arange(self.num_classes)[None,:].repeat(x.size(0),1).cuda()
            loss = torch.sum(-s * torch.log_softmax(logits, dim=1)[class_index==targets[:,None]])/x_adv.size(0)
        else:
            loss = self.xent(logits, targets)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def wmmr(self, model, inputs, targets, optimizer):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits = model(x_adv)
        
        class_index = torch.arange(self.num_classes)[None,:].repeat(x_adv.size(0),1).cuda()
        wrong_probs = logits.softmax(dim=1)[class_index!=targets[:,None]].view(x_adv.size(0), self.num_classes-1)
        correct_probs = logits.softmax(dim=1)[class_index==targets[:,None]]
        top2_probs = torch.topk(wrong_probs, k=1)[0]
        s = torch.exp(-self.tau*(correct_probs - top2_probs))
        loss = torch.sum(-s*torch.log_softmax(logits, dim=1)[class_index==targets[:,None]])/x_adv.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def gat(self, model, inputs, targets, optimizer):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits_nat = model(x_nat)
        logits_adv = model(x_adv)
        
        xent_loss = self.xent(logits_nat, targets)
        reg = torch.sum((logits_adv.softmax(dim=1) - logits_nat.softmax(dim=1))**2)/x_nat.size(0)
        loss = xent_loss + self.lam*reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits_adv.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
        
    def trades(self, model, inputs, targets, optimizer):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits_nat = model(x_nat)
        logits_adv = model(x_adv)
        
        xent_loss = self.xent(logits_nat, targets)
        kl_loss = torch.sum(self.kl(F.log_softmax(logits_adv, dim=1),
                                    F.softmax(logits_nat, dim=1)))/x_adv.size(0)
        loss = xent_loss + self.beta*kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits_adv.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def mart(self, model, inputs, targets, optimizer):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits_nat = model(x_nat)
        logits_adv = model(x_adv)
        
        class_index = torch.arange(self.num_classes)[None,:].repeat(x_adv.size(0),1).cuda()
        log_softmax_gt = F.log_softmax(logits_adv, dim=1)[class_index==targets[:,None]]
        log_softmax_others = F.log_softmax(logits_adv, dim=1)[class_index!=targets[:,None]].view(x_adv.size(0), self.num_classes-1)
        bce_loss = torch.sum(-log_softmax_gt - (1 - log_softmax_others.argmax(dim=1)))/x_adv.size(0)
        kl_loss = torch.sum(self.kl(F.log_softmax(logits_adv, dim=1),
                                    F.softmax(logits_nat, dim=1)), dim=1)
        loss = bce_loss + self.beta*torch.sum(kl_loss*(1 - logits_nat.softmax(dim=1)[class_index==targets[:,None]]))/x_adv.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits_adv.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def prob_compact(self, model, inputs, targets, optimizer, **kwargs):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits = model(x_adv)
        
        if kwargs.get('epoch') > self.warm_up:
            probs = torch.softmax(logits, dim=1)
            
            ## Probabilisically compact loss
            class_index = torch.arange(self.num_classes)[None,:].repeat(x_adv.size(0),1).cuda()
            wrong_probs = probs[class_index!=targets[:,None]].view(inputs.size(0),self.num_classes-1)
            correct_probs = probs[class_index==targets[:,None]][:,None].repeat(1, self.num_classes-1)
            pc = torch.sum(hinge(wrong_probs + self.xi - correct_probs)) / x_adv.size(0)
            
            ## Logits constraints
            wrong_logits = logits[class_index!=targets[:,None]].view(x_adv.size(0),self.num_classes-1)
            correct_logits = logits[class_index==targets[:,None]]
            top2_logits = torch.topk(wrong_logits, k=1)[0]
            const = torch.sum(hinge(correct_logits - top2_logits)) / x_adv.size(0)
            loss = pc + self.lam*const
        else:
            loss = self.xent(logits, targets)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct
    
    def prototype(self, model, inputs, targets, optimizer, **kwargs):
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits, features = model(x_adv)
        
        center_loss = self.center(features, targets)
        pc_loss = self.pc(features, targets)
        xent_loss = self.xent(logits, targets)
        loss = xent_loss + self.wc*center_loss - self.wp*pc_loss
        
        optimizer_center = kwargs.get('optimizer_center')
        optimizer_pc = kwargs.get('optimizer_pc')
        
        ## Updating model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Updating center
        optimizer_center.zero_grad()
        for param in self.center.parameters():
            param.grad.data *= (1/self.wc)
        optimizer_center.step()
        
        ## Updating prototype conforminy
        optimizer_pc.zero_grad()
        for param in self.pc.parameters():
            param.grad.data *= (1/self.wp)
        optimizer_pc.step()
        
        num_correct = logits.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        each_loss = '    | xent: %.4f | center: %.4f | pc: %.4f |' % (xent_loss.item(), center_loss.item(), pc_loss.item())
        return loss.item(), each_loss, num_correct
    
    def lbgat(self, model_nat, inputs, targets, optimizer, **kwargs):
        model_rob = kwargs.get('model_rob')
        x_nat, x_adv = torch.chunk(inputs, 2, dim=0)
        logits_nat = model_nat(x_nat)
        logits_adv = model_rob(x_adv)
        
        loss = 0.5*self.mse(logits_adv, logits_nat) + self.lam*self.xent(logits_nat, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_correct_nat = logits_nat.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        num_correct_rob = logits_adv.softmax(dim=1).argmax(dim=1).eq(targets).sum().item()
        return loss.item(), num_correct_nat, num_correct_rob