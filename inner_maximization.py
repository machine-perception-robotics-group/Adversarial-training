import torch
import torch.nn as nn
import torch.nn.functional as F

class Inner_maximize_selector:
    def __init__(self, training_type, alpha, epsilon, num_repeats, num_classes, lower=0, upper=1, tau=None, lam=None):
        self.training_type = training_type
        self.standard = ['alp', 'avmixup', 'mail', 'mart', 'prob_compact', 'prototype', 'standard', 'lbgat']
        
        self.lower=lower
        self.upper=upper
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_repeats = num_repeats
        self.num_classes = num_classes
        
        ## Variables for the specific method
        self.xent = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='none')
        self.tau = tau
        self.lam = lam
            
    def __call__(self, model, inputs, targets):
        if self.training_type in self.standard:
             return self.standard_pgd(model, inputs, targets)
        elif self.training_type == 'gairat':
             return self.gairat_pgd(model, inputs, targets)
        elif self.training_type == 'wmmr':
             return self.wmmr_pgd(model, inputs, targets)
        elif self.training_type == 'trades':
             return self.trades_pgd(model, inputs, targets)
        elif self.training_type == 'gat':
             return self.gama(model, inputs, targets)
        elif self.training_type == 'kernel_trick' or self.training_type == 'clp':
            return None
        else:
            assert 0, 'Error: %d is not supported.'

    def standard_pgd(self, model, inputs, targets):
        noise = torch.FloatTensor(inputs.shape).uniform_(0,1).cuda()
        x = torch.clamp(inputs+noise, min=self.lower, max=self.upper)
        
        for _ in range(self.num_repeats):
            x.requires_grad_()
            logits = model(x)
            loss = self.xent(logits, targets)
            loss.backward()
            grads = x.grad.data
            x = x.detach() + self.alpha*torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs-self.epsilon), inputs+self.epsilon).clamp(min=self.lower,max=self.upper)
        return x
    
    def gairat_pgd(self, model, inputs, targets):
        noise = torch.FloatTensor(inputs.shape).uniform_(0,1).cuda()
        x = torch.clamp(inputs+noise, min=self.lower, max=self.upper)
        kappa = torch.zeros(x.size(0)).cuda()
        
        for _ in range(self.num_repeats):
            x.requires_grad_()
            logits = model(x)
            loss = self.xent(logits, targets)
            loss.backward()
            grads = x.grad.data
            x = x.detach() + self.alpha*torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs-self.epsilon), inputs+self.epsilon).clamp(min=self.lower,max=self.upper)
            
            kappa += logits.softmax(dim=1).argmax(dim=1).eq(targets)
        return (x, kappa)
    
    def wmmr_pgd(self, model, inputs, targets):
        noise = torch.FloatTensor(inputs.shape).normal_(0, 1).cuda()
        x = torch.clamp(inputs + 0.001*noise, min=self.lower, max=self.upper)
        class_index = torch.arange(self.num_classes)[None,:].repeat(x.size(0),1).cuda()
        
        for _ in range(self.num_repeats):
            x.requires_grad_()
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            wrong_probs = probs[class_index!=targets[:,None]].view(x.size(0),self.num_classes-1)
            correct_probs = probs[class_index==targets[:,None]].unsqueeze(1)
            top2_probs = torch.topk(wrong_probs, k=1).values
            margin = correct_probs - top2_probs
            s = torch.exp(-self.tau*margin)
            
            loss = torch.sum(-s * F.log_softmax(logits, dim=1)[class_index==targets[:,None]])/x.size(0)
            loss.backward()
            grads = x.grad.data
            x = x.detach() + self.alpha*torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs-self.epsilon), inputs+self.epsilon).clamp(min=self.lower,max=self.upper)
        return x
    
    def trades_pgd(self, model, inputs, targets):
        noise = torch.FloatTensor(inputs.shape).normal_(0, 1).cuda()
        x = torch.clamp(inputs + 0.001*noise, min=self.lower, max=self.upper)
        
        for _ in range(self.num_repeats):
            x.requires_grad_()
            logits = model(x)
            loss = torch.sum(self.kl(F.log_softmax(logits, dim=1),
                                     F.softmax(model(inputs), dim=1)))/x.size(0)
            loss.backward()
            grads = x.grad.data
            x = x.detach() + self.alpha*torch.sign(grads).detach()
            x = torch.min(torch.max(x, inputs-self.epsilon), inputs+self.epsilon).clamp(min=self.lower,max=self.upper)
        return x
    
    def gama(self, model, inputs, targets):
        bern = torch.bernoulli(torch.empty(inputs.shape).uniform_(0, 1))
        delta = torch.where(bern==0, -self.epsilon, self.epsilon).cuda().requires_grad_()
        x = torch.clamp(inputs+delta, min=self.lower, max=self.upper)
        
        epsilon = self.epsilon/self.num_repeats
        for i in range(self.num_repeats):
            logits_nat = model(inputs)
            logits_adv = model(x)
            loss = self.xent(logits_adv, targets) + self.lam*torch.sum((logits_adv.softmax(dim=1) - logits_nat.softmax(dim=1))**2)/x.size(0)
            loss.backward()
            grads = delta.grad.data
            delta = delta.data.detach() + self.epsilon*torch.sign(grads).detach()
            delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)
            x = torch.clamp(inputs+delta, min=0, max=1)
        return x
    
    