import torch
import torch.nn as nn

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