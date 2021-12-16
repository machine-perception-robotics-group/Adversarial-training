import torch

def trainer(epoch, model_nat, model_rob, optimizer, dataloader, inner, outer, optimizer_center, optimizer_pc):
    model_nat.train()
    total = 0
    total_loss = 0
    total_correct_nat = 0
    if model_rob is not None:
        model_rob.train()
        total_correct_rob = 0
        
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch = inputs.size(0)
        kappa = None
        if inner:
            ## Inner maximization
            if model_rob is not None:
                items = inner(model_rob, inputs, targets)
            else:
                items = inner(model_nat, inputs, targets)
        
            if isinstance(items, tuple):
                x = items[0]
                kappa = items[1]
            else:
                x = items
            inputs = torch.cat((inputs, x), dim=0)
        
        total += batch
        ## Outer minimization
        if model_rob is not None:
            loss, num_correct_nat, num_correct_rob = outer(model_nat, inputs, targets, optimizer, model_rob=model_rob)
            total_loss += loss
            total_correct_nat += num_correct_nat
            total_correct_rob += num_correct_rob
            if idx % 10 == 0:
                print('%d epochs [%d/%d] | loss: %.4f (avg: %.4f) | acc nat: %.4f (avg: %.4f) | acc rob: %.4f (avg: %.4f) |'\
                      % (epoch, idx, len(dataloader), loss, total_loss/len(dataloader),
                         num_correct_nat/batch, total_correct_nat/total,
                         num_correct_rob/batch, total_correct_rob/total))
        elif optimizer_center is not None and optimizer_pc is not None:
            loss, each_loss, num_correct_nat = outer(model_nat, inputs, targets, optimizer, 
                                          optimizer_center=optimizer_center, optimizer_pc=optimizer_pc)
            total_loss += loss
            total_correct_nat += num_correct_nat
            if idx % 10 == 0:
                print('%d epochs [%d/%d] | loss: %.4f (avg: %.4f) | acc nat: %.4f (avg: %.4f) |'\
                      % (epoch, idx, len(dataloader), loss, total_loss/len(dataloader),
                         num_correct_nat/batch, total_correct_nat/total))
                print(each_loss)
        else:
            loss, num_correct_nat = outer(model_nat, inputs, targets, optimizer, epoch=epoch, kappa=kappa)
            total_loss += loss
            total_correct_nat += num_correct_nat
            if idx % 10 == 0:
                print('%d epochs [%d/%d] | loss: %.4f (avg: %.4f) | acc nat: %.4f (avg: %.4f) |'\
                      % (epoch, idx, len(dataloader), loss, total_loss/len(dataloader),
                         num_correct_nat/batch, total_correct_nat/total))