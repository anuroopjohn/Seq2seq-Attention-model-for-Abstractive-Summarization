import torch

def train(model, iterator, optimizer, criterion, clip,device):
    '''
    function to train the model for a single epoch
    args : model - initialized model
           iterator - training dataset in tensor format
           optimizer - initialized optimizer
            criterion- loss function
            device - gpu/cpu

    returns : total training epoch loss for a single epoch
    '''
    
    model.train()
    epoch_loss = 0
    print("Training batch no : " , end =" ")
    for i, batch in enumerate(iterator):
        #print(i, end=" ")
        if int(i)%10000==0: print(i, end=" ")
        src = batch[0].to(device)
        src_len = batch[2].to(device)
        trg = batch[1].contiguous().to(device)
        optimizer.zero_grad()
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion,device):
    '''
    function to evaluate the model for a single epoch
    args : model - initialized model
           iterator - training dataset in tensor format
           criterion- loss function
           device - gpu/cpu

    returns : total validation epoch loss for a single epoch
    '''
    model.eval()
    epoch_loss = 0
    print("Validation batch no : ",end=" ")
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            #print(i ,end=" ")
            if int(i)%10000==0: print(i, end=" ")
            src = batch[0].to(device)
            src_len = batch[2].to(device)
            trg = batch[1].contiguous().to(device)
            output = model(src, src_len, trg, 0) #turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)