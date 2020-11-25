import os
import time
#import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math

loss_fn = F.cross_entropy

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, dataset, batch_size, epoch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_epoch_loss = 0
    total_epoch_acc = 0

    # train on gpu
    model.cuda()

    # Adam optimizer
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    steps = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        last_idx = math.floor(len(dataset) / batch_size)
        if(idx == last_idx - 10):
            print("End of data. Resetting dataloader.")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        text = batch[0]
        # if(text.shape[0] != batch_size):


        target = batch[2]
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()

        # convert to float tensor
        text = torch.tensor(text, device="cuda").float()

        # optimizer
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if steps % 100 == 0:
            print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    # Save model
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'parameters': (epoch)
    }, f'model/{epoch}/trained.pth')
    print(f'Model successfully saved.')

    return total_epoch_loss / len(dataloader), total_epoch_acc / len(dataloader)