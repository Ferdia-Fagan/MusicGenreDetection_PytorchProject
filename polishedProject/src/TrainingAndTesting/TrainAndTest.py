import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import accuracy_score

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# I took the sample project given with this project and adjusted it small bit to this.
# tried afew loss functions, but crossEntropyLoss was best considering my limitation
# to working on local machine.

def train(model, device, train_loader, optimizer):
    model.train()
    model.to(device)
    cost = nn.CrossEntropyLoss()    # is the same as LogSoftmax and NLLoss
    # cost = nn.NLLLoss()

    with tqdm(total=len(train_loader), disable=False) as progress_bar:
        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            # print("data.shape: ", data.shape)
            # print("data.shape: ", label.shape)
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            # torch.max(label, 1)[1].cuda()
            loss = cost(output, label.flatten())
            loss.backward()
            optimizer.step()
            progress_bar.update(1)


def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        # Need this line for things like dropout etc.
    model.eval()
    preds = []
    targets = []
    cost = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(device)
            target = label.flatten().to(device)
            output = model(data)

            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

            losses.append(cost(output, target).cpu())

    loss = np.mean(losses)
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets = np.concatenate(targets)
    # acc = f1_score(targets, preds, average='micro')
    acc = accuracy_score(targets, preds)    # as specified in question
    return loss, acc


def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if device is not None:
                data = data.to(device)

            target = label.clone()
            output = model(data)

            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    preds = np.argmax(np.concatenate(preds), axis=1)
    targets = np.concatenate(targets)
    # acc = f1_score(targets, preds, average='micro')
    acc = accuracy_score(targets, preds)    # as specified in question

    # confusionMatrix = confusion_matrix()

    return acc





