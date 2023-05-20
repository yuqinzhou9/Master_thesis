import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random

import torchvision
import torchvision.transforms as transforms

from src.utils.optim.schedulers import CosineWarmup
from src.models.sequence.rnns.rnn import RNN


import os
import argparse
from tqdm.auto import tqdm


import wandb
wandb.login()


# Use cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device available now:', device)

if device == 'cuda':
    cudnn.benchmark = True



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Dataset
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'listops', 'imdb', 'aan', 'pathfinder'], type=str, help='Dataset')
###! imdb refers to TEXT, ann refers to RETRIEVAL 
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')


# Dataloader
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=50, type=int, help='Batch size')


# Optimizer
parser.add_argument('--lr', default= 3e-4, type=float, help='Learning rate')
parser.add_argument('--lr_factor', default= 0.25, type=float, help='Factor of Learning rate') 
parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')



# Scheduler
parser.add_argument('--epochs', default=200, type=float, help='Training epochs')


# Model
parser.add_argument('--n_layers', default= 6, type=int, help='Number of layers') #6
parser.add_argument('--d_model', default= 128, type=int, help='Model dimension') #512
parser.add_argument('--d_hidden', default= 256, type=int, help='Hidden (state) dimension ') #384
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_false', help='Prenorm')
parser.add_argument('--norm', default= 'BN', choices=['LN', 'BN'], help='Norm types')
parser.add_argument('--cell', default= 'rnn', type=str, help='RNN\'s cell')


# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


if args.dataset == 'cifar10':
    if args.grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            transforms.Lambda(lambda x: x.view(1, 1024).t())
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t())
        ])
    
    # S4 is trained on sequences with no data augmentation!
    transform_train = transform_test = transform

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=False, download=True, transform=transform_test)

    d_input = 3 if not args.grayscale else 1
    d_output = 10

else: raise NotImplementedError

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

class RNNbased(nn.Module):

    def __init__(
        self,
        d_input,
        d_output,
        lr,
        cell='rnn',
        d_model=256,
        d_hidden=128,
        n_layers=2,
        dropout=0.2,
        prenorm=True,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB) (like embedding layer)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.FFNs = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                RNN(d_input = d_model, d_model = d_hidden, lr = lr, cell = cell, return_output=True, transposed=True, dropout=0)
            )
            # self.norms.append(nn.LayerNorm(d_model))
            self.norms.append(nn.BatchNorm1d(d_model)) 
            self.dropouts.append(nn.Dropout1d(dropout))
            self.FFNs.append(nn.Sequential(nn.Conv1d(d_hidden, d_model*2, kernel_size= 1 ), nn.GLU(dim=-2))  #                
                                 )

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        

        for layer, norm, dropout, FFN in zip(self.layers, self.norms, self.dropouts, self.FFNs):
            ''' Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L) '''

            
            z = x #(B, d_model, L) -> (B, d_model, L) 
            if self.prenorm:
                # Prenorm (BN)
                z = norm(z) # (B, d_model, L) -> (B, d_model, L) 

            # Apply recurrence: we ignore the state input and output
            z, _ = layer(z) #(B, d_model, L) -> (B, d_hidden, L) (note that we transpose the input inside the layer)

            
            # Dropout on the output of the MLP 
            z = dropout(z) #(B, d_hidden, L) -> (B, d_hidden, L) for dropout1d
            
            # MLP +GLP
            z = FFN(z) #(B, d_hidden, L) -> (B, d_model, L) for conv1d

            # Residual connection
            x = z + x  #(B, d_model, L) -> (B, d_model, L)

            if not self.prenorm:
                # Post-norm (BN)
                x = norm(x) #(B, d_model, L) -> (B, d_model, L)
                
        # Pooling: average pooling over the sequence length
        x = x.transpose(-1, -2)
        x = x.mean(dim=1) # (B, L, d_model) -> (B, d_model)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
    

def setup_optimizer(model, lr, weight_decay, epochs):
    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        # e.g., p could be {'weight_decay': 0.0, 'lr': 1e-07}
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts 
    
    
    for hp in hps: 
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp] ## select parameter matrices that have "_optim" and assign "_optim = None" to matrices that do not have
        optimizer.add_param_group(
            {"params": params, **hp} ## <**hp> referes to hyperparameters e.g., {'weight_decay': 0.0}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scheduler = CosineWarmup(optimizer, T_max = epochs, eta_min= 1e-7, warmup_step= int(epochs * 0.1) + 1) 

    ''' Print optimizer info '''
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    
    
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler



# Training
def train(model, optimizer,  criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    print_every = 50

    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        wandb.log({"Train/Batch loss": train_loss/(batch_idx+1)})

        # if (batch_idx % print_every) == 0:
        #     print(f"Batch: {batch_idx}, Avg: {train_loss/(batch_idx+1)}")
        
        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )
    return train_loss/(batch_idx+1)



def eval(model, criterion, dataloader):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )
            
        return 100.*correct/total
    

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
run_id = wandb.util.generate_id()
# run_id = 'kw4enoiw'
CHECKPOINT_PATH = f'./checkpoint/checkpoint_{run_id}.pth'
print(CHECKPOINT_PATH)



total_runs = 1
for run in range(total_runs):
    wandb.init(
        id= run_id,
        project="test", 
        # Model + Cell + Run
        name=f"test_args", 
        config=args,
        resume = 'allow')
    
    # defining model, optimizer, scheduler (whether resume or not)
    model = RNNbased(d_input=d_input, 
                    d_output=d_output, 
                    lr = args.lr * args.lr_factor,
                    cell='rnn',
                    d_model=args.d_model, 
                    d_hidden=args.d_hidden, 
                    n_layers=args.n_layers, 
                    dropout=args.dropout, 
                    prenorm=args.prenorm)
    
    optimizer, scheduler = setup_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs)
    
    if not wandb.run.resumed:
        print('==> Building model / ...')
    else:
        print('==> Resuming from checkpoint...')
        checkpoint = torch.load(CHECKPOINT_PATH) #not use wandb.restore('checkpoint.tar') because of encoding error
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        
    model = model.to(device)

    ## defining training, validating and testing
    pbar = tqdm(range(start_epoch, args.epochs))
    for epoch in pbar:
        wandb.log({"Epoch": epoch, "lr_general": scheduler.get_last_lr()[0]}) #record general lr for the current lr
        wandb.log({"Epoch": epoch, "lr_special": scheduler.get_last_lr()[1]}) #record special lr for the current lr
        
        if epoch == 0:
            pbar.set_description('Epoch: %d' % (epoch))
        else:
            try: 
                val_acc # colab sometimes loss the defiitio of val_acc
            except:
                  pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, best_acc))
            else:
                  pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))

        print('==> Training...')
        epoch_loss = train(model = model,optimizer = optimizer, criterion = nn.CrossEntropyLoss())
        wandb.log({"Epoch": epoch, "Train/Epoch Loss": epoch_loss})


        print('==> Validating...')
        val_acc = eval(model = model, criterion = nn.CrossEntropyLoss(), dataloader = valloader)
        wandb.log({"Epoch": epoch, "Val/Val acc": val_acc})

        #Save checkpoints
        if val_acc > best_acc:
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # record lr for the best previous epoch
                'acc': val_acc,
                'epoch': epoch,

            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
                
            torch.save(state, CHECKPOINT_PATH)
            # wandb.save(CHECKPOINT_PATH)
            best_acc = val_acc
        
        scheduler.step() #update lr

    print('==> Testing...')
    test_acc = eval(model = model, criterion = nn.CrossEntropyLoss(), dataloader = testloader)
    wandb.log({"Test/Test acc": test_acc})
    
    # Mark the run as finished
    wandb.finish()
