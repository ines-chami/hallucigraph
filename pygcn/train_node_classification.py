from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim

import pygcn.models as models
from pygcn.utils.data_utils import load_data, accuracy, normalize, sparse_mx_to_torch_sparse_tensor

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100,
                    help='Patience for early stopping.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset to use.')
parser.add_argument('--model', type=str, default='GCN',
                    help='Model to use.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset_str=args.dataset)
features = normalize(features)
features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(labels)

adj = normalize(adj + sp.eye(adj.shape[0]))
adj = sparse_mx_to_torch_sparse_tensor(adj)

# Model and optimizer
model_class = getattr(models, args.model)
model = model_class(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test, acc_test


# Train model
t_total = time.time()
counter = 0
best_val_acc = -1
best_val_loss = np.inf
best_loss_test = None
best_acc_test = None
for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    if acc_val > best_val_acc or loss_val < best_val_loss:
        if acc_val > best_val_acc and loss_val < best_val_loss:
            best_loss_test, best_acc_test = test()
        counter = 0
        best_val_acc = acc_val
        best_val_loss = loss_val
    else:
        counter += 1
        if counter == args.patience:
            print("Early stopping")
            break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
if not best_loss_test:
    # Testing
    best_loss_test, best_acc_test = test()
print("Test set results:",
      "loss= {:.4f}".format(best_loss_test.item()),
      "accuracy= {:.4f}".format(best_acc_test.item()))
