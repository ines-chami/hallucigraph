from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle as pkl
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim

sys.path.append('/Users/ineschami/PycharmProjects/gae-pytorch/')
import pygcn.models as models
from pygcn.utils.data_utils import load_data, preprocess_graph, mask_l_u_edges
from pygcn.utils.eval_utils import get_roc_score, sigmoid
from pygcn.utils.train_utils import loss_function_with_masking as loss_function

# from gae.utils import load_data, preprocess_graph, mask_test_edges
# from gae.optimizer import loss_function

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=20,
                    help='Patience for early stopping.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset to use.')
parser.add_argument('--model', type=str, default='VGAE',
                    help='Model to use.')
parser.add_argument('--save', action='store_true', default=False,
                    help='Whether to save the adjacency scores.')

args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset))
    adj, features, _, l_indices, _, _ = load_data(dataset_str=args.dataset)
    features = torch.FloatTensor(np.array(features.todense()))
    n_nodes, feat_dim = features.shape
    u_indices = [i for i in range(n_nodes) if i not in l_indices]
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, adj_mask, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_l_u_edges(
            adj, l_indices, u_indices)
    adj = adj_train
    adj_mask = torch.FloatTensor(adj_mask)

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    model_class = getattr(models, args.model)
    model = model_class(feat_dim, args.hidden, args.dropout)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    hidden_emb = None
    counter = 0
    best_val_ap = -1
    best_val_roc = -1
    roc_score = None
    ap_score = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label, adj_mask=adj_mask,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )
        if ap_curr > best_val_ap:  # or roc_curr > best_val_roc:
            if ap_curr > best_val_ap:  # and roc_curr > best_val_roc:
                roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            if args.save:
                np.save(os.path.join('reconstructions', '{}-{}-{}-rec'.format(args.dataset, args.model, args.hidden)),
                        sigmoid(hidden_emb.dot(hidden_emb.T)), )
            best_val_ap = ap_curr
            best_val_roc = roc_curr
            counter = 0
        else:
            counter += 1
            if counter == args.patience:
                print("Early stopping.")
                break

    print("Optimization Finished!")
    if not roc_score:
        roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    if args.save:
        np.save(os.path.join('reconstructions', '{}-{}-{}-orig'.format(args.dataset, args.model, args.hidden)),
                adj_orig.toarray())
        pkl.dump(test_edges,
                 open(os.path.join('reconstructions',
                                   '{}-{}-{}-test-edges'.format(args.dataset, args.model, args.hidden)), 'wb'),
                 )
        pkl.dump(test_edges_false,
                 open(os.path.join('reconstructions',
                                   '{}-{}-{}-test-edges-false'.format(args.dataset, args.model, args.hidden)), 'wb'),
                 )


if __name__ == '__main__':
    gae_for(args)
