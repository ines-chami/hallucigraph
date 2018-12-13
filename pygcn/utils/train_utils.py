import torch
import torch.nn.functional as F
import torch.nn.modules.loss


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def loss_function_with_masking(preds, labels, adj_mask, mu, logvar, n_nodes, norm, pos_weight):
    """Link prediction loss with weighted sigmoid cross entropy."""
    masked_preds = preds - (10e8 * adj_mask)
    masked_labels = labels
    cost = norm * F.binary_cross_entropy_with_logits(masked_preds, masked_labels, pos_weight=pos_weight)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def rec_loss(preds, labels, pos_weight):
    return F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
