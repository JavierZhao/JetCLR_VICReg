import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch, Data


def translate_jets(batch, device, width=1.0):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta-phi translated jets, same shape as input
    """
    mask = (
        batch[:, 0] != 0
    ).float()  # 1 for constituents with non-zero pT, 0 otherwise

    # Calculating ptp (max - min) for eta and phi
    ptp_eta = (
        batch[:, 1, :].max(dim=-1, keepdim=True).values
        - batch[:, 1, :].min(dim=-1, keepdim=True).values
    )
    ptp_phi = (
        batch[:, 2, :].max(dim=-1, keepdim=True).values
        - batch[:, 2, :].min(dim=-1, keepdim=True).values
    )

    low_eta = -width * ptp_eta
    high_eta = +width * ptp_eta
    low_phi = torch.maximum(
        -width * ptp_phi,
        -torch.tensor(np.pi) - batch[:, 2, :].min(dim=1).values.reshape(ptp_phi.shape),
    )
    high_phi = torch.minimum(
        +width * ptp_phi,
        +torch.tensor(np.pi) - batch[:, 2, :].max(dim=1).values.reshape(ptp_phi.shape),
    )

    shift_eta = mask * (torch.rand_like(low_eta) * (high_eta - low_eta) + low_eta)
    shift_phi = mask * (torch.rand_like(low_phi) * (high_phi - low_phi) + low_phi)
    shift = torch.stack(
        [torch.zeros_like(shift_eta), shift_eta, shift_phi], dim=1
    ).squeeze().to(device)

    shifted_batch = batch + shift
    return shifted_batch


def rotate_jets(batch, device):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets rotated independently in eta-phi, same shape as input
    """
    rot_angle = torch.rand(batch.shape[0]) * 2 * torch.tensor(np.pi)
    c = torch.cos(rot_angle)
    s = torch.sin(rot_angle)
    o = torch.ones_like(rot_angle)
    z = torch.zeros_like(rot_angle)
    rot_matrix = (
        torch.stack([o, z, z, z, c, -s, z, s, c], dim=1)
        .reshape(-1, 3, 3)
        .transpose(0, 2)
    ).to(device)  # (batchsize, 3, 3)
    return torch.einsum("ijk,lji->ilk", batch, rot_matrix)


def normalise_pts(batch):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-normalised jets, pT in each jet sums to 1, same shape as input
    """
    batch_norm = batch.clone()
    pt_sum = batch_norm[:, 0, :].sum(dim=1, keepdim=True)
    normalized_pt = batch_norm[:, 0, :] / pt_sum
    normalized_pt = torch.nan_to_num(normalized_pt, posinf=0.0, neginf=0.0)
    batch_norm[:, 0, :] = normalized_pt
    return batch_norm


def rescale_pts(batch):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    """
    batch_rscl = batch.clone()
    rescaled_pt = batch_rscl[:, 0, :] / 600
    rescaled_pt = torch.nan_to_num(rescaled_pt, posinf=0.0, neginf=0.0)
    batch_rscl[:, 0, :] = rescaled_pt
    return batch_rscl


def crop_jets(batch, nc=50):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of cropped jets, each jet is cropped to nc constituents, shape (batchsize, 3, nc)
    """
    batch_crop = batch.clone()
    return batch_crop[:, :, 0:nc]


def distort_jets(batch, device, strength=0.1, pT_clip_min=0.1):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with each constituents position shifted independently,
            shifts drawn from normal with mean 0, std strength/pT, same shape as input
    """
    mask = (
        batch[:, 0] != 0
    ).float()  # 1 for constituents with non-zero pT, 0 otherwise
    pT = batch[:, 0]  # (batchsize, n_constit)
    clipped_pT = pT.clamp(min=pT_clip_min)

    shift_eta = mask * torch.randn_like(pT) * strength / clipped_pT
    shift_eta = torch.nan_to_num(shift_eta, posinf=0.0, neginf=0.0)

    shift_phi = mask * torch.randn_like(pT) * strength / clipped_pT
    shift_phi = torch.nan_to_num(shift_phi, posinf=0.0, neginf=0.0)

    zeros_tensor = torch.zeros_like(shift_eta)
    shift = torch.stack([zeros_tensor, shift_eta, shift_phi], dim=1).to(device)
    return batch + shift


def collinear_fill_jets(batch, device):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with collinear splittings, the function attempts to fill as many of the zero-padded args.nconstit
    entries with collinear splittings of the constituents by splitting each constituent at most once, same shape as input
    """
    batchb = batch.copy()
    nc = batch.shape[2]  # number of constituents
    nzs = np.array(
        [np.where(batch[:, 0, :][i] != 0.0)[0].shape[0] for i in range(len(batch))]
    )  # number of non-zero elements
    for k in range(len(batch)):
        zs1 = np.min([nzs[k], nc - nzs[k]])  # number of zero padded entries to fill
        els = np.random.choice(
            np.linspace(0, nzs[k] - 1, nzs[k]), size=zs1, replace=False
        )
        rs = np.random.uniform(size=zs1)  # scaling factor
        for j in range(zs1):
            # split pT
            batchb[k, 0, int(els[j])] = rs[j] * batch[k, 0, int(els[j])]
            batchb[k, 0, int(nzs[k] + j)] = (1 - rs[j]) * batch[k, 0, int(els[j])]
            # keep eta and phi
            batchb[k, 1, int(nzs[k] + j)] = batch[k, 1, int(els[j])]
            batchb[k, 2, int(nzs[k] + j)] = batch[k, 2, int(els[j])]
    return torch.tensor(batchb).to(device)
