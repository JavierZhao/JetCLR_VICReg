import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def translate_jets(batch, device, width=1.0):
    bb = batch.clone()
    X = bb.x.cpu().numpy()
    ptp_eta = np.ptp(X[:, 0], axis=-1, keepdims=True)
    ptp_phi = np.ptp(X[:, 1], axis=-1, keepdims=True)
    low_eta = -width * ptp_eta
    #     print(f"low eta: {low_eta}")
    high_eta = +width * ptp_eta
    #     print(f"high eta: {high_eta}")
    low_phi = np.maximum(
        -width * ptp_phi, -np.pi - np.min(X[:, 1]).reshape(ptp_phi.shape)
    )
    #     print(f"low phi: {low_phi}")
    high_phi = np.minimum(
        +width * ptp_phi, +np.pi - np.max(X[:, 1]).reshape(ptp_phi.shape)
    )
    #     print(f"high phi: {high_phi}")
    shift_eta_batch = np.random.uniform(
        low=low_eta, high=high_eta, size=(bb.y.shape[0], 1)
    )
    shift_phi_batch = np.random.uniform(
        low=low_phi, high=high_phi, size=(bb.y.shape[0], 1)
    )

    # To make sure that the components of each jet get shifted by the same amount
    for i in range(len(bb)):
        X_jet = bb[i].x.cpu().numpy()
        shift_eta_jet = np.ones((X_jet.shape[0], 1)) * shift_eta_batch[i]
        shift_phi_jet = np.ones((X_jet.shape[0], 1)) * shift_phi_batch[i]
        if i == 0:
            shift_eta = shift_eta_jet
            shift_phi = shift_phi_jet
        else:
            shift_eta = np.concatenate((shift_eta, shift_eta_jet))
            shift_phi = np.concatenate((shift_phi, shift_phi_jet))

    shift = np.hstack((shift_eta, shift_phi, np.zeros((X.shape[0], 5))))
    new_X = X + shift
    new_X = torch.tensor(new_X).to(device)
    bb.x = new_X
    return bb.to(device)


def rotate_jets(batch, device):
    bb = batch.clone()
    rot_angle = np.random.rand(len(bb)) * 2 * np.pi
    #     print(rot_angle)
    c = np.cos(rot_angle)
    s = np.sin(rot_angle)
    o = np.ones_like(rot_angle)
    z = np.zeros_like(rot_angle)
    rot_matrix = np.array([[z, c, -s], [z, s, c], [o, z, z]])  # (3, 3, 100)
    rot_matrix = rot_matrix.transpose(2, 0, 1)  # (100, 3, 3)

    for i in range(len(bb)):
        x_ = bb[i].x[:, :3]
        new_x = np.einsum(
            "ij,jk", bb[i].x.cpu()[:, :3], rot_matrix[i]
        )  # this is somehow (pT, eta', phi')
        new_x[:, [0, 2]] = new_x[:, [2, 0]]
        new_x[:, [0, 1]] = new_x[:, [1, 0]]  # now (phi', eta', pT)

        if i == 0:
            new_X = new_x
        else:
            new_X = np.concatenate((new_X, new_x), axis=0)

    new_X = torch.tensor(new_X).to(device)
    bb.x[:, :3] = new_X
    return bb.to(device)


def normalise_pts(batch):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-normalised jets, pT in each jet sums to 1, same shape as input
    """
    batch_norm = batch.copy()
    batch_norm[:, 0, :] = np.nan_to_num(
        batch_norm[:, 0, :] / np.sum(batch_norm[:, 0, :], axis=1)[:, np.newaxis],
        posinf=0.0,
        neginf=0.0,
    )
    return batch_norm


def rescale_pts(batch):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    """
    batch_rscl = batch.copy()
    batch_rscl[:, 0, :] = np.nan_to_num(
        batch_rscl[:, 0, :] / 600, posinf=0.0, neginf=0.0
    )
    return batch_rscl


def crop_jets(batch, nc):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of cropped jets, each jet is cropped to nc constituents, shape (batchsize, 3, nc)
    """
    batch_crop = batch.copy()
    return batch_crop[:, :, 0:nc]


def distort_jets(batch, strength=0.1, pT_clip_min=0.1):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with each constituents position shifted independently, shifts drawn from normal with mean 0, std strength/pT, same shape as input
    """
    pT = batch[:, 0]  # (batchsize, n_constit)
    shift_eta = np.nan_to_num(
        strength
        * np.random.randn(batch.shape[0], batch.shape[2])
        / pT.clip(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )  # * mask
    shift_phi = np.nan_to_num(
        strength
        * np.random.randn(batch.shape[0], batch.shape[2])
        / pT.clip(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )  # * mask
    shift = np.stack(
        [np.zeros((batch.shape[0], batch.shape[2])), shift_eta, shift_phi], 1
    )
    return batch + shift


def collinear_fill_jets(batch):
    """
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with collinear splittings, the function attempts to fill as many of the zero-padded args.nconstit
    entries with collinear splittings of the constituents by splitting each constituent at most once, same shape as input
    """
    batchb = batch.copy()
    nc = batch.shape[2]
    nzs = np.array(
        [np.where(batch[:, 0, :][i] > 0.0)[0].shape[0] for i in range(len(batch))]
    )
    for k in range(len(batch)):
        nzs1 = np.max([nzs[k], int(nc / 2)])
        zs1 = int(nc - nzs1)
        els = np.random.choice(np.linspace(0, nzs1 - 1, nzs1), size=zs1, replace=False)
        rs = np.random.uniform(size=zs1)
        for j in range(zs1):
            batchb[k, 0, int(els[j])] = rs[j] * batch[k, 0, int(els[j])]
            batchb[k, 0, int(nzs[k] + j)] = (1 - rs[j]) * batch[k, 0, int(els[j])]
            batchb[k, 1, int(nzs[k] + j)] = batch[k, 1, int(els[j])]
            batchb[k, 2, int(nzs[k] + j)] = batch[k, 2, int(els[j])]
    return batchb
