import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data


def translate_jets(batch, device, width=1.0):
    """
    Input: batch of jets. x shape: [n_particles_total, 7]
    dim 1 ordering: [eta, phi, log pT, log E, log pT/pT_jet, log E/E_jet, ∆R]
    Output: batch of eta-phi translated jets, same shape as input
    """
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
    """
    Input: batch of jets. x shape: [n_particles_total, 7]
    dim 1 ordering: [eta, phi, log pT, log E, log pT/pT_jet, log E/E_jet, ∆R]
    Output: batch of jets rotated independently in eta-phi, same shape as input
    """
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


def normalize_pts(batch, device):
    """
    Input: batch of jets. x shape: [n_particles_total, 7]
    dim 1 ordering: [eta, phi, log pT, log E, log pT/pT_jet, log E/E_jet, ∆R]
    Output: batch of jets with pT normalized to sum to 1, same shape as input
    """
    bb = batch.clone()
    for i in range(len(bb)):
        x_norm = bb[i].x.clone()
        new_pt = torch.nan_to_num(
            x_norm[:, 2] / torch.sum(x_norm[:, 2]), posinf=0.0, neginf=0.0
        )

        if i == 0:
            new_PT = new_pt
        else:
            new_PT = torch.concatenate((new_PT, new_pt), axis=0)
    new_PT = torch.tensor(new_PT).to(device)
    bb.x[:, 2] = new_PT
    return bb.to(device)


def rescale_pts(batch, device):
    """
    Input: batch of jets. x shape: [n_particles_total, 7]
    dim 1 ordering: [eta, phi, log pT, log E, log pT/pT_jet, log E/E_jet, ∆R]
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    """
    bb = batch.clone()
    x_rscl = bb.x.clone()
    bb.x[:, 2] = torch.nan_to_num(x_rscl[:, 2] / 600, posinf=0.0, neginf=0.0)
    return bb.to(device)


def crop_jets(batch, nc, device):
    """
    Input: batch of jets. x shape: [n_particles_total, 7]
    dim 1 ordering: [eta, phi, log pT, log E, log pT/pT_jet, log E/E_jet, ∆R]
    Output: batch of cropped jets, each jet is cropped to nc constituents, same shape as input
    """
    new_data_list = []
    for i in range(len(batch)):
        new_x = batch[i].x[0:nc, :]
        jet = Data(x=new_x.to(device), y=batch[i].y.to(device))
        new_data_list.append(jet)
    new_batch = Batch.from_data_list(new_data_list)
    return new_batch.to(device)


def distort_jets(batch, device, strength=0.1, pT_clip_min=0.1):
    """
    Input: batch of jets. x shape: [n_particles_total, 7]
    dim 1 ordering: [eta, phi, log pT, log E, log pT/pT_jet, log E/E_jet, ∆R]
    Output: batch of jets with each constituents position shifted independently,
            shifts drawn from normal with mean 0, std strength/pT, same shape as input
    """
    bb = batch.clone()
    pT = bb.x[:, 2]
    shift_eta = np.nan_to_num(
        strength * np.random.randn(bb.x.shape[0]) / pT.clip(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )
    shift_phi = np.nan_to_num(
        strength * np.random.randn(bb.x.shape[0]) / pT.clip(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )
    shift = np.hstack(
        (
            shift_eta.reshape(-1, 1),
            shift_phi.reshape(-1, 1),
            np.zeros((bb.x.shape[0], 5)),
        )
    )
    new_X = bb.x + shift
    new_X = torch.tensor(new_X).to(device)
    bb.x = new_X
    return bb.to(device)


def collinear_fill_jets(batch, device, split_ratio=0.5):
    """
    Input: batch of jets. x shape: [n_particles_total, 7]
    dim 1 ordering: [eta, phi, log pT, log E, log pT/pT_jet, log E/E_jet, ∆R]
    Output: batch of jets with collinear splittings, splitting a fraction of the jets, same shape as input
    """
    new_data_list = []
    for i in range(len(batch)):
        # construct a new jet
        # for each particle, randomly decide whether to split or not, with split probability equal to split_ratio
        new_x = []
        for j in range(batch[i].x.shape[0]):
            if np.random.uniform() <= split_ratio:
                # initialize the two particles produced by collinear splitting
                ptcl_a = batch[i].x[j].clone()
                ptcl_b = batch[i].x[j].clone()
                # split the pT of the particle
                pT = batch[i].x[j, 2].item()
                pT_a = np.random.uniform(low=0, high=pT)
                pT_b = pT - pT_a
                # assign the split pT to the new particles
                ptcl_a[2] = pT_a
                ptcl_b[2] = pT_b
                # add the two new particles to the list
                new_x.append(ptcl_a)
                new_x.append(ptcl_b)
            else:
                # Do nothing, add the original particle to the list
                new_x.append(batch[i].x[j])
        #         print(new_x)
        new_jet = Data(x=torch.stack(new_x).to(device), y=batch[i].y.to(device))
        new_data_list.append(new_jet)
    new_batch = Batch.from_data_list(new_data_list)
    return new_batch.to(device)
