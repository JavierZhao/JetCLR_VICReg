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
    """
    Input: batch of jets, shape (batchsize, 7, n_constit)
    dim 1 ordering: 'part_deta','part_dphi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR',
    Output: batch of eta-phi translated jets, same shape as input
    """
    batch = batch.to(device)
    mask = (batch[:, 0] != 0).float()  # 1 for constituents with non-zero pT, 0 otherwise

    # Calculating ptp (max - min) for eta and phi
    ptp_eta = batch[:, 0, :].max(dim=-1, keepdim=True).values - batch[:, 0, :].min(dim=-1, keepdim=True).values
    ptp_phi = batch[:, 1, :].max(dim=-1, keepdim=True).values - batch[:, 1, :].min(dim=-1, keepdim=True).values
    
    low_eta = -width * ptp_eta
    high_eta = +width * ptp_eta
    low_phi = torch.maximum(
        -width * ptp_phi,
        -torch.tensor(np.pi) - batch[:, 1, :].min(dim=1).values.reshape(ptp_phi.shape),
    )
    high_phi = torch.minimum(
        +width * ptp_phi,
        +torch.tensor(np.pi) - batch[:, 1, :].max(dim=1).values.reshape(ptp_phi.shape),
    )

    shift_eta = mask * (torch.rand_like(low_eta) * (high_eta - low_eta) + low_eta)
    shift_phi = mask * (torch.rand_like(low_phi) * (high_phi - low_phi) + low_phi)
    
    shift_eta = shift_eta.unsqueeze(1).to(device)
    shift_phi = shift_phi.unsqueeze(1).to(device)

    shift = (
        torch.cat([shift_eta, shift_phi, torch.zeros(batch[:, 2:,:].shape).to(device)], dim=1).to(device)
    )

    shifted_batch = batch + shift
    
    # recalculate \delta R
    shifted_eta = shifted_batch[:,0,:]
    shifted_phi = shifted_batch[:,1,:]
    delta_R = torch.sqrt(shifted_eta**2 + shifted_phi**2)
    # apply standardization
    delta_R = (delta_R - 0.2) * 4.0
    shifted_batch[:,-1,:] = delta_R
    return shifted_batch


def rotate_jets(batch, device):
    """
    Input: batch of jets, shape (batchsize, 7, n_constit)
    dim 1 ordering: 'part_deta','part_dphi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR',
    Output: batch of eta-phi translated jets, same shape as input
    """
    rot_angle = torch.rand(batch.shape[0]) * 2 * torch.tensor(np.pi)
    c = torch.cos(rot_angle).unsqueeze(-1)
    s = torch.sin(rot_angle).unsqueeze(-1)
    o = torch.ones_like(c)
    z = torch.zeros_like(c)
    
    # Construct the rotation matrix
    top_left = torch.stack([c, -s, s, c], dim=1).reshape(-1, 2, 2).to(device)
    
    identity_rest = torch.eye(7, 7, device=device).unsqueeze(0).expand(batch.shape[0], 7, 7).clone()
    identity_rest[:, :2, :2] = top_left
    
    return torch.einsum("ijk,ilj->ilk", batch, identity_rest).to(device)



def normalise_pts(batch, device):
    """
    Input: batch of jets, shape (batchsize, 7, n_constit)
    dim 1 ordering: 'part_deta','part_dphi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR',
    Output: batch of pT-normalised jets, pT in each jet sums to 1, same shape as input
    """
    
    # Extract log(pT) values and convert to pT
    log_pt_values = batch[:, 2, :]
    pt_values = torch.exp(log_pt_values).to(device)
    
    # Create a mask for non-zero padded values
    mask = (log_pt_values != 0).float()
    
    # Calculate the total pT for each jet, ignoring zero-padded entries
    total_pt_per_jet = torch.sum(pt_values * mask, dim=-1, keepdim=True)
    
    # If total_pt_per_jet is 0, replace it with 1 to avoid division by zero
    total_pt_per_jet = torch.where(total_pt_per_jet == 0, torch.tensor(1.).to(device), total_pt_per_jet)
    print(total_pt_per_jet)
    
    # Normalize pT values by dividing with the corresponding jet total
    normalized_pt_values = pt_values / total_pt_per_jet
    
    # Convert the normalized pT values back to logarithmic form
    normalized_log_pt_values = torch.where(mask != 0, torch.log(normalized_pt_values), torch.tensor(0.).to(device))
    
    # Replace the third entry (log(pT)) with the normalized log(pT) values and apply standardization
    batch[:, 2, :] = (normalized_log_pt_values - 1.7) * 0.7
    
    return batch.to(device)


def rescale_pts(batch):
    """
    Input: batch of jets, shape (batchsize, 7, n_constit)
    dim 1 ordering: 'part_deta','part_dphi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR',
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    """
    batch_rscl = batch.clone()
    rescaled_pt = batch_rscl[:, 2, :] / 600
    rescaled_pt = torch.nan_to_num(rescaled_pt, posinf=0.0, neginf=0.0)
    batch_rscl[:, 2, :] = rescaled_pt
    return batch_rscl


def crop_jets(batch, nc=50):
    """
    Input: batch of jets, shape (batchsize, 7, n_constit)
    dim 1 ordering: 'part_deta','part_dphi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR',
    Output: batch of cropped jets, each jet is cropped to nc constituents, shape (batchsize, 7, nc)
    """
    batch_crop = batch.clone()
    return batch_crop[:, :, 0:nc]


def distort_jets(batch, device, strength=0.1, pT_clip_min=0.1):
    '''
    Input: batch of jets, shape (batchsize, 7, n_constit)
    dim 1 ordering: 'part_deta','part_dphi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR',
    Output: batch of jets with each constituents position shifted independently, 
            shifts drawn from normal with mean 0, std strength/pT, same shape as input
    '''
    mask = (batch[:, 2] != 0).float() # 1 for constituents with non-zero pT, 0 otherwise
    # Extract log(pT) values and convert to pT
    log_pt_values = batch[:, 2, :]
    pT = torch.exp(log_pt_values).to(device)
    clipped_pT = pT.clamp(min=pT_clip_min)
    
    shift_eta = mask * torch.randn_like(pT) * strength / clipped_pT
    shift_eta = torch.nan_to_num(shift_eta, posinf=0.0, neginf=0.0)
    
    shift_phi = mask * torch.randn_like(pT) * strength / clipped_pT
    shift_phi = torch.nan_to_num(shift_phi, posinf=0.0, neginf=0.0)
    
    shift_eta = shift_eta.unsqueeze(1).to(device)
    shift_phi = shift_phi.unsqueeze(1).to(device)
    
    shift = (
        torch.cat([shift_eta, shift_phi, torch.zeros(batch[:, 2:,:].shape).to(device)], dim=1).to(device)
    )
    print(shift)

    shifted_batch = batch + shift
    
    # recalculate \delta R
    shifted_eta = shifted_batch[:,0,:]
    shifted_phi = shifted_batch[:,1,:]
    delta_R = torch.sqrt(shifted_eta**2 + shifted_phi**2)
    # apply standardization
    delta_R = (delta_R - 0.2) * 4.0
    shifted_batch[:,-1,:] = delta_R
    return shifted_batch


def collinear_fill_jets(batch, device):
    """
    Input: batch of jets, shape (batchsize, 7, n_constit)
    dim 1 ordering: 'part_deta','part_dphi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR',
    Output: batch of jets with collinear splittings, the function attempts to fill as many of the zero-padded args.nconstit
    entries with collinear splittings of the constituents by splitting each constituent at most once, same shape as input
    """
    batchb = batch.clone()
    nc = batch.size(2)  # number of constituents
    
    # Get non-zero elements for log(pT) which is index 2 in the new ordering
    nzs = (batch[:, 2, :] != 0.0).sum(dim=1)
    
    for k in range(batch.size(0)):
        zs1 = min(nzs[k], nc - nzs[k])  # number of zero padded entries to fill
        els = torch.randperm(nzs[k], device=device)[:zs1]
        rs = torch.rand(zs1, device=device)  # scaling factor
        
        for j in range(zs1):
            # keep the rest
            batchb[k, :, nzs[k] + j] = batch[k, :, els[j]]
            
            # Get pT values by exponentiating log(pT)
            pt_original = torch.exp(batch[k, 2, els[j]])
            
            # Split pT
            pt_split1 = rs[j] * pt_original
            pt_split2 = (1 - rs[j]) * pt_original
            
            # Assign back the log(pT) values
            batchb[k, 2, els[j]] = torch.log(pt_split1)
            batchb[k, 2, nzs[k] + j] = torch.log(pt_split2)
    # recalculate the rest of the kinematic features
    # part_e_log
    pt = torch.exp(batchb[:, 2, :])
    eta = batchb[:, 0, :]
    E = pt * torch.cosh(eta)
    batchb[:, 3, :] = (torch.log(E) - 2.0) * 0.7
    
    # part_logptrel
    pt_sum = torch.sum(pt, dim=-1, keepdim=True)
    pt_rel = pt / pt_sum
    batchb[:, 4, :] = (torch.log(pt_rel) - (-4.7)) * 0.7
    
    # part_logerel
    E_sum = torch.sum(E, dim=-1, keepdim=True)
    E_rel = E / E_sum
    batchb[:, 5, :] = (torch.log(E_rel) - (-4.7)) * 0.7
    return batchb

