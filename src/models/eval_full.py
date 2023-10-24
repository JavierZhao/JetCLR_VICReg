""" 
The full evaluation pipeline for VICReg.
1. Plot training and validation losses for the trained VICReg model.
2. Perform the Linear Classifier Test (LCT) on the representations learned by VICReg.
3. Generate pair plots for the learned representations.
4. Generate t-SNE plots for the learned representations.
"""

# load standard python modules
import argparse
from datetime import datetime
import copy
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tqdm
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from pathlib import Path

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# load custom modules required for jetCLR training
from src.models.transformer import Transformer
from src.features.perf_eval import get_perf_stats, linear_classifier_test
from src.models.pretrain_vicreg import VICReg
from src.models.jet_augs import *

project_dir = Path(__file__).resolve().parents[2]
print(f"project_dir: {project_dir}") # /ssl-jet-vol-v2/JetCLR_VICReg

# load the data files and the label files from the specified directory
def load_data(args, flag):
    frac = 1
    dataset_path = args.dataset_path
    data_file = f"{dataset_path}/{flag}_{frac}%/data/data.pt"
    data = torch.load(data_file)
    print(f"--- loaded data file from `{flag}_{frac}%` directory")            
    return data

# labels are only used for the LCT
def load_labels(args, flag):
    frac = 1
    dataset_path = args.dataset_path
    data_file = f"{dataset_path}/{flag}_{frac}%/label/labels.pt"
    data = torch.load(data_file)        
    print(f"--- loaded label file from `{flag}_{frac}%` directory")    
    return data


def get_backbones(args):
    x_backbone = Transformer(input_dim=args.x_inputs, output_dim=args.feature_dim, model_dim=args.model_dim, dim_feedforward=args.model_dim)
    y_backbone = x_backbone if args.shared else copy.deepcopy(x_backbone)
    return x_backbone, y_backbone


def augmentation_lct(x, device):
    """
    Applies all the augmentations specified in the args
    """
    y = x.clone()
    x = x.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
    y = y.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
    return x.to(device), y.to(device)

def augmentation(args, x, device):
    """
    Applies all the augmentations specified in the args
    """
    # crop all jets to a fixed number of constituents (default=50)
    x = crop_jets(x, args.nconstit)
    x = rotate_jets(x, device)
    y = x.clone()
    if args.do_rotation:
        y = rotate_jets(y, device)
    if args.do_cf:
        y = collinear_fill_jets(np.array(y.cpu()), device)
        y = collinear_fill_jets(np.array(y.cpu()), device)
    if args.do_ptd:
        y = distort_jets(y, device, strength=args.ptst, pT_clip_min=args.ptcm)
    if args.do_translation:
        y = translate_jets(y, device, width=args.trsw)
        x = translate_jets(x, device, width=args.trsw)
    x = rescale_pts(x)  # [batch_size, 3, n_constit]
    y = rescale_pts(y)  # [batch_size, 3, n_constit]
    x = x.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
    y = y.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
    return x, y

def plot_losses(args):
    label = args.label
    model_label = "vicreg_" + label
    cov_loss_train_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_cov_loss_train_epochs.npy")
    cov_loss_val_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_cov_loss_val_epochs.npy")
    loss_train_batches = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_loss_train_batches.npy")
    loss_train_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_loss_train_epochs.npy")
    loss_val_batches = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_loss_val_batches.npy")
    loss_val_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_loss_val_epochs.npy")
    repr_loss_train_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_repr_loss_train_epochs.npy")
    repr_loss_val_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_repr_loss_val_epochs.npy")
    std_loss_train_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_std_loss_train_epochs.npy")
    std_loss_val_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_std_loss_val_epochs.npy")
    lct_auc_epochs = np.load(f"{project_dir}/models/model_performances/JetClass/{label}/{model_label}_lct_auc_epochs.npy")

    # Plot loss curves in training
    fontsize = 20
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].plot(loss_train_epochs, 'r') #row=0, col=0
    ax[0, 0].set_xlabel("Epochs", fontsize=fontsize)
    ax[0, 0].set_ylabel("Total loss", fontsize=fontsize)
    ax[0, 0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1,0].plot(repr_loss_train_epochs, 'b') #row=1, col=0
    ax[1,0].set_xlabel("Epochs", fontsize=fontsize)
    ax[1,0].set_ylabel("Invariance loss", fontsize=fontsize)
    ax[1,0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[0,1].plot(std_loss_train_epochs, 'g') #row=0, col=1
    ax[0,1].set_xlabel("Epochs", fontsize=fontsize)
    ax[0,1].set_ylabel("Variance loss", fontsize=fontsize)
    ax[0,1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1,1].plot(cov_loss_train_epochs, 'y') #row=1, col=1
    ax[1,1].set_xlabel("Epochs", fontsize=fontsize)
    ax[1,1].set_ylabel("Covariance loss", fontsize=fontsize)
    ax[1,1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )
    plt.subplots_adjust(hspace=0.5, wspace=0.5) # adjust spacing between plots
    plt.figtext(0.5, 0.01, "Different loss terms in training", ha="center", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{label}/loss_train_epochs.png", dpi=300)
    plt.close()
    # plt.show()

    # Plot loss curves in validation
    fontsize = 20
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].plot(loss_val_epochs, 'r') #row=0, col=0
    ax[0, 0].set_xlabel("Epochs", fontsize=fontsize)
    ax[0, 0].set_ylabel("Total loss", fontsize=fontsize)
    ax[0, 0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1,0].plot(repr_loss_val_epochs, 'b') #row=1, col=0
    ax[1,0].set_xlabel("Epochs", fontsize=fontsize)
    ax[1,0].set_ylabel("Invariance loss", fontsize=fontsize)
    ax[1,0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[0,1].plot(std_loss_val_epochs, 'g') #row=0, col=1
    ax[0,1].set_xlabel("Epochs", fontsize=fontsize)
    ax[0,1].set_ylabel("Variance loss", fontsize=fontsize)
    ax[0,1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1,1].plot(cov_loss_val_epochs, 'y') #row=1, col=1
    ax[1,1].set_xlabel("Epochs", fontsize=fontsize)
    ax[1,1].set_ylabel("Covariance loss", fontsize=fontsize)
    ax[1,1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )
    plt.subplots_adjust(hspace=0.5, wspace=0.5) # adjust spacing between plots
    plt.figtext(0.5, 0.01, "Different loss terms in validation", ha="center", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{label}/loss_val_epochs.png", dpi=300)
    plt.close()

    # Total loss in training and validation across batches
    fontsize = 20
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    ax[0].plot(loss_train_batches, 'r') #row=0, col=0
    ax[0].set_xlabel("Batches", fontsize=fontsize)
    ax[0].set_ylabel("Total loss in training", fontsize=fontsize)
    ax[0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1].plot(loss_val_batches, 'b') #row=1, col=0
    ax[1].set_xlabel("Batches", fontsize=fontsize)
    ax[1].set_ylabel("Total loss in validation", fontsize=fontsize)
    ax[1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )
    plt.subplots_adjust(hspace=0.5, wspace=0.5) # adjust spacing between plots
    plt.figtext(0.5, 0.01, "Total loss in training and validation across batches", ha="center", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{label}/loss_batches.png", dpi=300)
    plt.close()
    # plt.show()

    # Plot LCT AUC vs epochs
    plt.plot(lct_auc_epochs)
    plt.title(f"LCT AUC vs epochs, max: {np.max(lct_auc_epochs):.4f}")
    plt.tight_layout()
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{label}/lct_auc_epochs.png", dpi=300)
    plt.close()

def lct(args, data_train, data_test, labels_train, labels_test, batch_size, train_its, test_its):
    args.x_backbone, args.y_backbone = get_backbones(args)
    args.return_representation = True
    args.return_embedding = False
    # load the desired trained VICReg model
    model_lct = VICReg(args).to(args.device)
    if args.metric == "auc":
        model_lct.load_state_dict(
            torch.load(f"{args.load_vicreg_path}/vicreg_{args.label}_lct_best.pth")
        )
        print(f"loaded {args.load_vicreg_path}/vicreg_{args.label}_lct_best.pth")
    else:
        model_lct.load_state_dict(
            torch.load(f"{args.load_vicreg_path}/vicreg_{args.label}_best.pth")
        )
        print(f"loaded {args.load_vicreg_path}/vicreg_{args.label}_best.pth")

    # perform the LCT for 10 times
    best_auc = 0
    best_imtafe = 0
    best_comb_imtafe = 0
    auc_lst = []
    comb_auc_lst = []
    for it in range(10):
        print(it)
        # obtain the representations from the trained VICReg model
        with torch.no_grad():
            model_lct.eval()
            train_loader = DataLoader(data_train, args.batch_size)
            test_loader = DataLoader(data_test, args.batch_size)
            tr_reps = []
            pbar = tqdm.tqdm(train_loader, total=train_its)
            for i, batch in enumerate(pbar):
                batch = batch.to(args.device)
                tr_reps.append(model_lct(batch, True)[1].detach().cpu().numpy())
                pbar.set_description(f"{i}")
            tr_reps = np.concatenate(tr_reps)
            te_reps = []
            pbar = tqdm.tqdm(test_loader, total=test_its)
            for i, batch in enumerate(pbar):
                batch = batch.to(args.device)
                te_reps.append(model_lct(batch, True)[1].detach().cpu().numpy())
                pbar.set_description(f"{i}")
            te_reps = np.concatenate(te_reps)

        # perform the linear classifier test (LCT) on the VICReg representations
        i = 0
        linear_input_size = tr_reps.shape[1]
        linear_n_epochs = 1000
        linear_learning_rate = 0.001
        linear_batch_size = batch_size
        out_dat_f, out_lbs_f, losses_f, val_losses_f = linear_classifier_test(
             linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, tr_reps, labels_train, te_reps, labels_test,
               n_hidden=args.n_hidden, hidden_size=args.hidden_size)
        auc, imtafe = get_perf_stats( out_lbs_f, out_dat_f )
        ep=0
        step_size = 50
        for j in range(len(losses_f[::step_size])):
            lss = losses_f[::step_size][j]
            val_lss = val_losses_f[::step_size][j]
            print( f"(rep layer {i}) epoch: " + str( ep ) + ", loss: " + str( lss ) + ", val loss: " + str( val_lss ), flush=True)
            ep+=step_size
        print(f"(rep layer {i}) auc: " + str(round(auc, 4)), flush=True)
        print(f"(rep layer {i}) imtafe: " + str(round(imtafe, 1)), flush=True)
        auc_lst.append(auc)
        if auc > best_auc:
            print("new best auc: " + str(round(auc, 4)), flush=True)
            best_auc = auc
            best_imtafe = imtafe
            np.save(args.eval_path + f"{args.label}/linear_losses_best.npy", losses_f)
            np.save(args.eval_path + f"{args.label}/test_linear_cl_best.npy", out_dat_f)
            np.save(args.eval_path + f"{args.label}/test_linear_cl_labels_best.npy", out_lbs_f)
            
        # LCT with raw features + VICReg features
        
        print("LCT with raw features + VICReg features")
        tr_reps_raw = data_train.view(data_train.shape[0], -1)
        te_reps_raw = data_test.view(data_test.shape[0], -1)
        
        tr_reps_comb = torch.cat((torch.tensor(tr_reps), tr_reps_raw), dim=1)
        te_reps_comb = torch.cat((torch.tensor(te_reps), te_reps_raw), dim=1)
        
        linear_input_size = tr_reps_comb.shape[1]
        linear_n_epochs = 1000
        linear_learning_rate = 0.001
        linear_batch_size = batch_size

        out_dat_f, out_lbs_f, losses_f, val_losses_f = linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, tr_reps_comb, labels_train, te_reps_comb, labels_test,
                                                                              n_hidden=args.n_hidden, hidden_size=args.hidden_size )
        auc, imtafe = get_perf_stats( out_lbs_f, out_dat_f )
        ep=0
        step_size = 50
        for j in range(len(losses_f[::step_size])):
            lss = losses_f[::step_size][j]
            val_lss = val_losses_f[::step_size][j]
            print( f"(rep layer {i}) epoch: " + str( ep ) + ", loss: " + str( lss ) + ", val loss: " + str( val_lss ), flush=True)
            ep+=step_size
        print( f"(rep layer {i}) comb auc: "+str( round(auc, 4) ), flush=True)
        print( f"(rep layer {i}) comb imtafe: "+str( round(imtafe, 1) ), flush=True)
        comb_auc_lst.append(auc)
        if imtafe > best_comb_imtafe:
            best_comb_imtafe = imtafe
        print("---------------------")
    print(f"AUC list: {auc_lst}", flush=True)
    print(f"comb AUC list: {comb_auc_lst}", flush=True)
    print(f"max AUC: {max(auc_lst)}", flush=True)
    print(f"max imtafe: {best_imtafe}", flush=True)
    print(f"max comb AUC: {max(comb_auc_lst)}", flush=True)
    print(f"max comb imtafe: {best_comb_imtafe}", flush=True)

def plot_pair_plots(args,data_train, data_test, labels_train, labels_test, batch_size, train_its, test_its):
    # obtain the representations from the trained VICReg model
    args.x_backbone, args.y_backbone = get_backbones(args)
    args.return_representation = True
    args.return_embedding = False
    # load the desired trained VICReg model
    model = VICReg(args).to(args.device)
    if args.metric == "auc":
        model.load_state_dict(
            torch.load(f"{args.load_vicreg_path}/vicreg_{args.label}_lct_best.pth")
        )
        print(f"loaded {args.load_vicreg_path}/vicreg_{args.label}_lct_best.pth")
    else:
        model.load_state_dict(
            torch.load(f"{args.load_vicreg_path}/vicreg_{args.label}_best.pth")
        )
        print(f"loaded {args.load_vicreg_path}/vicreg_{args.label}_best.pth")


    # split the representations into QCD and top
    data_test_QCD = data_test[labels_test == 0]
    data_test_top = data_test[labels_test == 1]

    # Top
    # obtain the representations from the trained VICReg model
    with torch.no_grad():
        model.eval()
        train_loader = DataLoader(data_train, args.batch_size)
        test_loader = DataLoader(data_test_top, args.batch_size)
        tr_reps = []
        pbar = tqdm.tqdm(train_loader, total=train_its)
        for i, batch in enumerate(pbar):
            batch = batch.to(args.device)
            tr_reps.append(model(batch, True)[1].detach().cpu().numpy())
            pbar.set_description(f"{i}")
        tr_reps = np.concatenate(tr_reps)
        te_reps = []
        pbar = tqdm.tqdm(test_loader, total=test_its)
        for i, batch in enumerate(pbar):
            batch = batch.to(args.device)
            te_reps.append(model(batch, True)[1].detach().cpu().numpy())
            pbar.set_description(f"{i}")
        te_reps_top = np.concatenate(te_reps)
    
    np.random.seed(0)
    data = te_reps_top

    # Plot the Pearson coefficients in matrix form
    num_feats = data.shape[1]
    corr_matrix = np.zeros((num_feats, num_feats))

    # Compute the correlation matrix
    for i in range(num_feats):
        for j in range(i):
            corr_coeff, _ = pearsonr(data[:, i], data[:, j])
            corr_matrix[i, j] = corr_coeff

    # Compute the mean
    corr_matrix = np.abs(corr_matrix)
    mean_value = corr_matrix[corr_matrix != 0].mean()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False)

    # Display the mean
    plt.title(f"top Mean Pearson Coefficient: {mean_value:.2f}")
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{args.label}/top_pearson_matrix_{args.metric}.png")
    # plt.show()
    plt.close()

    # plot the distribution of Pearson coefficients
    pearson_coeffs = corr_matrix[corr_matrix != 0]  

    sns.distplot(pearson_coeffs, kde=True)
    plt.axvline(x=np.mean(pearson_coeffs), color='r', linestyle='--', label=f"Mean: {np.mean(pearson_coeffs):.2f}")
    plt.xlabel('Pearson Coefficient')
    plt.ylabel('Density')
    plt.title('Distribution of Pearson Coefficients for Top')
    plt.legend()
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{args.label}/top_pearson_distribution_{args.metric}.png")
    # plt.show()
    plt.close()

    # Plot the pair plots
    num_feats = data.shape[1]
    fig, axs = plt.subplots(num_feats, num_feats, figsize=(30, 30))  # Increase figure size

    for i in range(num_feats):
        for j in range(i + 1):
            # Compute Pearson correlation coefficient using scipy
            corr_coeff, _ = pearsonr(data[:, i], data[:, j])
            
            # Diagonal histograms
            if i == j:
                axs[i, j].hist(data[:, i])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
            else:
                axs[i, j].scatter(data[:, j], data[:, i], s=5)
                axs[i, j].set_title(f"{corr_coeff:.2f}", fontsize=8, pad=-15)  # Decrease font size
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[j, i].axis('off')

    fig.suptitle('Top Pair Plots')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Increase spacing between subplots
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{args.label}/top_pair_plots_{args.metric}.png")
#     plt.show()
    plt.close()

    # QCD
    # obtain the representations from the trained VICReg model
    with torch.no_grad():
        model.eval()
        train_loader = DataLoader(data_train, args.batch_size)
        test_loader = DataLoader(data_test_QCD, args.batch_size)
        tr_reps = []
        pbar = tqdm.tqdm(train_loader, total=train_its)
        for i, batch in enumerate(pbar):
            batch = batch.to(args.device)
            tr_reps.append(model(batch, True)[1].detach().cpu().numpy())
            pbar.set_description(f"{i}")
        tr_reps = np.concatenate(tr_reps)
        te_reps = []
        pbar = tqdm.tqdm(test_loader, total=test_its)
        for i, batch in enumerate(pbar):
            batch = batch.to(args.device)
            te_reps.append(model(batch, True)[1].detach().cpu().numpy())
            pbar.set_description(f"{i}")
        te_reps_QCD = np.concatenate(te_reps)

    # Pair plots for QCD
    data = te_reps_QCD

    num_feats = data.shape[1]
    fig, axs = plt.subplots(num_feats, num_feats, figsize=(30, 30))  # Increase figure size

    for i in range(num_feats):
        for j in range(i + 1):
            # Compute Pearson correlation coefficient using scipy
            corr_coeff, _ = pearsonr(data[:, i], data[:, j])
            
            # Diagonal histograms
            if i == j:
                axs[i, j].hist(data[:, i])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
            else:
                axs[i, j].scatter(data[:, j], data[:, i], s=5)
                axs[i, j].set_title(f"{corr_coeff:.2f}", fontsize=8, pad=-15)  # Decrease font size
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                axs[j, i].axis('off')

    fig.suptitle('QCD Pair Plots')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Increase spacing between subplots
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{args.label}/QCD_pair_plots_{args.metric}.png")
    # plt.show()
    plt.close()

    # Pearson matrix for QCD
    np.random.seed(0)
    data = te_reps_QCD

    num_feats = data.shape[1]
    corr_matrix = np.zeros((num_feats, num_feats))

    # Compute the correlation matrix
    for i in range(num_feats):
        for j in range(i):
            corr_coeff, _ = pearsonr(data[:, i], data[:, j])
            corr_matrix[i, j] = corr_coeff

    # Compute the mean
    corr_matrix = np.abs(corr_matrix)
    mean_value = corr_matrix[corr_matrix != 0].mean()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False)

    # Display the mean
    plt.title(f"QCD Mean Pearson Coefficient: {mean_value:.2f}")
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{args.label}/QCD_pearson_matrix_{args.metric}.png")
    # plt.show()
    plt.close()

    # Plot the distribution of Pearson coefficients
    pearson_coeffs = corr_matrix[corr_matrix != 0]  

    sns.distplot(pearson_coeffs, kde=True)
    plt.axvline(x=np.mean(pearson_coeffs), color='r', linestyle='--', label=f"Mean: {np.mean(pearson_coeffs):.2f}")
    plt.xlabel('Pearson Coefficient')
    plt.ylabel('Density')
    plt.title('Distribution of Pearson Coefficients for QCD')
    plt.legend()
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{args.label}/QCD_pearson_distribution_{args.metric}.png")
    # plt.show()
    plt.close()

    # Top and QCD on the same canvas
    data_qcd = te_reps_QCD
    data_top = te_reps_top

    num_feats = data_qcd.shape[1]
    num_columns = 4
    num_rows = int(np.ceil(num_feats / num_columns))

    # Adjust the figure height for each subplot, e.g., 4 here
    height_per_subplot = 2
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * height_per_subplot))

    # Style settings
    sns.set_style("whitegrid")
    colors = ['blue', 'red']

    for i in range(num_feats):
        row = i // num_columns
        col = i % num_columns
        ax = axs[row, col]

        sns.histplot(data_top[:, i], bins=50, ax=ax, color=colors[0], label='top', alpha=0.6)
        sns.histplot(data_qcd[:, i], bins=50, ax=ax, color=colors[1], label='QCD', alpha=0.6)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(f"Feature {i}")
        ax.legend()
        
        sns.despine(ax=ax)

    # If the number of features isn't a multiple of the columns, we may need to remove some unused subplots
    if num_feats % num_columns != 0:
        for j in range(num_feats, num_rows * num_columns):
            fig.delaxes(axs.flatten()[j])

    fig.suptitle('top and QCD', y=1.02)
    plt.tight_layout(pad=2.0)  # Adjust padding for better appearance
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{args.label}/top_and_QCD_{args.metric}.png")
#     plt.show()
    plt.close()



def plot_tsne(args,data_train, data_test, labels_train, labels_test, batch_size, train_its, test_its):
    # obtain the representations from the trained VICReg model
    args.x_backbone, args.y_backbone = get_backbones(args)
    args.return_representation = True
    args.return_embedding = False
    # load the desired trained VICReg model
    model = VICReg(args).to(args.device)
    if args.metric == "auc":
        model.load_state_dict(
            torch.load(f"{args.load_vicreg_path}/vicreg_{args.label}_lct_best.pth")
        )
        print(f"loaded {args.load_vicreg_path}/vicreg_{args.label}_lct_best.pth")
    else:
        model.load_state_dict(
            torch.load(f"{args.load_vicreg_path}/vicreg_{args.label}_best.pth")
        )
        print(f"loaded {args.load_vicreg_path}/vicreg_{args.label}_best.pth")

    with torch.no_grad():
        model.eval()
        train_loader = DataLoader(data_train, args.batch_size)
        test_loader = DataLoader(data_test, args.batch_size)
        tr_reps = []
        pbar = tqdm.tqdm(train_loader, total=train_its)
        for i, batch in enumerate(pbar):
            batch = batch.to(args.device)
            tr_reps.append(model(batch, True)[1].detach().cpu().numpy())
            pbar.set_description(f"{i}")
        tr_reps = np.concatenate(tr_reps)
        te_reps = []
        pbar = tqdm.tqdm(test_loader, total=test_its)
        for i, batch in enumerate(pbar):
            batch = batch.to(args.device)
            te_reps.append(model(batch, True)[1].detach().cpu().numpy())
            pbar.set_description(f"{i}")
        te_reps = np.concatenate(te_reps)
    # Assuming your data tensor is named 'data' and has a shape of [num_samples, 8]
    # Flatten the data if needed and convert it to numpy
    data_test = te_reps
    data_numpy = data_test.reshape((data_test.shape[0], -1))

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)  # you can change these hyperparameters as needed
    tsne_results = tsne.fit_transform(data_numpy)

    # tsne_results now has a shape of [num_samples, 2], and you can plot it

    labels_numpy = labels_test.cpu().numpy()
    # Use boolean indexing to separate points for each label
    top_points = tsne_results[labels_numpy == 1]
    qcd_points = tsne_results[labels_numpy == 0]

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # Plot each class with a different color and label
    plt.scatter(top_points[:, 0], top_points[:, 1], color='b', alpha=0.5, label='top', s=1)
    plt.scatter(qcd_points[:, 0], qcd_points[:, 1], color='y', alpha=0.5, label='QCD', s=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title("t-SNE visualization of jet features")
    plt.legend(loc='upper right')  # place the legend at the upper right corner
    plt.savefig(f"{project_dir}/models/model_performances/JetClass/{args.label}/tsne_plot_{args.metric}.png", dpi=300, bbox_inches='tight')
    # plt.show()

def main(args):
    # define the global base device
    world_size = torch.cuda.device_count()
    if world_size:
        device = torch.device("cuda:0")
        for i in range(world_size):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = "cpu"
        print("Device: CPU")
    args.device = device
    args.augmentation = augmentation_lct
    
     # load the training and testing dataset
    data_train = load_data(args, "train")
    data_test = load_data(args, "test")

    labels_train = load_labels(args, "train")
    labels_test = load_labels(args, "test")



    n_train = data_train.shape[0]
    n_test = data_test.shape[0]

    batch_size = args.batch_size
    train_its = int(n_train / batch_size)
    test_its = int(n_test / batch_size)
   
    # plot losses
    print("-------------------")
    print("Plotting losses")
    plot_losses(args)
    print("-------------------")
    # plot pair plots
    print("Making pair plots")
    plot_pair_plots(args, data_train, data_test, labels_train, labels_test, batch_size, train_its, test_its)
    print("-------------------")
    # plot t-SNE
    print("Making t-SNE plots")
    plot_tsne(args, data_train, data_test, labels_train, labels_test, batch_size, train_its, test_its)
    print("-------------------")
    # LCT
    print("Doing LCT")
    lct(args, data_train, data_test, labels_train, labels_test, batch_size, train_its, test_its)
    print("-------------------")

if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        action="store",
        default="auc",
        help="metric to use for loading best model",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        action="store",
        default="/ssl-jet-vol-v2/JetClass/processed",
        help="Input directory with the dataset for LCT",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        action="store",
        default=f"{project_dir}/models/model_performances/JetClass",
        help="the evaluation results will be saved at args.eval_path/label",
    )
    parser.add_argument(
        "--load-vicreg-path",
        type=str,
        action="store",
        default=f"{project_dir}/models/trained_models/JetClass",
        help="Load weights from vicreg model if enabled",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        action="store",
        dest="feature_dim",
        default=32,
        help="dimension of learned feature space",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        action="store",
        dest="model_dim",
        default=32,
        help="dimension of the transformer-encoder",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        action="store",
        dest="n_hidden",
        default=0,
        help="number of hidden layers",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        action="store",
        dest="hidden_size",
        default=0,
        help="number of hidden layers",
    )
    parser.add_argument(
        "--x-inputs",
        type=int,
        action="store",
        dest="x_inputs",
        default=3,
        help="number of features/particle for view x",
    )
    parser.add_argument(
        "--y-inputs",
        type=int,
        action="store",
        dest="y_inputs",
        default=3,
        help="number of features/particle for view x",
    )
    parser.add_argument(
        "--transform-inputs",
        type=int,
        action="store",
        dest="transform_inputs",
        default=32,
        help="",
    )
    parser.add_argument(
        "--shared",
        type=bool,
        action="store",
        default=True,
        help="share parameters of backbone",
    )
    parser.add_argument(
        "--lct-best",
        type=bool,
        action="store",
        default=False,
        help="use the model with best lct, otherwise use the one with lowest val loss",
    )
    parser.add_argument(
        "--epoch", type=int, action="store", dest="epoch", default=200, help="Epochs"
    )
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="new",
        help="a label for the model used for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="store",
        dest="batch_size",
        default=2048,
        help="batch_size",
    )
    parser.add_argument(
        "--mlp",
        default="256-256-256",
        help="Size and number of layers of the MLP expander head",
    )
    parser.add_argument(
        "--mask",
        type=bool,
        action="store",
        default=False,
        help="use mask in transformer",
    )
    parser.add_argument(
        "--cmask",
        type=bool,
        action="store",
        default=True,
        help="use continuous mask in transformer",
    )

    args = parser.parse_args()
    main(args)
