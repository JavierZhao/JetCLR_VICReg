# Perform the Linear Classifier Test (LCT) on the representations learned by VICReg.

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

project_dir = Path(__file__).resolve().parents[2]


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


def augmentation(args, x, device):
    """
    Applies all the augmentations specified in the args
    """
    y = x.clone()
    x = x.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
    y = y.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
    return x, y


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

    args.augmentation = augmentation

    args.x_inputs = 3
    args.y_inputs = 3

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
    
    # print number of hidden layers and hidden size
    if args.n_hidden != 0:
        print(f"n_hidden: {args.n_hidden}, hidden_size: {args.hidden_size}", flush=True)
    else:
        print("no hidden layers")
    
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

    # perform the LCT for 10 times
    best_auc = 0
    auc_lst = []
    comb_auc_lst = []
    for it in range(10):
        print(it)
        # obtain the representations from the trained VICReg model
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

        out_dat_f, out_lbs_f, losses_f, val_losses_f = linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, tr_reps_comb, labels_train, te_reps_comb, labels_test,n_hidden=args.n_hidden, hidden_size=args.hidden_size )
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
        print("---------------------")
    print(f"AUC list: {auc_lst}", flush=True)
    print(f"comb AUC list: {comb_auc_lst}", flush=True)
    print(f"max AUC: {max(auc_lst)}", flush=True)
    print(f"max comb AUC: {max(comb_auc_lst)}", flush=True)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        action="store",
        default=f"{project_dir}/data/processed/train/",
        help="Input directory with the dataset for LCT",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        action="store",
        default=f"{project_dir}/models/model_performances/",
        help="the evaluation results will be saved at eval-path/label",
    )
    parser.add_argument(
        "--load-vicreg-path",
        type=str,
        action="store",
        default=f"{project_dir}/models/trained_models/",
        help="Load weights from vicreg model if enabled",
    )
    parser.add_argument(
        "--num-train-files",
        type=int,
        default=1,
        help="Number of files to use for training",
    )
    parser.add_argument(
        "--num-test-files",
        type=int,
        default=1,
        help="Number of files to use for testing",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        action="store",
        dest="outdir",
        default=f"{project_dir}/models/",
        help="Output directory",
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
        "--metric",
        type=str,
        action="store",
        default="auc",
        help="metric to use for loading best model",
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
