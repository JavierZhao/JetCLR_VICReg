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
def load_data(dataset_path, flag, n_files=-1):
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/3_features/*")

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(f"{dataset_path}/{flag}/processed/3_features/data_{i}.pt")
        print(f"--- loaded data file {i} from `{flag}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data


def load_labels(dataset_path, flag, n_files=-1):
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/3_features/*")

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(f"{dataset_path}/{flag}/processed/3_features/labels_{i}.pt")
        print(f"--- loaded label file {i} from `{flag}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data


def get_backbones(args):
    x_backbone = Transformer(input_dim=args.x_inputs)
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
    model.load_state_dict(
        torch.load(f"{args.load_vicreg_path}/vicreg_{args.label}_best.pth")
    )

    # load the training and testing dataset
    data_train = load_data(args.dataset_path, "train", n_files=args.num_train_files)
    data_test = load_data(args.dataset_path, "test", n_files=args.num_test_files)
    labels_train = load_labels(args.dataset_path, "train", n_files=args.num_train_files)
    labels_test = load_labels(args.dataset_path, "test", n_files=args.num_test_files)

    # concatenate the training and testing datasets
    data_train = torch.stack(data_train)
    data_test = torch.stack(data_test)
    labels_train = torch.tensor([t.item() for t in labels_train])
    labels_test = torch.tensor([t.item() for t in labels_test])

    n_train = data_train.shape[0]
    n_test = data_test.shape[0]

    batch_size = args.batch_size
    train_its = int(n_train / batch_size)
    test_its = int(n_test / batch_size)

    # obtain the representations from the trained VICReg model
    with torch.no_grad():
        model.eval()
        train_loader = DataLoader(data_train, args.batch_size)
        test_loader = DataLoader(data_test, args.batch_size)
        tr_reps = []
        pbar = tqdm.tqdm(train_loader, total=train_its)
        for i, batch in enumerate(pbar):
            batch = batch.to(args.device)
            tr_reps.append(model(batch)[0].detach().cpu().numpy())
            pbar.set_description(f"{i}")
        tr_reps = np.concatenate(tr_reps)
        te_reps = []
        pbar = tqdm.tqdm(test_loader, total=test_its)
        for i, batch in enumerate(pbar):
            batch = batch.to(args.device)
            te_reps.append(model(batch)[0].detach().cpu().numpy())
            pbar.set_description(f"{i}")
        te_reps = np.concatenate(te_reps)

    # perform the linear classifier test (LCT) on the representations
    i = 0
    linear_input_size = tr_reps.shape[1]
    linear_n_epochs = 750
    linear_learning_rate = 0.001
    linear_batch_size = 1024
    out_dat_f, out_lbs_f, losses_f = linear_classifier_test(
        linear_input_size,
        linear_batch_size,
        linear_n_epochs,
        linear_learning_rate,
        tr_reps,
        labels_train,
        te_reps,
        labels_test,
    )
    auc, imtafe = get_perf_stats(out_lbs_f, out_dat_f)
    ep = 0
    step_size = 25
    for lss in losses_f[::step_size]:
        print(f"(rep layer {i}) epoch: " + str(ep) + ", loss: " + str(lss), flush=True)
        ep += step_size
    print(f"(rep layer {i}) auc: " + str(round(auc, 4)), flush=True)
    print(f"(rep layer {i}) imtafe: " + str(round(imtafe, 1)), flush=True)
    np.save(args.eval_path + f"{args.label}/linear_losses_{i}.npy", losses_f)
    np.save(args.eval_path + f"{args.label}/test_linear_cl_{i}.npy", out_dat_f)
    np.save(args.eval_path + f"{args.label}/test_linear_cl_labels_{i}.npy", out_lbs_f)


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
        default=12,
        help="Number of files to use for training",
    )
    parser.add_argument(
        "--num-test-files",
        type=int,
        default=4,
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
        "--transform-inputs",
        type=int,
        action="store",
        dest="transform_inputs",
        default=32,
        help="transform_inputs",
    )
    parser.add_argument(
        "--De", type=int, action="store", dest="De", default=32, help="De"
    )
    parser.add_argument(
        "--Do", type=int, action="store", dest="Do", default=1000, help="Do"
    )
    parser.add_argument(
        "--hidden", type=int, action="store", dest="hidden", default=128, help="hidden"
    )
    parser.add_argument(
        "--shared",
        type=bool,
        action="store",
        default=False,
        help="share parameters of backbone",
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
        default=1024,
        help="batch_size",
    )
    parser.add_argument(
        "--mlp",
        default="256-256-256",
        help="Size and number of layers of the MLP expander head",
    )
    parser.add_argument(
        "--do-translation",
        type=bool,
        action="store",
        dest="do_translation",
        default=True,
        help="do_translation",
    )
    parser.add_argument(
        "--do-rotation",
        type=bool,
        action="store",
        dest="do_rotation",
        default=True,
        help="do_rotation",
    )
    parser.add_argument(
        "--do-cf",
        type=bool,
        action="store",
        dest="do_cf",
        default=True,
        help="do collinear splitting",
    )
    parser.add_argument(
        "--do-ptd",
        type=bool,
        action="store",
        dest="do_ptd",
        default=True,
        help="do soft splitting (distort_jets)",
    )
    parser.add_argument(
        "--nconstit",
        type=int,
        action="store",
        dest="nconstit",
        default=50,
        help="number of constituents per jet",
    )
    parser.add_argument(
        "--ptst",
        type=float,
        action="store",
        dest="ptst",
        default=0.1,
        help="strength param in distort_jets",
    )
    parser.add_argument(
        "--ptcm",
        type=float,
        action="store",
        dest="ptcm",
        default=0.1,
        help="pT_clip_min param in distort_jets",
    )
    parser.add_argument(
        "--trsw",
        type=float,
        action="store",
        dest="trsw",
        default=1.0,
        help="width param in translate_jets",
    )

    args = parser.parse_args()
    main(args)
