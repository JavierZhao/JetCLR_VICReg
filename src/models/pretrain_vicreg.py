import argparse
import copy
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
import tqdm
import yaml
from torch import nn

from src.models.transformer import Transformer
from src.models.jet_augs import *

# if torch.cuda.is_available():
#     import setGPU  # noqa: F401

project_dir = Path(__file__).resolve().parents[2]

# definitions = f"{project_dir}/src/data/definitions.yml"
# with open(definitions) as yaml_file:
#     defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

# N = defn["nobj_2"]  # number of charged particles
# N_sv = defn["nobj_3"]  # number of SVs
# n_targets = len(defn["reduced_labels"])  # number of classes
# params = defn["features_2"]
# params_sv = defn["features_3"]


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(
            args.mlp.split("-")[-1]
        )  # size of the last layer of the MLP projector
        self.x_transform = nn.Sequential(
            nn.BatchNorm1d(args.x_inputs),
            nn.Linear(args.x_inputs, args.transform_inputs),
            nn.BatchNorm1d(args.transform_inputs),
            nn.ReLU(),
        )
        self.y_transform = nn.Sequential(
            nn.BatchNorm1d(args.y_inputs),
            nn.Linear(args.y_inputs, args.transform_inputs),
            nn.BatchNorm1d(args.transform_inputs),
            nn.ReLU(),
        )
        self.augmentation = args.augmentation
        self.x_backbone = args.x_backbone
        self.y_backbone = args.y_backbone
        self.N_x = self.x_backbone.input_dim
        self.N_y = self.y_backbone.input_dim
        self.embedding = args.Do
        self.return_embedding = args.return_embedding
        self.return_representation = args.return_representation
        self.x_projector = Projector(args.mlp, self.embedding)
        self.y_projector = (
            self.x_projector if args.shared else copy.deepcopy(self.x_projector)
        )

    def forward(self, x, y):
        """
        x -> x_aug -> x_xform -> x_rep -> x_emb
        y -> y_aug -> y_xform -> y_rep -> y_emb
        _aug: augmented
        _xform: transformed by linear layer
        _rep: backbone representation
        _emb: projected embedding
        """
        # x: [N_x, x_inputs]
        # y: [N_y, y_inputs]
        x_aug = self.augmentation(self.args, x, self.args.device)
        y_aug = self.augmentation(self.args, y, self.args.device)

        x_xform = x_aug
        y_xform = y_aug
        x_xform.x = self.x_transform.to(torch.double)(
            x_aug.x.double()
        )  # [N_x, transform_inputs]?
        y_xform.x = self.y_transform.to(torch.double)(
            y_aug.x.double()
        )  # [N_y, transform_inputs]?

        x_rep = self.x_backbone(x_aug)  # [batch_size, output_dim]
        y_rep = self.y_backbone(y_aug)  # [batch_size, output_dim]
        if self.return_representation:
            return x_rep, y_rep

        x_emb = self.x_projector(x_rep)  # [batch_size, embedding_size]
        y_emb = self.y_projector(y_rep)  # [batch_size, embedding_size]
        if self.return_embedding:
            return x_emb, y_emb
        x = x_emb
        y = y_emb
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        if args.return_all_losses:
            return loss, repr_loss, std_loss, cov_loss
        else:
            return loss


def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_backbones(args):
    x_backbone = Transformer(input_dim=args.transform_inputs)
    y_backbone = x_backbone if args.shared else copy.deepcopy(x_backbone)
    return x_backbone, y_backbone


def augmentation(args, batch, device):
    """
    batch: DataBatch(x=[12329, 7], y=[batch_size], batch=[12329], ptr=[257])
    """
    if args.do_rotation:
        batch = rotate_jets(batch, device)
    if args.do_cf:
        batch = collinear_fill_jets(batch, device, split_ratio=args.split_ratio)
    if args.do_ptd:
        batch = distort_jets(batch, device)
    if args.do_translation:
        batch = translate_jets(batch, device, width=1.0)
    if args.do_rescale:
        batch = rescale_pts(batch, device)
    return batch.to(device)


# load the datafiles
def load_data(dataset_path, flag, n_files=-1):
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/*")

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(f"{dataset_path}/{flag}/processed/data_{i}.pt")
        print(f"--- loaded file {i} from `{flag}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data


def main(args):
    n_epochs = args.epoch
    batch_size = args.batch_size
    outdir = args.outdir
    label = args.label
    args.augmentation = augmentation

    model_loc = f"{outdir}/trained_models/"
    model_perf_loc = f"{outdir}/model_performances/"
    model_dict_loc = f"{outdir}/model_dicts/"
    os.system(
        f"mkdir -p {model_loc} {model_perf_loc} {model_dict_loc}"
    )  # -p: create parent dirs if needed, exist_ok

    # prepare data
    data_train = load_data(args.dataset_path, "train", n_files=args.num_train_files)
    data_valid = load_data(args.dataset_path, "val", n_files=args.num_val_files)

    n_train = len(data_train)
    n_val = len(data_valid)

    args.x_inputs = 7
    args.y_inputs = 7

    args.x_backbone, args.y_backbone = get_backbones(args)
    model = VICReg(args).to(args.device)

    train_its = int(n_train / batch_size)
    val_its = int(n_val / batch_size)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_val_epochs = []  # loss recorded for each epoch
    repr_loss_val_epochs, std_loss_val_epochs, cov_loss_val_epochs = [], [], []
    # invariance, variance, covariance loss recorded for each epoch
    loss_val_batches = []  # loss recorded for each batch
    loss_train_epochs = []  # loss recorded for each epoch
    repr_loss_train_epochs, std_loss_train_epochs, cov_loss_train_epochs = [], [], []
    # invariance, variance, covariance loss recorded for each epoch
    loss_train_batches = []  # loss recorded for each batch
    l_val_best = 999999
    for m in range(n_epochs):
        print(f"Epoch {m}\n")
        loss_train_epoch = []  # loss recorded for each batch in this epoch
        repr_loss_train_epoch, std_loss_train_epoch, cov_loss_train_epoch = [], [], []
        # invariance, variance, covariance loss recorded for each batch in this epoch
        loss_val_epoch = []  # loss recorded for each batch in this epoch
        repr_loss_val_epoch, std_loss_val_epoch, cov_loss_val_epoch = [], [], []
        # invariance, variance, covariance loss recorded for each batch in this epoch

        train_loader = DataLoader(data_train, batch_size)
        model.train()
        pbar = tqdm.tqdm(train_loader, total=train_its)
        for _, batch in tqdm.tqdm(enumerate(train_loader)):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            if args.return_all_losses:
                loss, repr_loss, std_loss, cov_loss = model.forward(
                    batch, copy.deepcopy(batch)
                )
                repr_loss_train_epoch.append(repr_loss.detach().cpu().item())
                std_loss_train_epoch.append(std_loss.detach().cpu().item())
                cov_loss_train_epoch.append(cov_loss.detach().cpu().item())
            else:
                loss = model.forward(batch, copy.deepcopy(batch))
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()
            loss_train_batches.append(loss)
            loss_train_epoch.append(loss)
            pbar.set_description(f"Training loss: {loss:.4f}")
        model.eval()
        valid_loader = DataLoader(data_valid, batch_size)
        pbar = tqdm.tqdm(valid_loader, total=val_its)
        for _, batch in tqdm.tqdm(enumerate(valid_loader)):
            batch = batch.to(args.device)
            if args.return_all_losses:
                loss, repr_loss, std_loss, cov_loss = model.forward(
                    batch, copy.deepcopy(batch)
                )
                repr_loss_val_epoch.append(repr_loss.detach().cpu().item())
                std_loss_val_epoch.append(std_loss.detach().cpu().item())
                cov_loss_val_epoch.append(cov_loss.detach().cpu().item())
                loss = loss.detach().cpu().item()
            else:
                loss = model.forward(batch, batch.deepcopy()).cpu().item()
            loss_val_batches.append(loss)
            loss_val_epoch.append(loss)
            pbar.set_description(f"Validation loss: {loss:.4f}")
        l_val = np.mean(np.array(loss_val_epoch))
        l_train = np.mean(np.array(loss_train_epoch))
        loss_val_epochs.append(l_val)
        loss_train_epochs.append(l_train)

        if args.return_all_losses:
            repr_l_val = np.mean(np.array(repr_loss_val_epoch))
            repr_l_train = np.mean(np.array(repr_loss_train_epoch))
            std_l_val = np.mean(np.array(std_loss_val_epoch))
            std_l_train = np.mean(np.array(std_loss_train_epoch))
            cov_l_val = np.mean(np.array(cov_loss_val_epoch))
            cov_l_train = np.mean(np.array(cov_loss_train_epoch))

            repr_loss_val_epochs.append(repr_l_val)
            std_loss_val_epochs.append(std_l_val)
            cov_loss_val_epochs.append(cov_l_val)

            repr_loss_train_epochs.append(repr_l_train)
            std_loss_train_epochs.append(std_l_train)
            cov_loss_train_epochs.append(cov_l_train)
        # save the model
        if l_val < l_val_best:
            print("New best model")
            l_val_best = l_val
            torch.save(model.state_dict(), f"{model_loc}/vicreg_{label}_best.pth")
        torch.save(model.state_dict(), f"{model_loc}/vicreg_{label}_last.pth")
    # After training
    np.save(
        f"{model_perf_loc}/vicreg_{label}_loss_train_epochs.npy",
        np.array(loss_train_epochs),
    )
    np.save(
        f"{model_perf_loc}/vicreg_{label}_loss_train_batches.npy",
        np.array(loss_train_batches),
    )
    np.save(
        f"{model_perf_loc}/vicreg_{label}_loss_val_epochs.npy",
        np.array(loss_val_epochs),
    )
    np.save(
        f"{model_perf_loc}/vicreg_{label}_loss_val_batches.npy",
        np.array(loss_val_batches),
    )
    if args.return_all_losses:
        np.save(
            f"{model_perf_loc}/vicreg_{label}_repr_loss_train_epochs.npy",
            np.array(repr_loss_train_epochs),
        )
        np.save(
            f"{model_perf_loc}/vicreg_{label}_std_loss_train_epochs.npy",
            np.array(std_loss_train_epochs),
        )
        np.save(
            f"{model_perf_loc}/vicreg_{label}_cov_loss_train_epochs.npy",
            np.array(cov_loss_train_epochs),
        )
        np.save(
            f"{model_perf_loc}/vicreg_{label}_repr_loss_val_epochs.npy",
            np.array(repr_loss_val_epochs),
        )
        np.save(
            f"{model_perf_loc}/vicreg_{label}_std_loss_val_epochs.npy",
            np.array(std_loss_val_epochs),
        )
        np.save(
            f"{model_perf_loc}/vicreg_{label}_cov_loss_val_epochs.npy",
            np.array(cov_loss_val_epochs),
        )


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        action="store",
        default=f"{project_dir}/data/processed/train/",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--num-train-files",
        type=int,
        default=12,
        help="Number of files to use for training",
    )
    parser.add_argument(
        "--num-val-files",
        type=int,
        default=4,
        help="Number of files to use for validation",
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
        "--Do", type=int, action="store", dest="Do", default=64, help="Do"
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
        "--epoch", type=int, action="store", dest="epoch", default=200, help="Epochs"
    )
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="new",
        help="a label for the model",
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
        "--device",
        action="store",
        dest="device",
        default="cuda",
        help="device to train gnn; follow pytorch convention",
    )
    parser.add_argument(
        "--mlp",
        default="256-256-256",
        help="Size and number of layers of the MLP expander head",
    )
    parser.add_argument(
        "--sim-coeff",
        type=float,
        default=25.0,
        help="Invariance regularization loss coefficient",
    )
    parser.add_argument(
        "--std-coeff",
        type=float,
        default=25.0,
        help="Variance regularization loss coefficient",
    )
    parser.add_argument(
        "--cov-coeff",
        type=float,
        default=1.0,
        help="Covariance regularization loss coefficient",
    )
    parser.add_argument(
        "--return-embedding",
        type=bool,
        action="store",
        dest="return_embedding",
        default=False,
        help="return_embedding",
    )
    parser.add_argument(
        "--return-representation",
        type=bool,
        action="store",
        dest="return_representation",
        default=False,
        help="return_representation",
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
        "--do-rescale",
        type=bool,
        action="store",
        dest="do_rescale",
        default=True,
        help="do rescale_pts",
    )
    parser.add_argument(
        "split-ratio",
        type=float,
        action="store",
        dest="split_ratio",
        default=0.5,
        help="split_ratio param in collinear_fill_jets",
    )
    parser.add_argument(
        "--return-all-losses",
        type=bool,
        action="store",
        dest="return_all_losses",
        default=False,
        help="return the three terms in the loss function as well",
    )

    args = parser.parse_args()
    main(args)
