import argparse
import copy
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import yaml
from torch import nn

from src.models.transformer import Transformer
from src.models.jet_augs import *
from src.features.perf_eval import get_perf_stats, linear_classifier_test

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
        self.embedding = args.feature_dim
        self.return_embedding = args.return_embedding
        # self.return_representation = args.return_representation
        self.x_projector = Projector(args.mlp, self.embedding)
        self.y_projector = (
            self.x_projector if args.shared else copy.deepcopy(self.x_projector)
        )

    def forward(self, x, x_labels=None, data=None, labels=None, return_rep=False):
        """
        x -> x_aug -> (x_xform) -> x_rep -> x_emb
        y -> y_aug -> (y_xform) -> y_rep -> y_emb
        _aug: augmented
        _xform: transformed by linear layer (skipped because it destroys the zero padding)
        _rep: backbone representation
        _emb: projected embedding
        """
        if return_rep:
            # Don't do augmentation, just return the backbone representation
            y = x.clone()
            x_aug = x.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
            y_aug = y.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
        else:
            x_aug, y_aug = self.augmentation(
                x, x_labels, data, labels, self.args.device
            )  # [batch_size, n_constit, 3]
            x_aug = x_aug.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
            y_aug = y_aug.transpose(1, 2)  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
        #         print(f"x_aug contains nan: {contains_nan(x_aug)}")
        #         print(f"y_aug contains nan: {contains_nan(y_aug)}")

        # x_xform = self.x_transform.to(torch.double)(
        #     x_aug.x.double()
        # )  # [batch_size, n_constit, transform_inputs]?
        # y_xform = self.y_transform.to(torch.double)(
        #     y_aug.x.double()
        # )  # [batch_size, n_constit, transform_inputs]?

        x_rep = self.x_backbone(
            x_aug, use_mask=self.args.mask, use_continuous_mask=self.args.cmask
        )  # [batch_size, output_dim]
        y_rep = self.y_backbone(
            y_aug, use_mask=self.args.mask, use_continuous_mask=self.args.cmask
        )  # [batch_size, output_dim]
        #         print(f"x_rep contains nan: {contains_nan(x_rep)}")
        #         print(f"y_rep contains nan: {contains_nan(y_rep)}")
        if return_rep:
            return x_rep, y_rep

        x_emb = self.x_projector(x_rep)  # [batch_size, embedding_size]
        y_emb = self.y_projector(y_rep)  # [batch_size, embedding_size]
        #         print(f"x_emb contains nan: {contains_nan(x_emb)}")
        #         print(f"y_emb contains nan: {contains_nan(y_emb)}")
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
    x_backbone = Transformer(input_dim=args.x_inputs, output_dim=args.feature_dim)
    y_backbone = x_backbone if args.shared else copy.deepcopy(x_backbone)
    return x_backbone, y_backbone


def augmentation(batch_x, batch_x_label, data_train, labels_train, device):
    """
    For each jet in the batch, sample another jet of the same class (signal/background)
    from the training data as the second view y and return x, y.

    batch_x: [batch_size, 3, n_constit]
    """
    # List to store the y samples for the entire batch
    batch_y = []

    # Loop over all jets and labels in the batch
    for x, x_label in zip(batch_x, batch_x_label):
        # Indices of all data points with the same label as x_label
        same_label_indices = torch.where(labels_train == x_label)[0]

        # Randomly select one of the indices while ensuring it's not the same as the current jet
        x_index = torch.where((data_train == x).all(dim=1).all(dim=1))[0][0].to(device)
        potential_indices = same_label_indices[same_label_indices != x_index]
        y_index = potential_indices[
            torch.randint(0, len(potential_indices), (1,))
        ].item()

        # Get the data point corresponding to the chosen index
        y = data_train[y_index]
        batch_y.append(y)
        assert (
            labels_train[y_index].item() == x_label
        ), "Mismatch in labels between x and y."
        assert not torch.all(torch.eq(x, y)), "sampled the same jet"

    batch_y = torch.stack(batch_y).to(device)
    return batch_x, batch_y


# load the datafiles
def load_data(dataset_path, flag, n_files=-1):
    data_files = glob.glob(f"{dataset_path}/{flag}/processed/3_features/*")

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(f"{dataset_path}/{flag}/processed/3_features/data_{i}.pt")
        print(f"--- loaded file {i} from `{flag}` directory")
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

    n_epochs = args.epoch
    batch_size = args.batch_size
    outdir = args.outdir
    label = args.label

    model_loc = f"{outdir}/trained_models/"
    model_perf_loc = f"{outdir}/model_performances/{label}"
    model_dict_loc = f"{outdir}/model_dicts/"
    os.system(
        f"mkdir -p {model_loc} {model_perf_loc} {model_dict_loc}"
    )  # -p: create parent dirs if needed, exist_ok

    # prepare data
    data_train = load_data(args.dataset_path, "train", n_files=args.num_train_files)
    data_valid = load_data(args.dataset_path, "val", n_files=args.num_val_files)
    data_test = load_data(args.dataset_path, "test", n_files=1)

    labels_train = load_labels(args.dataset_path, "train", n_files=args.num_train_files)
    labels_valid = load_labels(args.dataset_path, "val", n_files=args.num_val_files)
    labels_test = load_labels(args.dataset_path, "test", n_files=1)

    # only take the first 10k jets for LCT
    labels_train_lct = labels_train[:10000]
    labels_test_lct = labels_test[:10000]

    labels_train = torch.tensor([t.item() for t in labels_train]).to(device)
    labels_valid = torch.tensor([t.item() for t in labels_valid]).to(device)
    labels_train_lct = torch.tensor([t.item() for t in labels_train_lct]).to(device)
    labels_test_lct = torch.tensor([t.item() for t in labels_test_lct]).to(device)
    
    data_train = torch.stack(data_train).to(device)
    data_valid = torch.stack(data_valid).to(device)
    data_test = torch.stack(data_test).to(device)

    n_train = data_train.shape[0]
    n_val = data_valid.shape[0]

    args.augmentation = augmentation

    args.x_inputs = 3
    args.y_inputs = 3

    args.x_backbone, args.y_backbone = get_backbones(args)
    model = VICReg(args).to(args.device)
    print(model)

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

        dataset = TensorDataset(data_train, labels_train)
        train_loader = DataLoader(dataset, batch_size)
        model.train()
        pbar_t = tqdm.tqdm(train_loader, total=train_its)
        for _, (batch_data, batch_labels) in enumerate(pbar_t):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            if args.return_all_losses:
                loss, repr_loss, std_loss, cov_loss = model.forward(batch_data, batch_labels, data_train, labels_train)
                #             print(loss, repr_loss, std_loss, cov_loss)
                repr_loss_train_epoch.append(repr_loss.detach().cpu().item())
                std_loss_train_epoch.append(std_loss.detach().cpu().item())
                cov_loss_train_epoch.append(cov_loss.detach().cpu().item())
            else:
                loss = model.forward(batch_data, batch_labels, data_train, labels_train)
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()
            loss_train_batches.append(loss)
            loss_train_epoch.append(loss)
            pbar_t.set_description(f"Training loss: {loss:.4f}")
        #             print(f"Training loss: {loss:.4f}")
        l_train = np.mean(np.array(loss_train_epoch))
        print(f"Training loss: {l_train:.4f}")
        model.eval()
        dataset = TensorDataset(data_valid, labels_valid)
        valid_loader = DataLoader(dataset, batch_size)
        pbar_v = tqdm.tqdm(valid_loader, total=val_its)
        with torch.no_grad():
            for _, (batch_data, batch_labels) in enumerate(pbar_v):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                if args.return_all_losses:
                    loss, repr_loss, std_loss, cov_loss = model.forward(batch_data, batch_labels, data_valid, labels_valid)
                    repr_loss_val_epoch.append(repr_loss.detach().cpu().item())
                    std_loss_val_epoch.append(std_loss.detach().cpu().item())
                    cov_loss_val_epoch.append(cov_loss.detach().cpu().item())
                    loss = loss.detach().cpu().item()
                else:
                    loss = model.forward(batch_data, batch_labels, data_valid, labels_valid).cpu().item()
                loss_val_batches.append(loss)
                loss_val_epoch.append(loss)
                pbar_v.set_description(f"Validation loss: {loss:.4f}")
        #             print(f"Validation loss: {loss:.4f}")
        l_val = np.mean(np.array(loss_val_epoch))
        print(f"Validation loss: {l_val:.4f}")
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
        if m % 10 == 0:
            # do a short LCT
            model.eval()
            print("Doing LCT")
            with torch.no_grad():
                train_loader = DataLoader(data_train[:10000], args.batch_size)
                test_loader = DataLoader(data_test[:10000], args.batch_size)
                tr_reps = []
                batch_size = args.batch_size
                # train_its = int(10000 / batch_size)
                # test_its = int(10000 / batch_size)
                # pbar = tqdm.tqdm(train_loader, total=train_its)
                for i, batch in enumerate(train_loader):
                    batch = batch.to(args.device)
                    tr_reps.append(
                        model(batch, return_rep=True)[0].detach().cpu().numpy()
                    )
                    # pbar.set_description(f"{i}")
                tr_reps = np.concatenate(tr_reps)
                te_reps = []
                # pbar = tqdm.tqdm(test_loader, total=test_its)
                for i, batch in enumerate(test_loader):
                    batch = batch.to(args.device)
                    te_reps.append(
                        model(batch, return_rep=True)[0].detach().cpu().numpy()
                    )
                    # pbar.set_description(f"{i}")
                te_reps = np.concatenate(te_reps)

            # perform the linear classifier test (LCT) on the representations
            i = 0
            linear_input_size = tr_reps.shape[1]
            linear_n_epochs = 750
            linear_learning_rate = 0.001
            linear_batch_size = 1000
            out_dat_f, out_lbs_f, losses_f = linear_classifier_test(
                linear_input_size,
                linear_batch_size,
                linear_n_epochs,
                linear_learning_rate,
                tr_reps,
                labels_train_lct,
                te_reps,
                labels_test_lct,
            )
            auc, imtafe = get_perf_stats(out_lbs_f.cpu(), out_dat_f)
            ep = 0
            step_size = 100
            for lss in losses_f[::step_size]:
                print(
                    f"(rep layer {i}) epoch: " + str(ep) + ", loss: " + str(lss),
                    flush=True,
                )
                ep += step_size
            print(f"(rep layer {i}) auc: " + str(round(auc, 4)), flush=True)
            print(f"(rep layer {i}) imtafe: " + str(round(imtafe, 1)), flush=True)
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
        "--feature-dim",
        type=int,
        action="store",
        dest="feature_dim",
        default=1000,
        help="dimension of learned feature space",
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
    # parser.add_argument(
    #     "--device",
    #     action="store",
    #     dest="device",
    #     default="cuda",
    #     help="device to train gnn; follow pytorch convention",
    # )
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
        "--return-all-losses",
        type=bool,
        action="store",
        dest="return_all_losses",
        default=True,
        help="return the three terms in the loss function as well",
    )

    args = parser.parse_args()
    main(args)
