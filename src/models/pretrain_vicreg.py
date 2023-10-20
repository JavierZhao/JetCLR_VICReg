import argparse
import copy
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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

    def forward(self, x, return_rep=False):
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
            x_aug = x.transpose(
                1, 2
            )  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
            y_aug = y.transpose(
                1, 2
            )  # [batch_size, 3, n_constit] -> [batch_size, n_constit, 3]
        else:
            x_aug, y_aug = self.augmentation(
                self.args, x, self.args.device
            )  # [batch_size, n_constit, 3]
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
    x_backbone = Transformer(input_dim=args.x_inputs, output_dim=args.feature_dim, model_dim=args.model_dim, dim_feedforward=args.model_dim)
    y_backbone = x_backbone if args.shared else copy.deepcopy(x_backbone)
    return x_backbone, y_backbone


def augmentation(args, x, device):
    """
    Applies all the augmentations specified in the args
    """
    # crop all jets to a fixed number of constituents (default=50)
    # x = crop_jets(x, args.nconstit)
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


# load the datafiles
def load_data(args, flag):
    frac = args.percent_data
    dataset_path = args.dataset_path
    data_file = f"{dataset_path}/{flag}_{frac}%/data/data.pt"
    data = torch.load(data_file)
    print(f"--- loaded data file from `{flag}_{frac}%` directory")            
    return data

# labels are only used for the LCT
def load_labels(args, flag):
    frac = args.percent_data
    dataset_path = args.dataset_path
    data_file = f"{dataset_path}/{flag}_{frac}%/label/labels.pt"
    data = torch.load(data_file)        
    print(f"--- loaded label file from `{flag}_{frac}%` directory")    
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

    model_loc = f"{outdir}/trained_models/JetClass/{label}"
    model_perf_loc = f"{outdir}/model_performances/JetClass/{label}"
    os.system(
        f"mkdir -p {model_loc} {model_perf_loc} "
    )  # -p: create parent dirs if needed, exist_ok

    # prepare data
    data_train = load_data(args, "train")
    data_valid = load_data(args, "val")
    data_test = load_data(args, "test")

    labels_train = load_labels(args, "train")
    labels_test = load_labels(args, "test")

    # only take the first 10k jets for LCT
    labels_train = labels_train[:10000]
    labels_test = labels_test[:10000]

    n_train = len(data_train)
    n_val = len(data_valid)

    args.augmentation = augmentation

    args.x_inputs = 3
    args.y_inputs = 3

    args.x_backbone, args.y_backbone = get_backbones(args)
    model = VICReg(args).to(args.device)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)

    train_its = int(n_train / batch_size)
    val_its = int(n_val / batch_size)

    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.0001 * batch_size / 512)
    lct_auc_epochs = [] # LCT AUC recorded for each epoch
    loss_val_epochs = []  # loss recorded for each epoch
    repr_loss_val_epochs, std_loss_val_epochs, cov_loss_val_epochs = [], [], []
    # invariance, variance, covariance loss recorded for each epoch
    loss_val_batches = []  # loss recorded for each batch
    loss_train_epochs = []  # loss recorded for each epoch
    repr_loss_train_epochs, std_loss_train_epochs, cov_loss_train_epochs = [], [], []
    # invariance, variance, covariance loss recorded for each epoch
    loss_train_batches = []  # loss recorded for each batch
    l_val_best = 999999
    lct_auc_best = 0
    for m in range(n_epochs):
        print(f"Epoch {m}\n")
        loss_train_epoch = []  # loss recorded for each batch in this epoch
        repr_loss_train_epoch, std_loss_train_epoch, cov_loss_train_epoch = [], [], []
        # invariance, variance, covariance loss recorded for each batch in this epoch
        loss_val_epoch = []  # loss recorded for each batch in this epoch
        repr_loss_val_epoch, std_loss_val_epoch, cov_loss_val_epoch = [], [], []
        # invariance, variance, covariance loss recorded for each batch in this epoch

        # Training
        train_loader = DataLoader(data_train, batch_size)
        model.train()
        pbar_t = tqdm.tqdm(train_loader, total=train_its)
        for _, batch in enumerate(pbar_t):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            if args.return_all_losses:
                loss, repr_loss, std_loss, cov_loss = model.forward(batch)
                repr_loss_train_epoch.append(repr_loss.detach().cpu().item())
                std_loss_train_epoch.append(std_loss.detach().cpu().item())
                cov_loss_train_epoch.append(cov_loss.detach().cpu().item())
            else:
                loss = model.forward(batch)
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()
            loss_train_batches.append(loss)
            loss_train_epoch.append(loss)
            pbar_t.set_description(f"Training loss: {loss:.4f}")
        l_train = np.mean(np.array(loss_train_epoch))
        print(f"Training loss: {l_train:.4f}")

        # Validation
        model.eval()
        valid_loader = DataLoader(data_valid, batch_size)
        pbar_v = tqdm.tqdm(valid_loader, total=val_its)
        with torch.no_grad():
            for _, batch in enumerate(pbar_v):
                batch = batch.to(args.device)
                if args.return_all_losses:
                    loss, repr_loss, std_loss, cov_loss = model.forward(batch)
                    repr_loss_val_epoch.append(repr_loss.detach().cpu().item())
                    std_loss_val_epoch.append(std_loss.detach().cpu().item())
                    cov_loss_val_epoch.append(cov_loss.detach().cpu().item())
                    loss = loss.detach().cpu().item()
                else:
                    loss = model.forward(batch).cpu().item()
                loss_val_batches.append(loss)
                loss_val_epoch.append(loss)
                pbar_v.set_description(f"Validation loss: {loss:.4f}")
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
        # save model performances
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
        
        # do a short LCT
        model.eval()
        print("Doing LCT")
        with torch.no_grad():
            train_loader = DataLoader(data_train[:10000], args.batch_size)
            test_loader = DataLoader(data_test[:10000], args.batch_size)
            tr_reps = []
            batch_size = args.batch_size
            for i, batch in enumerate(train_loader):
                batch = batch.to(args.device)
                tr_reps.append(
                    model(batch, return_rep=True)[0].detach().cpu().numpy()
                )
            tr_reps = np.concatenate(tr_reps)
            te_reps = []
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
        linear_n_epochs = 1000
        linear_learning_rate = 0.001
        linear_batch_size = 1000
        out_dat_f, out_lbs_f, losses_f, val_losses_f = linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, tr_reps, labels_train, te_reps, labels_test )
        auc, imtafe = get_perf_stats( out_lbs_f, out_dat_f )
        ep=0
        step_size = 200
        for j in range(len(losses_f[::step_size])):
            lss = losses_f[::step_size][j]
            val_lss = val_losses_f[::step_size][j]
            print( f"(rep layer {i}) epoch: " + str( ep ) + ", loss: " + str( lss ) + ", val loss: " + str( val_lss ), flush=True)
            ep+=step_size
        print( f"(rep layer {i}) auc: "+str( round(auc, 4) ), flush=True)
        print( f"(rep layer {i}) imtafe: "+str( round(imtafe, 1) ), flush=True)
        lct_auc_epochs.append(auc)
        np.save(f"{model_perf_loc}/vicreg_{label}_lct_auc_epochs.npy", np.array(lct_auc_epochs))
        if auc > lct_auc_best:
            print("New best LCT model")
            lct_auc_best = auc
            torch.save(model.state_dict(), f"{model_loc}/vicreg_{label}_lct_best.pth")
    #  training complete


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        action="store",
        default="/ssl-jet-vol-v2/JetClass/processed",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--percent-data",
        type=int,
        default=100,
        help="Percent of dataset to use for training, options: 1, 5, 10, 50, 100",
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
        "--shared",
        type=bool,
        action="store",
        default=True,
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
        default=2048,
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
