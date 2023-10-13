# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import argparse
import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
import torch
import os
import os.path as osp
import glob
import os



def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x

def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))

def build_features_and_labels(tree, transform_features=True):
    # load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])

    # compute new features
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])

    # apply standardization
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0

    feature_list = {
        'pf_features': [
            'part_deta',
            'part_dphi',
            'part_pt_log', 
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
        ],
        'pf_mask': ['part_mask']
    }

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)

    label_list = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)
    
    return out

def main(args):
    """Runs data processing scripts to turn raw data from (/ssl-jet-vol-v2/JetClass/Pythia/) into
    cleaned data ready to be analyzed (saved in /ssl-jet-vol-v2/JetClass/processed).
    Convert root to pt files, each containing 1M zero-padded jets cropped to 128 constituents
    Only contains kinematic features
    Shape: (100k, 7, 128)
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    label = args.label
    if label == "train":
        label += "_100M"
    elif label == "val":
        label += "_5M"
    elif label == "test":
        label += "_20M"
    data_dir = f"/ssl-jet-vol-v2/JetClass/Pythia/{label}"
    data_files = glob.glob(f"{data_dir}/*")
    label_orig = label.split("_")[0] # without _100M, _5M, _20M
    processed_dir = f"/ssl-jet-vol-v2/JetClass/processed/{label_orig}"
    os.system(f"mkdir -p {processed_dir}")  # -p: create parent dirs if needed, exist_ok

    for i, file in enumerate(data_files):
        tree = uproot.open(file)['tree']
        file_name = file.split("/")[-1].split(".")[0]
        print(f"--- loaded data file {i} {file_name} from `{label}` directory")
        f_dict = build_features_and_labels(tree)
        features_tensor = torch.from_numpy(f_dict['pf_features'])
        labels_tensor = torch.from_numpy(f_dict['label'])
        torch.save(features_tensor, osp.join(processed_dir, f"{file_name}.pt"))
        torch.save(labels_tensor, osp.join(processed_dir, f"labels_{file_name}.pt"))
        print(f"--- saved data file {i} {file_name} to `{processed_dir}` directory")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        type=str,
        action="store",
        default="train",
        help="train/val/test",
    )
    args = parser.parse_args()
    main(args)
