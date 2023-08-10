# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import argparse
import awkward as ak
import numpy as np
import pandas as pd
import vector
import mplhep as hep
import torch
import os
import os.path as osp


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def zero_pad(arr, max_nconstit=50):
    """
    arr: torch tensor
    """
    arr = arr[:max_nconstit]
    if arr.shape[0] < max_nconstit:
        zeros = torch.zeros(max_nconstit - arr.shape[0], 1)
        padded_arr = torch.cat([arr, zeros], axis=0)
        return padded_arr
    else:
        return arr


def main(args):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    Convert h5 to pt files, each containing 100k zero-padded jets cropped to 50 constituents
    Shape: (100k, 3, 50)
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    label = args.label
    hdf5_file = f"/ssl-jet-vol-v2/toptagging/{label}/raw/{label}.h5"
    vector.register_awkward()

    df = pd.read_hdf(hdf5_file, key="table")

    def _col_list(prefix, max_particles=200):
        return ["%s_%d" % (prefix, i) for i in range(max_particles)]

    _px = df[_col_list("PX")].values
    _py = df[_col_list("PY")].values
    _pz = df[_col_list("PZ")].values
    _e = df[_col_list("E")].values

    mask = _e > 0
    n_particles = np.sum(mask, axis=1)

    px = ak.unflatten(_px[mask], n_particles)
    py = ak.unflatten(_py[mask], n_particles)
    pz = ak.unflatten(_pz[mask], n_particles)
    energy = ak.unflatten(_e[mask], n_particles)

    p4 = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "energy": energy,
        },
        with_name="Momentum4D",
    )

    jet_p4 = ak.sum(p4, axis=-1)

    # outputs
    v = {}
    v["label"] = df["is_signal_new"].values

    v["jet_pt"] = jet_p4.pt.to_numpy()
    v["jet_eta"] = jet_p4.eta.to_numpy()
    v["jet_phi"] = jet_p4.phi.to_numpy()
    v["jet_energy"] = jet_p4.energy.to_numpy()
    v["jet_mass"] = jet_p4.mass.to_numpy()
    v["jet_nparticles"] = n_particles

    v["part_px"] = px
    v["part_py"] = py
    v["part_pz"] = pz
    v["part_energy"] = energy

    v["part_deta"] = p4.deltaeta(jet_p4)
    v["part_dphi"] = p4.deltaphi(jet_p4)

    part_pt = np.hypot(v["part_px"], v["part_py"])

    features = []
    labels = []
    c = 0
    processed_dir = f"/ssl-jet-vol-v2/toptagging/{label}/processed/3_features"
    os.system(f"mkdir -p {processed_dir}")
    for jet_index in range(len(df)):
        pt = zero_pad(torch.from_numpy(np.array(part_pt[jet_index]).reshape(-1, 1)))
        deta = zero_pad(
            torch.from_numpy(np.array(v["part_deta"][jet_index]).reshape(-1, 1))
        )
        dphi = zero_pad(
            torch.from_numpy(np.array(v["part_dphi"][jet_index]).reshape(-1, 1))
        )

        jet = torch.cat([pt, deta, dphi], axis=1).transpose(0, 1)
        y = torch.tensor(v["label"][jet_index]).long()

        features.append(jet)
        labels.append(y)

        if jet_index % 100000 == 0 and jet_index != 0:
            print(f"saving datafile data_{c}")
            torch.save(torch.stack(features), osp.join(processed_dir, f"data_{c}.pt"))
            torch.save(torch.stack(labels), osp.join(processed_dir, f"labels_{c}.pt"))
            c += 1
            features = []
            labels = []


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
