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

def modify_path(path):
    """
    Given a path for a data file, e.g. path = '/ssl-jet-vol-v2/JetClass/processed/val/data/HToBB_123.pt',
    Constructs the path for the corresponding label file, e.g. new_path = '/ssl-jet-vol-v2/JetClass/processed/val/label/labels_HToBB_123.pt'
    """
    # Split the string into parts
    parts = path.split('/')
    
    # Replace 'data' with 'label' in the second-to-last part
    parts[-2] = 'label'
    
    # Insert 'labels_' before the last part
    parts[-1] = 'labels_' + parts[-1]
    
    # Join the parts back together
    new_path = '/'.join(parts)
    return new_path

def main(args):
    """
    Samples a fraction of jets from all data files and saves them to a new directory.
    Shape: (100k, 7, 128)
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    
    label = args.label
    data_dir = f"/ssl-jet-vol-v2/JetClass/processed/{label}"
    data_files = glob.glob(f"{data_dir}/data/*")
    frac_lst = [10, 50]
    
    for frac in frac_lst:
        print(f"Sampling {frac}% of data from `{label}` directory")
        processed_data_dir = f"/ssl-jet-vol-v2/JetClass/processed/{label}_{frac}%/data"
        processed_label_dir = f"/ssl-jet-vol-v2/JetClass/processed/{label}_{frac}%/label"
        os.system(f"mkdir -p {processed_data_dir} {processed_label_dir}")  # -p: create parent dirs if needed, exist_ok

        # Compute the total number of samples
        
        if label == "train":
            total_samples = int(frac / 100 * 100000000) # 100M
        elif label == "test":
            total_samples = int(frac / 100 * 20000000) # 20M 
        elif label == "val":
            total_samples = int(frac / 100 * 5000000) # 5M

        # Pre-allocate memory
        all_sampled_data = torch.empty((total_samples, 7, 128))  # adjust dimensions accordingly
        all_sampled_labels = torch.empty(total_samples)  # adjust dimensions if needed

        offset = 0  # keeps track of where to insert samples in the pre-allocated tensor
        for i, file in enumerate(data_files):
            data = torch.load(file)
            data_file_name = file.split("/")[-1].split(".")[0]
            print(f"--- loaded data file {i} {data_file_name} from `{label}` directory")

            label_path = modify_path(file)
            labels = torch.load(label_path)
            labels_file_name = label_path.split("/")[-1].split(".")[0]
            print(f"--- loaded data file {i} {labels_file_name} from `{label}` directory")

            # Calculate the number of samples you need
            num_samples = int(frac/100 * data.shape[0])

            # Generate random indices
            indices = torch.randperm(data.shape[0])[:num_samples]

            sampled_data = data[indices]
            sampled_labels = labels[indices]

            # Fill the pre-allocated tensor
            all_sampled_data[offset:offset+num_samples] = sampled_data
            all_sampled_labels[offset:offset+num_samples] = sampled_labels

            offset += num_samples

            # Free up memory
            del data, labels, sampled_data, sampled_labels
            torch.cuda.empty_cache()

        torch.save(all_sampled_data, osp.join(processed_data_dir, "data.pt"))
        torch.save(all_sampled_labels, osp.join(processed_label_dir, "labels.pt"))
        
        del all_sampled_data, all_sampled_labels
        torch.cuda.empty_cache()
        print("finished sampling and saving data")



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