import torch
import numpy as np
import os
import click

from utils.preprocessing import data_preprocessing
from train_folder.trainer import training
from inference_folder.inferencer import inferencing

from torch.utils.tensorboard import SummaryWriter


@click.command()
@click.option("-fault_id", "--fault_id", default = "01")

def main(fault_id):
    # define config
    config = {
        "epoch": 10000,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "lambda_cycle": 2,
        "lambda_identity": 0,
        "num_workers": 2,
        "dim_z": 16,
        "checkpoint": "checkpoint",
        "window_size": 80,
        "wavelet_level": 2,
        "save_step": 2000, # 2000
        "generated_data": "generated_data",
    }

    # ---------------------------------- setup & data preprocessing ---------------------------------------------

    # GPU usage
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    tep_normal = np.genfromtxt("tep_data/d00_te.dat")
    tep_fault = np.genfromtxt(f"tep_data/d{fault_id}_te.dat")

    data_normal = tep_normal[200:600]  # (sample_size, 52)
    data_fault = tep_fault[200:600]  # (sample_size, 52)

    # data preprocessing includes: white noise filtering, sliding window, wavelet, fre normalization
    preprocess_result = data_preprocessing(data_normal, data_fault, config)
    data = {"normal": preprocess_result['fre_faulty_norm'], "faulty": preprocess_result['fre_normal_norm']}

    # update config
    config["sample_size"] = preprocess_result['fre_faulty_norm'].shape[0]
    config["var_num"] = preprocess_result['fre_faulty_norm'].shape[1]
    config["seq_len"] = preprocess_result['fre_faulty_norm'].shape[2]

    # save location setting
    os.makedirs(config["checkpoint"], exist_ok=True)
    config["checkpoint"] = (config["checkpoint"] + f"/fault_{fault_id}")
    os.makedirs(config["checkpoint"], exist_ok=True)

    os.makedirs(config["generated_data"], exist_ok=True)
    config["generated_data"] = (config["generated_data"] + f"/fault_{fault_id}")
    os.makedirs(config["generated_data"], exist_ok=True)

    # basic setting
    writer = SummaryWriter("./logs")
    torch.cuda.manual_seed_all(42)

    # ---------------------------------- training % inference ---------------------------------------------

    """ train """
    training(config, data, device, writer)


if __name__ == "__main__":
    main()
