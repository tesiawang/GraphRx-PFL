# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
import argparse
import os
import sys
import matplotlib.pyplot  as plt
import random
import itertools
# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import sionna as sn

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from Data.get_config import BasicConfig
from Utils.models import SingleEntireMainNet
from Evaluation.eval_nn import EvalEntireNet
from Data.partition_data import get_all_client_data_paths, get_client_data_configs, get_all_client_data_paths_from_config


parser = argparse.ArgumentParser()
# path config
parser.add_argument("--seed", type=int, default=83, required=False)
parser.add_argument("--num_bs_ant", type=int, default=2, required=False)
parser.add_argument("--data_save_folder", type=str, default=root_path + "/Data/DataFiles/NoneSIR_2Nr/OnlineData", required=False)

# config the FL method
parser.add_argument("--dist_metric", type=str, default="cosine", required=False)
parser.add_argument("--client_data_dist_type", type=int, default=5, required=False)
parser.add_argument("--num_rounds", type=int, default=10, required=False)   

# evaluation 
parser.add_argument("--min_ebNo", type=float, default=-4, required=False)
parser.add_argument("--max_ebNo", type=float, default=-3, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=2, required=False)
parser.add_argument("--max_mc_iter", type=int, default=10, required=False)

parser.add_argument("--num_total_clients", type=int, default=6, required=False)
parser.add_argument("--eval_baseline", type=int, default=0, required=False)
parser.add_argument("--test", type=str, default="local", required=False)
run_args = parser.parse_args()
print(run_args)



if __name__ == "__main__":
    import wandb
    wandb.login()

    # ----------------------------- set wandb config ----------------------------- #
    wandb_config = {**vars(run_args)} 
    run = wandb.init(project='Infocom2025', config=wandb_config, 
                        name="AblationEval_Dist{:d}_{:s}".format(run_args.client_data_dist_type,run_args.dist_metric))

    # set random seed
    tf.random.set_seed(run_args.seed)
    np.random.seed(run_args.seed)
    random.seed(run_args.seed)

    eval_result_save_folder = root_path + "/Evaluation/Ablation/Dist{:d}".format(run_args.client_data_dist_type)
    os.makedirs(eval_result_save_folder,exist_ok=True)
    link_config = BasicConfig(num_bs_ant=run_args.num_bs_ant)

    # build model
    entire_main_net = SingleEntireMainNet()
    batch_pilots_rg = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_ut_ant), dtype=np.complex64)
    batch_y = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_bs_ant), dtype=np.complex64)
    batch_N0 = np.zeros(1, dtype=np.float32)
    entire_main_net([batch_pilots_rg, batch_y, batch_N0])


    # set neural network weights path
    net_path = ""
    if run_args.dist_metric == "cosine":
        net_path = root_path + "/OnlineAdapt/FedGraphs/HighSIR/AdaptedModels_Dist{:d}_cosine".format(run_args.client_data_dist_type) # the best c has been found
    elif run_args.dist_metric == "l1":
        net_path = root_path + "/OnlineAdapt/FedGraphs/HighSIR/AdaptedModels_Dist{:d}_l1".format(run_args.client_data_dist_type) # the best c has been found
    else:
        raise ValueError("Invalid distance metric")
    print(net_path)


    # ------------------- Get the data paths for the evaluation ------------------ #
    local_data_configs = get_client_data_configs(dist_type = run_args.client_data_dist_type, 
                                                 num_total_clients = run_args.num_total_clients)
    if run_args.test == "local":
        # use local data configs to generate eval data paths
        data_paths = get_all_client_data_paths_from_config(local_data_configs, run_args.data_save_folder)
    else:
        raise ValueError("Invalid evaluation experiments")


    # ------------------------------ Evaluation ------------------------------ #
    for client_id in range(run_args.num_total_clients):
        # load the weights into neural receiver 
        weight_path=Path(net_path).joinpath('Client%d' % client_id,'local_para_round_%d.pkl' % (run_args.num_rounds-1))
        entire_main_net.set_weights(pickle.load(open(weight_path, 'rb')))
        
        # set the ber saving path only for cosine and l1
        result_save_path = [eval_result_save_folder + "/" + "{:s}_bs{:d}.pkl".format(run_args.dist_metric,client_id)]

        eval_obj = EvalEntireNet(entire_main_net=entire_main_net,
                                config = link_config,
                                ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                         run_args.max_ebNo,
                                                         run_args.num_ebNo_points),
                                result_save_path=result_save_path,
                                max_mc_iter=run_args.max_mc_iter,
                                eval_data_paths=data_paths[client_id])
        eval_obj.load_data_pkl() # load the evaluation data into self._batched_data
        ber,bler= eval_obj.eval()

        print("Dist_Metric {:s}, Client {:d}, BER {}".format(run_args.dist_metric, client_id, ber))


    # ---------------------------------------------------------------------------- #
    run.finish()
