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
from Evaluation.eval_baseline import EvalBsRx
from Data.partition_data import get_all_client_data_paths, get_client_data_configs, get_all_client_data_paths_from_config
from Evaluation.eval_baseline import EvalBsRx



parser = argparse.ArgumentParser()
# path config
parser.add_argument("--num_bs_ant", type=int, default=2, required=False)
parser.add_argument("--data_save_folder", type=str, default=root_path + "/Data/DataFiles/NoneSIR_2Nr/OnlineData", required=False)

# config the FL method
parser.add_argument("--fl_method", type=str, default="fedgraph_new", required=False)
parser.add_argument("--seed", type=int, default=83, required=False)
parser.add_argument("--num_rounds", type=int, default=50, required=False)
parser.add_argument("--num_epochs", type=int, default=50, required=False)
# evaluation 
parser.add_argument("--min_ebNo", type=float, default=-4, required=False)
parser.add_argument("--max_ebNo", type=float, default=15, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=20, required=False)
parser.add_argument("--max_mc_iter", type=int, default=200, required=False)

parser.add_argument("--num_total_clients", type=int, default=6, required=False)
parser.add_argument("--eval_baseline", type=int, default=0, required=False)
parser.add_argument("--client_data_dist_type", type=int, default=5, required=False)
parser.add_argument("--test", type=str, default="general", required=False)
run_args = parser.parse_args()
print(run_args)



if __name__ == "__main__":
    import wandb
    wandb.login()

    # ----------------------------- set wandb config ----------------------------- #
    wandb_config = {**vars(run_args)} 
    run = wandb.init(project='Infocom2025', config=wandb_config, 
                        name = "OnlineEval_Dist{:d}_{}_{}".format(run_args.client_data_dist_type, run_args.test,run_args.fl_method))

    # set random seed
    tf.random.set_seed(run_args.seed)
    np.random.seed(run_args.seed)
    random.seed(run_args.seed)

    eval_result_save_folder = root_path + "/Evaluation/NewOnlineEvalResults/Dist{:d}/{}".format(run_args.client_data_dist_type,run_args.test)
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
    if run_args.fl_method == "fedavg":
        net_path = root_path + "/OnlineAdapt/FedAvgs/AdaptedModels_Dist{:d}".format(run_args.client_data_dist_type)
    elif run_args.fl_method == "ditto":
        net_path = root_path + "/OnlineAdapt/Dittos/AdaptedModels_Dist{:d}".format(run_args.client_data_dist_type)
    elif run_args.fl_method == "fedrep":
        net_path = root_path + "/OnlineAdapt/FedReps/AdaptedModels_Dist{:d}".format(run_args.client_data_dist_type)
    elif run_args.fl_method == "localonly":
        net_path = root_path + "/OnlineAdapt/LocalOnly/AdaptedModels_Dist{:d}".format(run_args.client_data_dist_type)
    elif run_args.fl_method == "fedavgFT":
        net_path = root_path + "/OnlineAdapt/FedAvgFT/AdaptedModels_Dist{:d}".format(run_args.client_data_dist_type)
    elif run_args.fl_method == "fedgraph_old":
        net_path = root_path + "/OnlineAdapt/FedGraphs/AdaptedModels_Dist{:d}_cosine".format(run_args.client_data_dist_type)
    elif run_args.fl_method == "fedgraph_new":
        if run_args.client_data_dist_type == 0:
            net_path = root_path + "/OnlineAdapt/FedGraphs/AdaptedModels_Dist0_l2_c0.60"
        if run_args.client_data_dist_type == 5:
            net_path = root_path + "/OnlineAdapt/FedGraphs/AdaptedModels_Dist5_l2_c0.20"
        # if run_args.client_data_dist_type == 4:
        #     net_path = root_path + "/OnlineAdapt/FedGraphs/AdaptedModels_Dist4_l2_c0.60"
    print(net_path)


    # ------------------- Get the data paths for the evaluation ------------------ #
    local_data_configs = get_client_data_configs(dist_type = run_args.client_data_dist_type, 
                                                 num_total_clients = run_args.num_total_clients)
    if run_args.test == "local":
        # use local data configs to generate eval data paths
        data_paths = get_all_client_data_paths_from_config(local_data_configs, run_args.data_save_folder)

    elif run_args.test == "general" and run_args.client_data_dist_type == 0:
        # use all the non-local data configs to generate eval data paths
        general_data_configs = {}
        for c_id in range(run_args.num_total_clients):
            config_list =  [local_data_configs[key] for key in local_data_configs.keys() if key != c_id]
            merged_config_list = list(itertools.chain(*config_list))
            general_data_configs[c_id] = list(set(merged_config_list))
            print(general_data_configs[c_id])
        data_paths = get_all_client_data_paths_from_config(general_data_configs, run_args.data_save_folder)

    elif run_args.test == "general" and run_args.client_data_dist_type == 5:
        # generate the general test data configs based on the collaboration graph
        # these non-local data configs should be similar to the local distributions!
        general_data_configs = {0: [0,9],
                                1: [10,11],
                                2: [0,1,2,9],
                                3: [0,1,2],
                                4: [12,15,16,17],
                                5: [12,13]}
        data_paths = get_all_client_data_paths_from_config(general_data_configs, run_args.data_save_folder)

    else:
        raise ValueError("Invalid evaluation experiments")


    # ------------------------------ Evaluation ------------------------------ #
    for client_id in range(run_args.num_total_clients):
        if run_args.fl_method == "none":
            break
        if run_args.fl_method == "fedavg":
            weight_path=Path(net_path).joinpath('Global','glob_para_round_%d.pkl' % (run_args.num_rounds-1))
            entire_main_net.set_weights(pickle.load(open(weight_path, 'rb')))
        elif run_args.fl_method == "localonly":
            weight_path=Path(net_path).joinpath('Client%d' % client_id,'local_para_epoch_%d.pkl' % (run_args.num_epochs-1))
            entire_main_net.set_weights(pickle.load(open(weight_path, 'rb')))
        else:
            weight_path=Path(net_path).joinpath('Client%d' % client_id,'local_para_round_%d.pkl' % (run_args.num_rounds-1))
            entire_main_net.set_weights(pickle.load(open(weight_path, 'rb')))
        
        # set the ber saving path
        result_save_path = [eval_result_save_folder + "/" + run_args.fl_method + "_ber_c{:d}.pkl".format(client_id),
                            eval_result_save_folder + "/" + run_args.fl_method + "_bler_c{:d}.pkl".format(client_id)]

        eval_obj = EvalEntireNet(entire_main_net=entire_main_net,
                                config = link_config,
                                ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                         run_args.max_ebNo,
                                                         run_args.num_ebNo_points),
                                result_save_path=result_save_path,
                                max_mc_iter=run_args.max_mc_iter,
                                eval_data_paths=data_paths[client_id])
        eval_obj.load_data_pkl() # load the evaluation data into self._batched_data
        ber, bler = eval_obj.eval()
        print("{:s}, Client {:d}, BER {}".format(run_args.fl_method, client_id, ber))


    # ---------------------------- Evaluate baselines ---------------------------- #
    if run_args.eval_baseline == 1:
        for client_id in range(run_args.num_total_clients):
            result_save_path = [eval_result_save_folder + "/lmmse_ber_c{:d}.pkl".format(client_id),
                                eval_result_save_folder + "/lmmse_bler_c{:d}.pkl".format(client_id)]

            bs1_eval_obj = EvalBsRx(perfect_csi=False,
                                    config = link_config,
                                    ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                            run_args.max_ebNo,
                                                            run_args.num_ebNo_points),
                                    result_save_path=result_save_path,
                                    max_mc_iter=run_args.max_mc_iter,
                                    eval_data_paths=data_paths[client_id])
            bs1_eval_obj.load_data_pkl() # load data into self._batched_data
            ber, bler = bs1_eval_obj.eval()
            print('LMMSE Detection, Client {:d}, BER {}'.format(client_id, ber))

            result_save_path = [eval_result_save_folder + "/ideal_ber_c{:d}.pkl".format(client_id),
                                eval_result_save_folder + "/ideal_bler_c{:d}.pkl".format(client_id)]
            bs2_eval_obj = EvalBsRx(perfect_csi=True,
                                    config = link_config,
                                    ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                            run_args.max_ebNo,
                                                            run_args.num_ebNo_points),
                                    result_save_path=result_save_path,
                                    max_mc_iter=run_args.max_mc_iter,
                                    eval_data_paths=data_paths[client_id])
            bs2_eval_obj.load_data_pkl() # load data into self._batched_data
            ber, bler = bs2_eval_obj.eval()
            print('Ideal Detection, Client {:d}, BER {}'.format(client_id, ber))

    # ---------------------------------------------------------------------------- #
    run.finish()
