# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
import argparse
import os
import sys
import matplotlib.pyplot  as plt
# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from Data.get_config import BasicConfig
from Data.DataGenerator import DataGenerator
from Utils.models import SingleEntireMainNet
from Evaluation.eval_nn import EvalEntireNet
from Evaluation.eval_baseline import EvalBsRx
from Data.partition_data import get_data_paths

# ----------------------------- ARGUMENT CONFIG ----------------------------- #
parser = argparse.ArgumentParser()
# path config
parser.add_argument("--num_bs_ant", type=int, default=2, required=False)
parser.add_argument("--data_save_folder", type=str, default=root_path + "/Data/DataFiles/NoneSIR_2Nr/OfflineData", required=False)
parser.add_argument("--pretrained_entire_main_net_path", type=str, default=root_path + "/OfflinePretrain/PretrainedModels_NoneSIR_2Nr/updated_parameters_49.pkl", required=False)
###
parser.add_argument("--eval_result_save_folder", type=str, default=root_path + "/Evaluation/OfflineEvalResults", required=False)
parser.add_argument("--config_id", type=int, default=10, required=False) # TODO: why?? this config may have bias; maybe we need to shuffle the data
# evaluation config
parser.add_argument("--eval_load_ratio", type=float, default=1, required=False)
parser.add_argument("--min_ebNo", type=float, default=-4, required=False) 
parser.add_argument("--max_ebNo", type=float, default=15, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=20, required=False)
parser.add_argument("--max_mc_iter", type=int, default=200, required=False)

run_args = parser.parse_args()
print(run_args)

# new_folder = run_args.eval_result_save_folder + "/with_config%d" % run_args.config_id
new_folder = run_args.eval_result_save_folder
os.makedirs(new_folder,exist_ok=True)


# newly defined config
link_config = BasicConfig(num_bs_ant=run_args.num_bs_ant)
entire_main_net = SingleEntireMainNet()
batch_pilots_rg = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_ut_ant), dtype=np.complex64)
batch_y = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_bs_ant), dtype=np.complex64)
batch_N0 = np.zeros(1, dtype=np.float32)
entire_main_net([batch_pilots_rg, batch_y, batch_N0])


# load the pretrained weights
with open(run_args.pretrained_entire_main_net_path, 'rb') as f:
    entire_weights = pickle.load(f)
entire_main_net.set_weights(weights=entire_weights)


# # ----------------------- Generate data for evaluation ----------------------- #
# link_config.set_channel_models(model_type = "TDL",PDP_mode="A", delay_spread=50e-9, min_speed=0,max_speed=1.5,delta_delay_spread=0)
# generator = DataGenerator(init_config = link_config, # set 'add_interference = True' in this config 
#                           ebNo_dB_range=np.linspace(-4,10,15),
#                           SIR_dB_range = np.linspace(100,100,1), # set the SIR to 100 dB, which means the interference can be ignored
#                           apply_encoder = True,
#                           add_interference = False)
# realizations = generator.receive_data(batch_size=32, num_batch=200)
# with open(run_args.data_save_folder + "/config%d_%s%s_ds%d_mobi%d.pkl" % (0,"TDL","A",50,1),'wb') as f:
#     pickle.dump(realizations, f)

# link_config.set_channel_models(model_type="TDL",PDP_mode="C", delay_spread=100e-9, min_speed=10,max_speed=10,delta_delay_spread=0)
# generator.reset_link_config(link_config)
# realizations = generator.receive_data(batch_size=32, num_batch=200)
# with open(run_args.data_save_folder + "/config%d_%s%s_ds%d_mobi%d.pkl" % (1,"TDL","C",100,10),'wb') as f:
#     pickle.dump(realizations, f)


# ----------------------- Load data for evaluation ----------------------- #
# TODO: the evaluation data set-up is very important!!!
data_paths = get_data_paths(run_args.data_save_folder, [run_args.config_id])
print(data_paths)


# ------------------------------ Evaluation ------------------------------ #
eval_obj = EvalEntireNet(entire_main_net=entire_main_net,
                         config = link_config,
                         ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                run_args.max_ebNo,
                                                run_args.num_ebNo_points),
                         result_save_path=[str(Path(new_folder).joinpath('nnrx_ber.pkl')),
                                           str(Path(new_folder).joinpath('nnrx_bler.pkl'))],
                         max_mc_iter=run_args.max_mc_iter,
                         eval_data_paths=data_paths,
                         load_ratio=run_args.eval_load_ratio)
eval_obj.load_data_pkl() # load data into self._batched_data
nnrx_ber, nnrx_bler = eval_obj.eval()


# ---------------------------------------------------------------------------- #
bs1_eval_obj = EvalBsRx(perfect_csi=False,
                        config = link_config,
                        ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                  run_args.max_ebNo,
                                                  run_args.num_ebNo_points),
                        result_save_path=[str(Path(run_args.eval_result_save_folder).joinpath('lmmse_ber.pkl')),
                                           str(Path(run_args.eval_result_save_folder).joinpath('lmmse_bler.pkl'))],
                        max_mc_iter=run_args.max_mc_iter,
                        eval_data_paths=data_paths,
                        load_ratio=run_args.eval_load_ratio)
bs1_eval_obj.load_data_pkl() # load data into self._batched_data
lmmse_ber, lmmse_bler = bs1_eval_obj.eval()


# ---------------------------------------------------------------------------- #
bs2_eval_obj = EvalBsRx(perfect_csi=True,
                        config = link_config,
                        ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                  run_args.max_ebNo,
                                                  run_args.num_ebNo_points),
                        result_save_path=[str(Path(new_folder).joinpath('ideal_ber.pkl')),
                                           str(Path(new_folder).joinpath('ideal_bler.pkl'))],
                        max_mc_iter=run_args.max_mc_iter,
                        eval_data_paths=data_paths,
                        load_ratio=run_args.eval_load_ratio)
bs2_eval_obj.load_data_pkl() # load data into self._batched_data
ideal_ber, ideal_bler = bs2_eval_obj.eval()


# print the results
print("nnrx_ber: ", nnrx_ber)
print("nnrx_bler: ", nnrx_bler)
print("lmmse_ber: ", lmmse_ber)
print("lmmse_bler: ", lmmse_bler)
print("ideal_ber: ", ideal_ber)
print("ideal_bler: ", ideal_bler)