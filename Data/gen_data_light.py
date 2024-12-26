# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
import argparse
import os
import sys
# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from Data.get_config import BasicConfig
from Data.DataGenerator import DataGenerator

# ----------------------------- ARGUMENT CONFIG ----------------------------- #
parser = argparse.ArgumentParser()
# path config
parser.add_argument("--num_bs_ant", type=int, default=2, required=False)
parser.add_argument("--data_save_folder", type=str, default=root_path + "/Data/DataFiles/Eval_Data", required=False)
run_args = parser.parse_args()
print(run_args)


os.makedirs(run_args.data_save_folder,exist_ok=True)
link_config = BasicConfig(num_bs_ant=run_args.num_bs_ant)

# ----------------------- Generate data for evaluation ----------------------- #
link_config.set_channel_models(model_type = "TDL",PDP_mode="A", delay_spread=50e-9, min_speed=0,max_speed=0,delta_delay_spread=0)
generator = DataGenerator(init_config = link_config, # set 'add_interference = True' in this config 
                          ebNo_dB_range=np.linspace(3,6,4),
                          SIR_dB_range = np.linspace(100,100,1), # set the SIR to 100 dB, which means the interference can be ignored
                          apply_encoder = True,
                          add_interference = False)
realizations = generator.receive_data(batch_size=32, num_batch=200)
with open(run_args.data_save_folder + "/config%d_%s%s_ds%d_mobi%d.pkl" % (4,"TDL","A",50,0),'wb') as f:
    pickle.dump(realizations, f)


# # ---------------------------------------------------------------------------- #
# link_config.set_channel_models(model_type="TDL",PDP_mode="B", delay_spread=100e-9, min_speed=10,max_speed=10,delta_delay_spread=0)
# generator.reset_link_config(link_config)
# realizations = generator.receive_data(batch_size=32, num_batch=200)
# with open(run_args.data_save_folder + "/config%d_%s%s_ds%d_mobi%d.pkl" % (1,"TDL","B",100,10),'wb') as f:
#     pickle.dump(realizations, f)


# link_config.set_channel_models(model_type="TDL",PDP_mode="E", delay_spread=100e-9, min_speed=10,max_speed=10,delta_delay_spread=0)
# generator.reset_link_config(link_config)
# realizations = generator.receive_data(batch_size=32, num_batch=200)
# with open(run_args.data_save_folder + "/config%d_%s%s_ds%d_mobi%d.pkl" % (1,"TDL","E",100,10),'wb') as f:
#     pickle.dump(realizations, f)

# print("Finish generating data for evaluation.")