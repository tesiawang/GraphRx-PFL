# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
import argparse
import os
import sys
import matplotlib.pyplot  as plt
import sionna as sn

# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from Data.get_config import BasicConfig
from Utils.models import SingleEntireMainNet
from Evaluation.eval_nn import EvalEntireNet
from Evaluation.eval_baseline import EvalBsRx

def build_model():
    entire_main_net = SingleEntireMainNet()
    batch_pilots_rg = np.zeros((1, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size, 1), dtype=np.complex64)
    batch_y = np.zeros((1, BasicConfig()._num_ofdm_symbols, BasicConfig()._fft_size, 1), dtype=np.complex64)
    batch_N0 = np.zeros(1, dtype=np.float32)
    entire_main_net([batch_pilots_rg, batch_y, batch_N0])
    return entire_main_net


def evaluate_ber(model: tf.keras.Model, 
                 weight_path: Path,
                 config: BasicConfig, 
                 config_id: int,
                 ebNo_dB_range: np.ndarray, 
                 eval_result_save_folder: str,
                 fl_method: str):
    
    with open(weight_path,'rb') as f:
        net_w = pickle.load(f)
    model.set_weights(weights=net_w)

    # if config_id == -1:
    #     eval_obj = EvalEntireNet(entire_main_net=model,
    #                          config = config, # channel should be set carefully
    #                          ebNo_dB_range=ebNo_dB_range,
    #                          result_save_path=str(Path(eval_result_save_folder).joinpath(fl_method + '_ber_new_config.pkl')))
    #     ber = eval_obj.eval()
    # elif config_id >= 0:
    
    eval_obj = EvalEntireNet(entire_main_net=model,
                            config = config, # channel should be set carefully
                            ebNo_dB_range=ebNo_dB_range,
                            result_save_path=str(Path(eval_result_save_folder).joinpath(fl_method + '_ber_config%d.pkl' % config_id)))
    ber = eval_obj.eval() 
    # else:
    #     raise ValueError('config_id should be -1 or non-negative: config_id =-1 -> avg_ber; config_id >= 0 -> config_id-th user')
    
    return ber


def load_ber_from_file(eval_result_save_folder: str, 
                       config_id: int,
                       fl_method: str):
    new_file_name = fl_method + '_ber_c%d.pkl' % config_id
    with open(Path(eval_result_save_folder).joinpath(new_file_name),'rb') as f:
        ber = pickle.load(f)
    return ber

