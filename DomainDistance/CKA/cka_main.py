
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import List, Dict
import os
import sys
# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = str(0)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from cka_model import SingleEntireMainNet
from cka_class import CKA

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from Data.partition_data import get_all_client_data_paths, get_data_paths

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, required=False)
parser.add_argument("--model1_client_id", type=int, default=1, required=False)
parser.add_argument("--model2_client_id", type=int, default=2, required=False)
parser.add_argument("--dataset1_cfg", type=int, default=0, required=False) # simple TDLB
parser.add_argument("--dataset2_cfg", type=int, default=6, required=False) # simple TDLC
parser.add_argument("--compare_model", type=int, default=0, required=False) # 0: compare pretrained model on two datasets, 1: compare two clients' models on the same dataset
run_args = parser.parse_args()
print(run_args)


if __name__=='__main__':
    
    # initialize two models
    model1 = SingleEntireMainNet(num_bits_per_symbol=4,
                                 input_pilot=True)
    model2 = SingleEntireMainNet(num_bits_per_symbol=4,
                                 input_pilot=True)
    
    x_ = np.zeros((1, 14, 72, 1), dtype=np.complex64)
    y_ = np.zeros((1, 14, 72, 2), dtype=np.complex64)
    n0_ = np.zeros(1, dtype=np.float32)
    model1([x_,y_,n0_])
    model2([x_,y_,n0_])
    print("Model1 and Model2 are built successfully with pilot symbols as inputs")

    # ---------------------------------------------------------------------------- #
    if run_args.compare_model == 1:
        # load model weights from two trained models
        model_weight_path1 = root_path + "/OnlineAdapt/LocalOnly/AdaptedModels/Client{:d}/local_para_epoch_19.pkl".format(run_args.model1_client_id)
        model_weight_path2 = root_path + "/OnlineAdapt/LocalOnly/AdaptedModels/Client{:d}/local_para_epoch_19.pkl".format(run_args.model2_client_id)
    else:
        # load pretrained models
        model_weight_path1 = root_path+"/OfflinePretrain/PretrainedModels_NoneSIR_2Nr/updated_parameters_19.pkl"
        model_weight_path2 = root_path+"/OfflinePretrain/PretrainedModels_NoneSIR_2Nr/updated_parameters_19.pkl"
    # ---------------------------------------------------------------------------- #

    with open(model_weight_path1, 'rb') as f:
        model1_weights = pickle.load(f)
    with open(model_weight_path2, 'rb') as f:
        model2_weights = pickle.load(f)
    model1.set_weights(model1_weights)
    model2.set_weights(model2_weights)

    # ------------------------------ get data paths ------------------------------ #

    if run_args.compare_model == 1:
        # get the common offline evaluation dataset
        data_save_folder = root_path+"/Data/DataFiles/Eval_CKA/TDLC"
        eval_data_path1 = get_data_paths(data_save_folder, [5,6,7,8,9])
        eval_data_path2 = get_data_paths(data_save_folder, [5,6,7,8,9])
    else: 
        # get two different online data configs
        # test the pretrained model on two datasets
        data_save_folder = root_path+"/Data/DataFiles/NoneSIR_2Nr/OnlineData"
        eval_data_path1 = get_data_paths(data_save_folder, [run_args.dataset1_cfg])
        eval_data_path2 = get_data_paths(data_save_folder, [run_args.dataset2_cfg])
    # ---------------------------------------------------------------------------- #


    # create CKA object
    cka_obj = CKA(model1, model2, model1_layers = list(range(0,13)), model2_layers = list(range(0,13)))
    hsic_matrix = cka_obj.compare(eval_data_path1,eval_data_path2)

    if run_args.compare_model == 1:
        # test two clients' models on the common dataset
        cka_obj.plot_results(save_path=root_path + "/CKA_Compare_C{:d}C{:d}.png".format(run_args.model1_client_id, run_args.model2_client_id))
    else:
        # test the pretrained model on two datasets
        cka_obj.plot_results(save_path=root_path + "/CKA_Compare_Pretrained_on_D{}D{}.png".format(run_args.dataset1_cfg, run_args.dataset2_cfg))

