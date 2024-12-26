# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
import argparse
import os
import sys
import time
import shutil
import wandb
wandb.login()

# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = str(0)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from Utils.model_train import TrainEntireMainNet
from Utils.model_test import TestMainNet
from Data.partition_data import get_data_paths
from Data.get_config import BasicConfig, get_offline_config
from Utils.models import SingleEntireMainNet

parser = argparse.ArgumentParser()
# seed
parser.add_argument("--seed", type=int, default=42, required=False)
# training 
parser.add_argument("--num_epochs", type=int, default=3, required=False)
parser.add_argument("--lr", type=float, default=1e-3, required=False)
# other para
parser.add_argument("--memory_limit", type=float, default=20, required=False)
parser.add_argument("--save_model_every", type=int, default=1, required=False)
parser.add_argument("--train_load_ratio", type=float, default=0.9, required=False)
parser.add_argument("--SIR_pattern", type=str, default="NoneSIR", required=False)
parser.add_argument("--interleave", type=bool, default=False, required=False)
# parser.add_argument("--data_save_folder", type=str, default=root_path + "/Data/DataFiles/NoneSIR_2Nr/AugData", required=False)
run_args = parser.parse_args()
print(run_args)



if __name__=='__main__':
    
    # --------------------------------- Set seeds -------------------------------- #
    tf.keras.utils.set_random_seed(run_args.seed)
    # tf.config.experimental.enable_op_determinism()
    np.random.seed(run_args.seed)
    _, cfg = get_offline_config()


    # --------------------------- Set config for wandb --------------------------- #
    wandb_config = {**vars(run_args),**cfg} # one-line code 
    run = wandb.init(project='Infocom2025', group = "pretrain", 
                     name = 'pretrain with augmented data', config=wandb_config)
    

    # ------------------------------ Refresh folder ------------------------------ #
    data_save_folder = root_path + "/Data/DataFiles/" + run_args.SIR_pattern + "_" + str(cfg['num_bs_ant']) + "Nr" + "/OfflineData"
    # model_save_folder = root_path + "/OfflinePretrain/Aug_PretrainedModels_" + run_args.SIR_pattern + "_" + str(cfg['num_bs_ant'])+ "Nr"
    # metric_save_folder = root_path + "/OfflinePretrain/Aug_TrainMetrics_" + run_args.SIR_pattern + "_" + str(cfg['num_bs_ant'])+ "Nr"

    # for folder in [model_save_folder, metric_save_folder]:
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #     else:
    #         shutil.rmtree(folder)
    #         os.makedirs(folder)
    
    # -------------------------------- Build Model ------------------------------- #
    link_config = BasicConfig(num_bs_ant=cfg['num_bs_ant'],
                              fft_size=cfg['fft_size'],
                              num_bits_per_symbol= cfg['num_bits_per_symbol'],
                              pilot_ofdm_symbol_indices=cfg['pilot_ofdm_symbol_indices'])
    
    # create a single end-to-end network (instead of the cascaded model of channel estimator and demapper)
    entire_main_net = SingleEntireMainNet()
    x_ = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_ut_ant), dtype=np.complex64)
    y_ = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_bs_ant), dtype=np.complex64)
    n0_ = np.zeros(1, dtype=np.float32)

    # build the model with several batches of data
    entire_main_net([x_, y_, n0_]) # this line determines the number of channels of the _input_conv layer!!!
    print("entire_main_net is built successfully!")


    # ------------------------------ Set data paths ------------------------------ #
    _,_,num_points = cfg['snr_range'].values()
    config_idx = list(np.linspace(0, num_points-1, num_points, dtype=int))
    data_paths = get_data_paths(data_save_folder, config_idx)
    print(data_paths)


    train_obj = TrainEntireMainNet(entire_main_net=entire_main_net,
                                    lr=run_args.lr,
                                    train_data_paths = data_paths,
                                    load_ratio=0.3,
                                    config = link_config)
    
    test_obj = TestMainNet(entire_main_net=entire_main_net,
                           test_data_paths=data_paths, 
                           load_ratio=0.1,
                           config = link_config)

    train_loss_list = []
    test_loss_list = []
    test_rate_list = []

    batched_train_data = train_obj.load_data_pkl()
    batched_test_data = test_obj.load_data_pkl()

    # Train and test the model epoch by epoch:
    for e_id in range(run_args.num_epochs):
        
        # ---------------------------- Train for one epoch --------------------------- #
        start_time = time.time()
        epoch_trained_parameters, train_loss = train_obj.train(num_epochs=1, batched_data=batched_train_data)
        
        # ---------------------------- Test at the current epoch ---------------------------- #            
        # set the model weights as the current local model weights
        test_obj.set_model_weights(epoch_trained_parameters)
        test_loss, test_rate = test_obj.test(batched_test_data)

        # train_loss_list.append(train_loss)
        # test_loss_list.append(test_loss)
        # test_rate_list.append(test_rate)
        end_time = time.time()
        wandb.log({'epochs': e_id,
                   'epoch_train_loss': train_loss,
                   'epoch_test_loss': test_loss,
                   'epoch_test_rate': test_rate})
        
        # ---------------------------------------------------------------------------- #
        print("=====> Pretraining Epoch {:d}, Training Time: {:2f}, Train Loss: {:.3f}, Test loss: {:.3f}, Test BMD Rate: {:.2f}".format(
              e_id, end_time-start_time, train_loss, test_loss, test_rate))
            
        # save models for every _ epochs
        # if (e_id+1)% run_args.save_model_every == 0 and e_id > 0:
        #     with open(Path(model_save_folder).joinpath('updated_parameters_%d.pkl' % e_id), 'wb') as f:
        #         pickle.dump(epoch_trained_parameters, f)

    run.finish()

    # ---------------- save training metrics after all the epochs: --------------- #
    ## OfflinePretrain/TrainMetrics/all_epoch_train_loss.pkl
    # with open(Path(metric_save_folder).joinpath('all_epoch_train_loss.pkl'), 'wb') as f:
    #     pickle.dump(train_loss_list,f)

    # with open(Path(metric_save_folder).joinpath('all_epoch_test_loss.pkl'), 'wb') as f:
    #     pickle.dump(test_loss_list,f)

    # with open(Path(metric_save_folder).joinpath('all_epoch_test_rate.pkl'), 'wb') as f:
    #     pickle.dump(test_rate_list,f)

