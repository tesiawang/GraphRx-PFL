# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
import argparse
import os
os.environ['PYTHONHASHSEED'] = str(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import shutil
import time
# turn off the device INFO messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# set root path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)
from Data.get_config import BasicConfig, get_online_config
from Utils.models import SingleEntireMainNet
from Data.partition_data import get_all_client_data_paths, get_data_paths
from Utils.functions import log_print
from Utils.model_train import TrainEntireMainNet
from Utils.model_test import TestMainNet
from Utils.coreset import CoresetSelector

# --------------------------------- Arguments --------------------------------- #
parser = argparse.ArgumentParser()
# seed
parser.add_argument("--seed", type=int, default=42, required=False)
# key mechansims
parser.add_argument("--use_coreset", type=int, default=0, required=False)
parser.add_argument("--buffer_size", type=int, default=0, required=False)
parser.add_argument("--apply_aug", type=int, default=1, required=False)
parser.add_argument("--aug_times", type=int, default=8, required=False)
# training
parser.add_argument("--num_epochs", type=int, default=3, required=False) # train one client locally for 30 epochs
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--num_total_clients", type=int, default=6, required=False)
parser.add_argument("--train_load_ratio", type=float, default=1, required=False)
parser.add_argument("--test_load_ratio", type=float, default=0.2, required=False)
parser.add_argument("--save_model_every", type=int, default=1, required=False)
parser.add_argument("--client_data_dist_type", type=int, default=0, required=False)
parser.add_argument("--load_pretrained", type=int, default=0, required=False)
parser.add_argument("--SIR_pattern", type=str, default='NoneSIR', required=False)
parser.add_argument("--debug", type=int, default=1, required=False)


run_args = parser.parse_args()
print(run_args)

if run_args.debug == 0:
    import wandb
    wandb.login()

if __name__=='__main__':

    tf.keras.utils.set_random_seed(run_args.seed)
    np.random.seed(run_args.seed)
    _,cfg = get_online_config()

    if run_args.debug == 0:
        wandb_config = {**vars(run_args),**cfg} # one-line code 
        run = wandb.init(project='Infocom2025', config=wandb_config, name='LocalOnly_dist{:d}'.format(run_args.client_data_dist_type))
        
    pretrained_entire_main_net_path = root_path + "/OfflinePretrain/PretrainedModels_" + run_args.SIR_pattern +"_2Nr/updated_parameters_39.pkl"
    data_save_folder = root_path + "/Data/DataFiles/"+ run_args.SIR_pattern +"_2Nr/OnlineData"

    model_save_folder = root_path + "/OnlineAdapt/LocalOnly/"+ run_args.SIR_pattern + "/AdaptedModels_Dist{:d}_Aug{:d}_CS{:d}".format(run_args.client_data_dist_type, run_args.aug_times, run_args.use_coreset)
    metric_save_folder = root_path + "/OnlineAdapt/LocalOnly/"+ run_args.SIR_pattern + "/TrainMetrics_Dist{:d}_Aug{:d}_CS{:d}".format(run_args.client_data_dist_type, run_args.aug_times, run_args.use_coreset)
    for folder in [model_save_folder, metric_save_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

    # ------------------------------ build the model ----------------------------- #
    link_config = BasicConfig(num_bs_ant=cfg['num_bs_ant'], 
                              fft_size=cfg['fft_size'], 
                              num_bits_per_symbol=cfg['num_bits_per_symbol'],
                              pilot_ofdm_symbol_indices=cfg['pilot_ofdm_symbol_indices'])
    entire_main_net = SingleEntireMainNet()
    x_ = np.zeros((10,link_config._num_ofdm_symbols,link_config._fft_size,link_config._num_ut_ant), dtype=np.complex64)
    y_ = np.zeros((10,link_config._num_ofdm_symbols,link_config._fft_size,link_config._num_bs_ant), dtype=np.complex64)
    n0_ = np.zeros(10, dtype=np.float32)
    entire_main_net([x_, y_, n0_]) # this line determines the number of changit nels of the _input_conv layer!!!

    # --------------------------- load pretrained model -------------------------- #
    if run_args.load_pretrained == 1:
        initial_para = pickle.load(open(pretrained_entire_main_net_path, 'rb'))
    else:
        initial_para = entire_main_net.get_weights() # get random initial parameters
    print('Entire Main Net is initialized')

    # ------------------------------ set data paths ------------------------------ #
    client_data_paths = get_all_client_data_paths(run_args.client_data_dist_type, run_args.num_total_clients, data_save_folder)

    # ------------------------------ Initialization ------------------------------ #
    shared_train_obj = TrainEntireMainNet(entire_main_net=entire_main_net,
                                            lr=run_args.lr,
                                            train_data_paths="",
                                            load_ratio=run_args.train_load_ratio,
                                            config = link_config,
                                            apply_aug=run_args.apply_aug,
                                            aug_times=run_args.aug_times)
    shared_test_obj = TestMainNet(entire_main_net=entire_main_net,
                                    test_data_paths="",
                                    load_ratio=run_args.test_load_ratio,
                                    config = link_config)
    
    # --------------- Select online coreset for each client's data --------------- #
    if run_args.use_coreset == 1:
        coreset_selector = CoresetSelector(entire_main_net = entire_main_net, config = link_config, buffer_size = run_args.buffer_size)
        core_train_batch_ids = dict()
        for client_id in range(run_args.num_total_clients):
            shared_train_obj.set_train_data_paths(client_data_paths[client_id])
            batched_train_data = shared_train_obj.load_data_pkl()  # based on train_load_ratio
            # print(len(batched_train_data))

            batch_ids = coreset_selector.select_coreset_batch_wise(batched_train_data)
            core_train_batch_ids[client_id] = batch_ids
            print("Client {} has selected the {} most useful batches".format(client_id, len(batch_ids)))
    
    # log_dir = root_path + '/OnlineAdapt/LocalOnly/profiler_logs'
    # os.mkdir(log_dir)
    # tf.profiler.experimental.start(log_dir)
    # --------------------------------- Training --------------------------------- #
    for client_id in range(1): # run_args.num_total_clients
        
        print('--- Local Training for Client {} --- '.format(client_id))
        if run_args.apply_aug == 1:
            print('Apply Data Augmentation for Client {}'.format(client_id))
            
        # load the training data
        shared_train_obj.set_train_data_paths(client_data_paths[client_id])
        batched_train_data = shared_train_obj.load_data_pkl()
        
        # enable coreset selection function for training
        if run_args.use_coreset == 1:
            core_train_data = [batched_train_data[b] for b in core_train_batch_ids[client_id]]
        else:
            core_train_data = batched_train_data
            
        shared_test_obj.set_test_data_paths(client_data_paths[client_id])
        batched_test_data = shared_test_obj.load_data_pkl()

        # intialize the local model
        shared_train_obj.set_model_weights(initial_para)

        # create folders for saving each client's local models
        client_model_path = Path(model_save_folder).joinpath('Client%d' % client_id)
        if not os.path.exists(client_model_path):
            os.mkdir(client_model_path)

        client_metric_path = Path(metric_save_folder).joinpath('Client%d' % client_id)
        if not os.path.exists(client_metric_path):
            os.mkdir(client_metric_path)

        train_loss_list = []
        test_loss_list = []
        test_ber_list = []

        # Train and test the model epoch by epoch:
        for e_id in range(run_args.num_epochs):
            
            # ---------------------------- Train for one epoch --------------------------- #
            epoch_trained_parameters, train_loss = shared_train_obj.train(num_epochs=1, batched_data=core_train_data)
            # train_loss_list.append(train_loss)

            # ---------------------------- Test for the current epoch ---------------------------- #            
            # shared_test_obj.set_model_weights(epoch_trained_parameters)
            # test_loss, test_ber = shared_test_obj.test(batched_test_data)
            # test_loss_list.append(test_loss)
            # test_ber_list.append(test_ber)

            # log_print("For Client {:d}, Epoch {:3d}, Train Loss: {:.3f}, Test loss: {:.3f}, Test ber: {:.2f}".format(
            #         client_id, e_id, train_loss, test_loss, test_ber), color='r')
                

            # # save models for every 5 epochs
            # # AdaptedModels/Client0/local_para_epoch_%d.pkl
            # if (e_id+1) % run_args.save_model_every == 0 and e_id > 0:
            #     with open(client_model_path.joinpath('local_para_epoch_%d.pkl' % e_id), 'wb') as file:
            #         pickle.dump(epoch_trained_parameters, file)

            # if run_args.debug == 0:
                # wandb.log({'local_epoch_bs{}'.format(client_id): e_id,
                #            'local_train_loss_bs{}'.format(client_id): train_loss,
                #            'local_test_loss{}'.format(client_id): test_loss,
                #            'local_test_ber{}'.format(client_id): test_ber})

        
        # ---------------- save training metrics after all the epochs: --------------- #
        ### TrainMetrics/Client0/all_epoch_train_loss.pkl
        # with open(client_metric_path.joinpath('all_epoch_train_loss.pkl'), 'wb') as f:
        #     pickle.dump(train_loss_list,f)

        # with open(client_metric_path.joinpath('all_epoch_test_loss.pkl'), 'wb') as f:
        #     pickle.dump(test_loss_list,f)

        # with open(client_metric_path.joinpath('all_epoch_test_ber.pkl'), 'wb') as f:
        #     pickle.dump(test_ber_list,f)

        print('--- Local Training for Client {} Finished --- '.format(client_id))

    # tf.profiler.experimental.stop()
    if run_args.debug == 0:
        run.finish()