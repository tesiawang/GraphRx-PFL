print("Successfully in the fedavg.py file")
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
import random

# turn off the device INFO messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# set root path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)
from Data.get_config import BasicConfig, get_online_config
from Utils.models import SingleEntireMainNet
from Utils.functions import log_print, select_clients, save_global_model,save_local_model
from Utils.model_train import TrainEntireMainNet
from Data.partition_data import get_all_client_data_paths, get_data_paths
from Utils.model_test import TestMainNet, multi_client_global_model_test, multi_client_local_model_test
from Utils.model_aggregate import aggregate_model_updates
from Utils.coreset import CoresetSelector


# --------------------------------- Arguments --------------------------------- #
parser = argparse.ArgumentParser()
# seed
parser.add_argument("--seed", type=int, default=42, required=False) 
# training
parser.add_argument("--fed_alg", type=str, default="FedAvgs", required=False)
parser.add_argument("--exp_id", type=int, default=999, required=False)
# key mechansims
parser.add_argument("--use_coreset", type=int, default=0, required=False)
parser.add_argument("--buffer_size", type=int, default=0, required=False)
parser.add_argument("--apply_aug", type=int, default=0, required=False)
parser.add_argument("--aug_times", type=int, default=1, required=False)
# common parameters
parser.add_argument("--SIR_pattern", type=str, default="NoneSIR", required=False) # choose from "NoneSIR", "HighSIR"
parser.add_argument("--save_model_every", type=int, default=10, required=False)
parser.add_argument("--num_rounds", type=int, default=2, required = False)
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--clients_per_round", type=int, default=6, required = False)
parser.add_argument("--num_total_clients", type=int, default=6, required = False)
parser.add_argument("--train_load_ratio", type=float, default=0.5, required=False)
parser.add_argument("--test_load_ratio", type=float, default=0.1, required=False)
parser.add_argument("--client_data_dist_type", type=int, default=4, required=False)
parser.add_argument("--load_pretrained", type=int, default=1, required=False)
parser.add_argument("--memory_limit", type=int, default=24, required=False)
# for fedavg
parser.add_argument("--num_epochs", type=int, default=1, required=False)
# for fedrep
parser.add_argument("--num_epochs_for_head", type=int, default=2, required=False)
parser.add_argument("--num_epochs_for_rep", type=int, default=2, required=False)
parser.add_argument("--num_glob_layers", type=int, default=9, required=False)
# for ditto
parser.add_argument("--lam", type=float, default=0.1, required = False)
parser.add_argument("--local_lr", type=float, default=3e-3, required = False)
parser.add_argument("--local_iter", type=int, default=15, required = False)
parser.add_argument("--record_batch_loss_every", type=int, default=5, required = False)
# for fedavgFT
parser.add_argument("--FT_num_epochs", type=int, default=1, required=False)
parser.add_argument("--FT", type=int, default=0, required=False)
# TODO: change the path for FedAvg-FT...
parser.add_argument("--load_fedavg_path", type=str, default=root_path + "/OnlineAdapt/FedAvgs/AdaptedModels_EXP0/Global/glob_para_round_29.pkl", required=False)

parser.add_argument("--debug", type=int, default=0, required=False)
run_args = parser.parse_args()



if __name__=='__main__':
    if run_args.debug == 0:
        import wandb
        wandb.login()

    # ------------------------------ set random seed ----------------------------- #
    tf.keras.utils.set_random_seed(run_args.seed)
    np.random.seed(run_args.seed)
    random.seed(run_args.seed)
    _,cfg = get_online_config()

    # ----------------------------- set wandb config ----------------------------- #
    if run_args.debug == 0:
        wandb_config = {**vars(run_args),**cfg} # one-line code 
        run = wandb.init(project='Infocom2025', config=wandb_config, 
                         name = "{:s}-dist{:d}-aug{:d}-CS{:d}-{:s}".format(run_args.fed_alg, run_args.client_data_dist_type,run_args.aug_times,run_args.use_coreset,run_args.SIR_pattern))


    # ------------------------------ set file paths ------------------------------ #
    pretrained_entire_main_net_path = root_path + "/OfflinePretrain/PretrainedModels_" + run_args.SIR_pattern +"_2Nr/updated_parameters_39.pkl"
    data_save_folder = root_path + "/Data/DataFiles/"+ run_args.SIR_pattern +"_2Nr/OnlineData"

    
    model_save_folder = root_path + "/OnlineAdapt/" + run_args.fed_alg + \
                        "/" + run_args.SIR_pattern + "/AdaptedModels_Dist{:d}_Aug{:d}_CS{:d}".format(run_args.client_data_dist_type, run_args.aug_times, run_args.use_coreset)
    # metric_save_folder = root_path + "/OnlineAdapt/" + run_args.fed_alg + \
    #                     "/TrainMetrics_Dist{:d}_Aug{:d}_Coreset{:d}".format(run_args.client_data_dist_type, run_args.aug_times, run_args.use_coreset)
    
    for folder in [model_save_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)


    # ------------------------------ Build the model ----------------------------- #
    link_config = BasicConfig(num_bs_ant=cfg['num_bs_ant'], 
                              fft_size=cfg['fft_size'], 
                              num_bits_per_symbol=cfg['num_bits_per_symbol'],
                              pilot_ofdm_symbol_indices=cfg['pilot_ofdm_symbol_indices'])
    net_glob = SingleEntireMainNet()
    batch_pilots_rg = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_ut_ant), dtype=np.complex64)
    batch_y = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_bs_ant), dtype=np.complex64)
    batch_N0 = np.zeros(1, dtype=np.float32)
    net_glob([batch_pilots_rg, batch_y, batch_N0])


    # -------------------------- Load pretrained weights ------------------------- #
    if run_args.load_pretrained == 1 and run_args.FT == 0: # if FT == 1, we load the FedAvg model weights later
        initial_para = pickle.load(open(pretrained_entire_main_net_path, 'rb'))
    else:
        initial_para = net_glob.get_weights() # get random initial parameters


    # ------------------------------ Set data paths ------------------------------ #
    client_data_paths = get_all_client_data_paths(run_args.client_data_dist_type, run_args.num_total_clients, data_save_folder)


    # ------------------------------ Initialization ------------------------------ #
    client_model_updates = dict()
    client_layermask = dict()
    for client_id in range(run_args.num_total_clients):
        client_layermask[client_id] = [1 for _ in range(len(initial_para))]
    
    # Assuming quantity_vec is a list of quantities for each client
    quantity_vec = [len(client_data_paths[c_id]) for c_id in range(run_args.num_total_clients)]
    quantity_vec = np.array(quantity_vec) / sum(quantity_vec)

    # Turn quantity_vec into a dictionary
    client_aggregation_weight = {c_id: np.array([quantity_vec[c_id]], dtype=np.float64) for c_id in range(run_args.num_total_clients)}
    print(client_aggregation_weight)

    # initialize the local trainer
    if run_args.fed_alg in ["FedAvgs","FedAvgFT"]:
        shared_train_obj = TrainEntireMainNet(entire_main_net=net_glob,
                                            lr=run_args.lr,
                                            train_data_paths="",
                                            load_ratio=run_args.train_load_ratio,
                                            config = link_config,
                                            apply_aug=bool(run_args.apply_aug),
                                            aug_times=run_args.aug_times)
        
        shared_test_obj = TestMainNet(entire_main_net=net_glob,
                                     test_data_paths="",
                                     load_ratio=run_args.test_load_ratio,
                                     config = link_config)
    

    elif run_args.fed_alg in ["FedReps", "Dittos"]:
        # initialize the local trainer
        net_local = SingleEntireMainNet()
        net_local([batch_pilots_rg, batch_y, batch_N0])
        shared_train_obj = TrainEntireMainNet(entire_main_net=net_local,
                                            lr=run_args.lr,
                                            train_data_paths="",
                                            load_ratio=run_args.train_load_ratio,
                                            config = link_config,
                                            apply_aug=bool(run_args.apply_aug),
                                            aug_times=run_args.aug_times)
    
        shared_test_obj = TestMainNet(entire_main_net=net_local,
                                    test_data_paths="",
                                    load_ratio=run_args.test_load_ratio,
                                    config = link_config)
        if run_args.fed_alg == "Dittos":
            # Add one more training object for the global model
            shared_train_obj_glob = TrainEntireMainNet(entire_main_net=net_glob,
                                                        lr=run_args.lr,
                                                        train_data_paths="",
                                                        load_ratio=run_args.train_load_ratio,
                                                        config = link_config,
                                                        apply_aug=bool(run_args.apply_aug),
                                                        aug_times=run_args.aug_times)


    # --------------- Select online coreset for each client's data --------------- #
    if run_args.use_coreset == 1:
        coreset_selector = CoresetSelector(entire_main_net = net_glob, config = link_config, buffer_size = run_args.buffer_size)
        core_train_batch_ids = dict()
        for client_id in range(run_args.num_total_clients):
            shared_train_obj.set_train_data_paths(client_data_paths[client_id])
            batched_train_data = shared_train_obj.load_data_pkl()  # based on train_load_ratio
            # print(len(batched_train_data))

            batch_ids = coreset_selector.select_coreset_batch_wise(batched_train_data)
            core_train_batch_ids[client_id] = batch_ids
            print("Client {} has selected the {} most useful batches".format(client_id, len(batch_ids)))
    

    # ------------------------------ Round Training ----------------------------- #
    # initialize the global model and local model weights
    latest_w_glob = initial_para
    if run_args.fed_alg in ["FedAvgFT", "Dittos"]:
        local_weights_saver = dict()
        for client_id in range(run_args.num_total_clients):
            local_weights_saver[client_id] = latest_w_glob
    if run_args.fed_alg == "FedReps":
        local_weights_saver = dict()
        local_weights_after_aggre = dict()
        for client_id in range(run_args.num_total_clients):
            local_weights_after_aggre[client_id] = latest_w_glob
            local_weights_saver[client_id] = latest_w_glob
    
    log_print('=== Strat Training with {} clients ==='.format(run_args.clients_per_round), color='r')
    for i in range(run_args.num_rounds):
        # If this is a finetune process: directly break the loop
        if run_args.FT == 1:
            break

        log_print('--- Round {} ---'.format(i), color='r')
        
        # client selection
        round_selected_clients = select_clients(round=i, 
                                                num_total_clients=run_args.num_total_clients, 
                                                num_selected_clients=run_args.clients_per_round,
                                                client_sampling=0,
                                                data_save_folder=data_save_folder)

        train_loss_dict = dict()
        avg_train_loss = 0.
        for client_id in round_selected_clients:
            
            log_print('Start Local Training At client {}'.format(client_id), color='r')
            # load the training data for the current client
            shared_train_obj.set_train_data_paths(client_data_paths[client_id])
            batched_train_data = shared_train_obj.load_data_pkl()

            # enable coreset selection function for training
            if run_args.use_coreset == 1:
                core_train_data = [batched_train_data[b] for b in core_train_batch_ids[client_id]]
            else:
                core_train_data = batched_train_data

            # Set the initial model weights for this round and do local training
            if run_args.fed_alg == "FedAvgs":
                shared_train_obj.set_model_weights(latest_w_glob)
                updated_parameters, train_loss = shared_train_obj.train(num_epochs=run_args.num_epochs, batched_data=core_train_data)
                grads = [u - v for (u,v) in zip(updated_parameters, latest_w_glob)]
                client_model_updates[client_id] = grads
                train_loss_dict[client_id] = train_loss

            elif run_args.fed_alg == "FedReps":
                glob_layers = list(np.arange(run_args.num_glob_layers))
                shared_train_obj.set_model_weights(local_weights_after_aggre[client_id])
                _ , temp_train_loss  = shared_train_obj.train_head(glob_layers, num_epochs = run_args.num_epochs_for_head, batched_data=core_train_data)
                w_local, train_loss = shared_train_obj.train_rep(glob_layers, num_epochs = run_args.num_epochs_for_rep, batched_data=core_train_data)
                
                # compute client model updates ( = a list of gradients layer by layer)
                grads = [u - v for (u,v) in zip(w_local, latest_w_glob)]                
                client_model_updates[client_id] = grads
                # save loss and weights
                train_loss_dict[client_id] = train_loss
                local_weights_saver[client_id] = w_local # only use w_local to save temporary results

            elif run_args.fed_alg == "Dittos":
                # update the local models with regularization
                print('Update the personalized model locally')
                shared_train_obj.set_model_weights(local_weights_saver[client_id])
                num_local_iter = run_args.local_iter * len(client_data_paths[client_id])
                w_local, local_train_loss = shared_train_obj.train_ditto(num_local_iter, 
                                                                        run_args.lam,
                                                                        latest_w_glob,
                                                                        run_args.local_lr,
                                                                        run_args.record_batch_loss_every,
                                                                        core_train_data)

                # update the global model
                print('Update the global model locally')
                shared_train_obj_glob.set_model_weights(latest_w_glob)
                updated_parameters, _ = shared_train_obj_glob.train(run_args.num_epochs, core_train_data)

                # calcuate parameter difference between updated_parameters and global parameters
                grads = [u - v for (u,v) in zip(updated_parameters, latest_w_glob)]
                client_model_updates[client_id] = grads
                local_weights_saver[client_id] = w_local
                train_loss_dict[client_id] = local_train_loss
    

        # ------------------------------- Aggregation ------------------------------ #
        aggregated_model_updates = aggregate_model_updates(client_model_updates=client_model_updates,
                                                           client_aggregation_weight=client_aggregation_weight,
                                                           client_layermask=client_layermask)
        updated_w_glob = [latest_w_glob[layer_idx] + aggregated_model_updates[layer_idx]
                          for layer_idx in range(len(latest_w_glob))]
        
        # ---------------------- Get the latest global model ---------------------- #
        latest_w_glob = updated_w_glob
        avg_train_loss = sum(train_loss_dict.values()) / len(train_loss_dict)
        

        # ------------------------------- Testing ------------------------------ #    
        # test loss = the average of test losses of the global model with local test data
        # test ber = the average of test bers of each neural receiver with local test data
        if run_args.fed_alg == "FedAvgs":
            # test global models
            test_loss_dict, test_ber_dict, avg_test_loss, avg_test_ber = multi_client_global_model_test(shared_test_obj, 
                                                                                                        latest_w_glob,
                                                                                                        client_data_paths)
            if (i+1) % run_args.save_model_every == 0:
                save_global_model(model_save_folder, i, latest_w_glob)
        
        elif run_args.fed_alg in ["Dittos"]:
            # test local models
            test_loss_dict, test_ber_dict, avg_test_loss, avg_test_ber = multi_client_local_model_test(shared_test_obj, 
                                                                                                        local_weights_saver,
                                                                                                        client_data_paths)
            if (i+1) % run_args.save_model_every == 0:
                save_local_model(model_save_folder, i, local_weights_saver)


        elif run_args.fed_alg in ["FedReps"]:
            for client_id in range(run_args.num_total_clients):
                net_glob.set_weights(latest_w_glob)
                net_local.set_weights(local_weights_saver[client_id])
                glob_layers = list(np.arange(run_args.num_glob_layers))
                for layer_idx in range(len(net_glob.layers)): # len(net_glob.layers) = 13
                    if layer_idx in glob_layers:
                        net_local.layers[layer_idx].set_weights(net_glob.layers[layer_idx].get_weights())
                local_weights_after_aggre[client_id] = net_local.get_weights()

            test_loss_dict, test_ber_dict, avg_test_loss, avg_test_ber = multi_client_local_model_test(shared_test_obj, 
                                                                                                        local_weights_after_aggre,
                                                                                                        client_data_paths)
            if (i+1) % run_args.save_model_every == 0:
                save_local_model(model_save_folder, i, local_weights_after_aggre)
       
       
        # ------------------------- Save metrics ------------------------- #
        # save the global model weights and losses for the current training round
        # /AdaptedModels/Global/glob_para_round_%d.pkl
        # /TrainMetrics/Global/glob_metrics_round_%d.pkl
        # save_global_metrics(metric_save_folder, i, avg_train_loss, avg_test_loss, avg_test_ber)
        # save_local_metrics(metric_save_folder, i, train_loss_dict, test_loss_dict, test_ber_dict)
        
        
        log_test_loss = {'test_loss_bs' + str(key): value for key, value in test_loss_dict.items()}
        log_test_ber = {'test_ber_bs' + str(key): value for key, value in test_ber_dict.items()}
        if run_args.debug == 0:
            wandb.log({'round': i,
                    'avg_train_loss': avg_train_loss,
                    'avg_test_loss': avg_test_loss,
                    'avg_test_ber': avg_test_ber,
                    **log_test_loss,
                    **log_test_ber})
        print('Round {:2d}, Average train loss: {:.3f}, Average test loss: {:.3f}, Average test ber: {:.4f}'.format(i, avg_train_loss, avg_test_loss, avg_test_ber))
    


    # ---------------------------------------------------------------------------- #
    #                              FedAvg + FineTuning                             #
    # ---------------------------------------------------------------------------- #
    if run_args.FT == 1 and run_args.fed_alg == "FedAvgFT":
        train_loss_dict = dict()
        for client_id in range(run_args.num_total_clients):
            log_print("Fine-tuning the local model at client {}".format(client_id), color="r")
            fedavg_glob_w_path = run_args.load_fedavg_path
            with open(fedavg_glob_w_path, 'rb') as f:
                latest_w_glob = pickle.load(f)

            shared_train_obj.set_model_weights(latest_w_glob) # FT based on the aggregated global model
            shared_train_obj.set_train_data_paths(client_data_paths[client_id])
            batched_train_data = shared_train_obj.load_data_pkl()
            updated_parameters, train_loss = shared_train_obj.train(num_epochs=run_args.FT_num_epochs, batched_data=core_train_data)
            local_weights_saver[client_id] = updated_parameters
            train_loss_dict[client_id]  = train_loss

        # save the fine-tuned local models  
        test_loss_dict, test_ber_dict, avg_test_loss, avg_test_ber = multi_client_local_model_test(shared_test_obj, 
                                                                                                   local_weights_saver,
                                                                                                   client_data_paths)
        
        log_test_loss = {'test_loss_bs' + str(key): value for key, value in test_loss_dict.items()}
        log_test_ber = {'test_ber_bs' + str(key): value for key, value in test_ber_dict.items()}
        if run_args.debug == 0:
            wandb.log({'avg_train_loss': avg_train_loss,
                    'avg_test_loss': avg_test_loss,
                    'avg_test_ber': avg_test_ber,
                    **log_test_loss,
                    **log_test_ber})
            
        save_local_model(model_save_folder, run_args.num_rounds, local_weights_saver)  
        # save_local_model_metrics(model_save_folder, metric_save_folder, run_args.num_rounds, local_weights_saver, train_loss_dict, test_loss_dict, test_ber_dict)

 # ---------------------------------------------------------------------------- #
    run.finish()

    
