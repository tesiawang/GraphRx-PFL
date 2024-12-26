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
import wandb
wandb.login()

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# set root path
root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)

from Data.get_config import BasicConfig, get_online_config
from Utils.models import SingleEntireMainNet
from Utils.model_train import TrainEntireMainNet
from Data.partition_data import get_client_data_configs, get_data_paths, get_all_client_data_paths
from Utils.functions import log_print, select_clients, save_global_metrics, save_local_metrics, save_local_model
from Utils.model_test import TestMainNet,multi_client_local_model_test
from OnlineAdapt.Algo.fedgraph_utils import aggregation_by_graph, aggregation_by_graph_norm,aggregation_by_graph_norm_update
from OnlineAdapt.Algo.fedgraph_utils import cal_model_diff, update_graph_matrix_directed, update_graph_matrix_undirected
import numpy as np
from Utils.coreset import CoresetSelector



# --------------------------------- Arguments --------------------------------- #
parser = argparse.ArgumentParser()
# seed
parser.add_argument("--seed", type=int, default=42, required=False)
# key mechansims
parser.add_argument("--use_coreset", type=int, default=0, required=False)
parser.add_argument("--buffer_size", type=int, default=0, required=False)
parser.add_argument("--apply_aug", type=int, default=1, required=False)
parser.add_argument("--aug_times", type=int, default=2, required=False)
# critical hyper-parameters
parser.add_argument("--penalty", type=int, default=0, required=False)
parser.add_argument("--local_iter", type=int, default=3, required=False) 
# NOTE : we define local_iter for the client with the smallest data quantity; 
# for other clients: local_iter = local_iter * data_quantity_ratio_compared_to_the_smallest
parser.add_argument("--SIR_pattern", type=str, default="HighSIR", required=False) # choose from "NoneSIR", "HighSIR"
parser.add_argument("--lam", type=float, default=0.01, required=False) # for local training regularization
parser.add_argument("--alpha", type=float, default=0.8, required=False) # for optimizing collab graph
parser.add_argument("--weighted_initial", type=int, default=1, required=False)
parser.add_argument("--opt_objective", type=int, default=1, required=False)
parser.add_argument("--hyper_c", type=float, default=0.8, required=False) 
parser.add_argument("--dist_metric", type=str, default="l2", required=False) # "cosine" or "l2" or "l1"
parser.add_argument("--directed_graph", type=int, default=1, required=False)
# common training set-up
parser.add_argument("--client_data_dist_type", type=int, default=0, required=False)
parser.add_argument("--load_pretrained", type=int, default=1, required=False)
parser.add_argument("--num_rounds", type=int, default=1, required = False)
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--record_batch_loss_every", type=int, default=1, required=False)
parser.add_argument("--save_model_every", type=int, default=10, required=False)
parser.add_argument("--clients_per_round", type=int, default=6, required = False)
parser.add_argument("--num_total_clients", type=int, default=6, required = False)
parser.add_argument("--train_load_ratio", type=float, default=0.1, required=False)
parser.add_argument("--test_load_ratio", type=float, default=0.1, required=False)
run_args = parser.parse_args()
# hyper-parameters
print(run_args)


if __name__=='__main__':
    tf.keras.utils.set_random_seed(run_args.seed)
    np.random.seed(run_args.seed)
    # tf.config.experimental.enable_op_determinism()
    _, cfg = get_online_config()

    # ------------------------------ set metric and model saving file paths ------------------------------ #
    
    pretrained_entire_main_net_path = root_path + "/OfflinePretrain/PretrainedModels_" + run_args.SIR_pattern +"_2Nr/updated_parameters_39.pkl"
    data_save_folder = root_path + "/Data/DataFiles/"+ run_args.SIR_pattern +"_2Nr/OnlineData"

    
    if run_args.dist_metric == "l2":
        # metric_save_folder = root_path+"/OnlineAdapt/FedGraphs/TrainMetrics_Dist{:d}_{:s}_c{:.2f}".format(run_args.client_data_dist_type, run_args.dist_metric, run_args.hyper_c) 
        model_save_folder = root_path+"/OnlineAdapt/FedGraphs/"+ run_args.SIR_pattern + "/new_AdaptedModels_Dist{:d}_{:s}_c{:.2f}_Aug{:d}_CS{:d}".format(run_args.client_data_dist_type, run_args.dist_metric, run_args.hyper_c,
                                                                                                                                                     run_args.aug_times, run_args.use_coreset)
    else:
        # metric_save_folder = root_path+"/OnlineAdapt/FedGraphs/TrainMetrics_Dist{:d}_{:s}".format(run_args.client_data_dist_type, run_args.dist_metric) 
        model_save_folder = root_path+"/OnlineAdapt/FedGraphs/"+ run_args.SIR_pattern + "/new_AdaptedModels_Dist{:d}_{:s}_Aug{:d}_CS{:d}".format(run_args.client_data_dist_type, run_args.dist_metric,
                                                                                                                                            run_args.aug_times, run_args.use_coreset)

    for folder in [model_save_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

    # --------------------------- Set config for wandb --------------------------- #
    wandb_config = {**vars(run_args),**cfg} # one-line code 
    run = wandb.init(project ="Infocom2025", config=wandb_config, name = "fedgraph-dist{:d}-{:s}-aug{:d}-CS{:d}-{:s}".format(run_args.client_data_dist_type,run_args.dist_metric, 
                                                                                                                            run_args.aug_times, run_args.use_coreset, run_args.SIR_pattern))

    # link configuration
    link_config = BasicConfig(num_bs_ant=cfg['num_bs_ant'], 
                              fft_size=cfg['fft_size'], 
                              num_bits_per_symbol=cfg['num_bits_per_symbol'],
                              pilot_ofdm_symbol_indices=cfg['pilot_ofdm_symbol_indices'])
    
        
    # --------------------------- Build and load pretrained model --------------------------- #    
    entire_main_net = SingleEntireMainNet()
    batch_pilots_rg = np.zeros((10, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_ut_ant), dtype=np.complex64)
    batch_y = np.zeros((10, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_bs_ant), dtype=np.complex64)
    batch_N0 = np.zeros(10, dtype=np.float32)
    entire_main_net([batch_pilots_rg, batch_y, batch_N0]) # build model

    if run_args.load_pretrained == 1:
        initial_para = pickle.load(open(pretrained_entire_main_net_path, 'rb'))
    else:
        initial_para = entire_main_net.get_weights()

    # ------------------------------ set data paths ------------------------------ #
    ### get the data paths for each client
    client_data_paths = get_all_client_data_paths(run_args.client_data_dist_type, run_args.num_total_clients, data_save_folder)
    for key in client_data_paths.keys():
        print("Client {:d} has {:d} data files".format(key, len(client_data_paths[key])))
    quantity_vec = [len(client_data_paths[c_id]) for c_id in range(run_args.num_total_clients)]
    quantity_vec = np.array(quantity_vec)/sum(quantity_vec)
    print(quantity_vec)
    
    # ------------------------------ Initialization ------------------------------ #
    weighted_model_para = {} 
    client_updated_model_para = {} 
    norm_weighted_para = {}
    norm_weighted_update = {}
    for c in range(run_args.num_total_clients):
        weighted_model_para[c] = initial_para  # initialize weighted models with the pretrained model
        client_updated_model_para[c] = initial_para # initialize them with the pretrained model
        norm_weighted_para[c] = 0
        norm_weighted_update[c] = 0

    # NOTE: initialize the collaboration graph W: a fully connected graph with self-weight = 0
    # The learned collaboration graph may be affected by the initialization
    graph_matrix = np.ones((run_args.num_total_clients, run_args.num_total_clients)) / (run_args.num_total_clients)
    graph_matrix[range(run_args.num_total_clients), range(run_args.num_total_clients)] = 0  
    # graph_matrix = tf.convert_to_tensor(graph_matrix)

    print("The inital collaboration graph matrix is:")
    print(graph_matrix)

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
    

    # --------------------- Start federated training --------------------- #
    log_print('=== Strat Training with {} Clients ==='.format(run_args.clients_per_round), color='r')
    for i in range(run_args.num_rounds):
    
        log_print('------------- Round {} ---------------'.format(i), color='r')
        # client selection
        round_selected_clients = select_clients(round=i, 
                                                num_total_clients=run_args.num_total_clients, 
                                                num_selected_clients=run_args.clients_per_round,
                                                client_sampling=0,
                                                data_save_folder=data_save_folder)
        train_loss_dict = dict()
        avg_train_loss = 0.
        for client_id in round_selected_clients:
            
            log_print('Start Local Training At Client {}'.format(client_id), color='r')

            # weighted initialization: initial the client model with the weighted cluster model
            # Here we consider the un-normed version
            if run_args.weighted_initial == 1:
                shared_train_obj.set_model_weights(weighted_model_para[client_id])
            else: 
                shared_train_obj.set_model_weights(client_updated_model_para[client_id]) # without weighted initial:


            # local training for X epochs
            shared_train_obj.set_train_data_paths(client_data_paths[client_id])
            batched_train_data = shared_train_obj.load_data_pkl()
            num_local_iter = run_args.local_iter * len(client_data_paths[client_id])

            # enable coreset selection function for training
            if run_args.use_coreset == 1:
                core_train_data = [batched_train_data[b] for b in core_train_batch_ids[client_id]]
            else:
                core_train_data = batched_train_data

            if run_args.penalty == 0 or run_args.penalty == 1:
                updated_parameters, train_loss = shared_train_obj.train_graph(round = i,
                                                                        local_iter = num_local_iter,
                                                                        lam = run_args.lam,
                                                                        weighted_model=weighted_model_para[client_id],
                                                                        record_batch_loss_every = run_args.record_batch_loss_every,
                                                                        batched_data=core_train_data,
                                                                        penalty = run_args.penalty)
            elif run_args.penalty == 2:
                updated_parameters, train_loss = shared_train_obj.train_graph(round = i,
                                                    local_iter = num_local_iter,
                                                    lam = run_args.lam,
                                                    weighted_model=norm_weighted_para[client_id],
                                                    record_batch_loss_every = run_args.record_batch_loss_every,
                                                    batched_data=core_train_data,
                                                    penalty = run_args.penalty)
            elif run_args.penalty == 3:
                updated_parameters, train_loss = shared_train_obj.train_graph(round = i,
                                                    local_iter = num_local_iter,
                                                    lam = run_args.lam,
                                                    weighted_model=norm_weighted_update[client_id],
                                                    record_batch_loss_every = run_args.record_batch_loss_every,
                                                    batched_data=core_train_data,
                                                    penalty = run_args.penalty,
                                                    initial_para = initial_para)
                
            # update and save the local model for the next round                      
            client_updated_model_para[client_id] = updated_parameters
            train_loss_dict[client_id] = train_loss

        # ---------------------------------------------------------------------------- #
        # approximation when computing cosine distance： if d < -0.9, d = -1
        model_diff_mat, client_ids = cal_model_diff(client_updated_model_para, initial_para, run_args.dist_metric)

        # update the collaboration graph W
        if run_args.directed_graph == 1:
            graph_matrix = update_graph_matrix_directed(graph_matrix, model_diff_mat, client_ids, run_args.alpha, run_args.opt_objective, run_args.hyper_c, quantity_vec)  
        else:  # learn an un-directed graph
            graph_matrix = update_graph_matrix_undirected(graph_matrix, model_diff_mat, client_ids, run_args.hyper_c, quantity_vec)
        

        print('Model difference (or model distance matrix): ')
        np.set_printoptions(precision=4, suppress=True)
        print(model_diff_mat)

        print('Collab graph matrix:')
        np.set_printoptions(precision=4, suppress=True)
        print(graph_matrix)


        # Weighted aggregation for the next round
        # Whatever the penalty term is, use the un-normed weighted aggregation model (weighted_model_para) as the next-round initial model (if weighted_initial == 1)
        if run_args.penalty == 0 or run_args.penalty == 1:
            weighted_model_para = aggregation_by_graph(graph_matrix, client_updated_model_para)
        elif run_args.penalty == 2:
            weighted_model_para, norm_weighted_para = aggregation_by_graph_norm(graph_matrix, client_updated_model_para)
        elif run_args.penalty == 3:
            weighted_model_para, norm_weighted_update = aggregation_by_graph_norm_update(graph_matrix, client_updated_model_para, initial_para)
        else:
            raise ValueError("Input a valid penalty term indicator: 0, 1, 2, 3")

        # compute the average loss of all the clients
        avg_train_loss = sum(train_loss_dict.values()) / len(train_loss_dict)

        # ------------------------------- Start Testing ------------------------------ #    
        # test loss = the average of test losses of the global model with local test data
        # test ber = the average of test bers of each neural receiver with local test data
        # NOTE: test weighted_model_para 
        test_loss_dict, test_ber_dict, avg_test_loss, avg_test_ber = multi_client_local_model_test(shared_test_obj, 
                                                                                                   weighted_model_para,
                                                                                                   client_data_paths)
        
        # print average train loss, average test loss
        print('Round {:3d}, Global train loss: {:.4f}, Global test loss: {:.4f}, Global test BER: {:.6f}'.format(
              i, avg_train_loss, avg_test_loss, avg_test_ber))
        
        # ------------------------- Saving Models and Losses ------------------------- #
        # save the global model weights and losses for the current training round
        # /AdaptedModels/Global/glob_para_round_%d.pkl
        # /TrainMetrics/Global/glob_metrics_round_%d.pkl
        # save_global_metrics(metric_save_folder, i, avg_train_loss, avg_test_loss, avg_test_ber)
        # save_local_metrics(metric_save_folder, i, train_loss_dict, test_loss_dict, test_ber_dict)
        
        log_test_loss = {'test_loss_bs' + str(key): value for key, value in test_loss_dict.items()}
        log_test_ber = {'test_ber_bs' + str(key): value for key, value in test_ber_dict.items()}
        
        wandb.log({'round': i,
                   'avg_train_loss': avg_train_loss,
                   'avg_test_loss': avg_test_loss,
                   'avg_test_ber': avg_test_ber,
                   **log_test_loss,
                   **log_test_ber})

        wandb.log({'round': i,
                   'avg_train_loss': avg_train_loss})
        
        # NOTE: save weighted_model_para
        if (i+1) % run_args.save_model_every == 0:
            save_local_model(model_save_folder,i, weighted_model_para)
   
    # ---------------------------------------------------------------------------- #
    run.finish()