# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from pathlib import Path
from termcolor import colored
import tensorflow as tf


# must explicitly set the seeds!!! 
# def set_seeds(seed):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     tf.random.set_seed(seed)
#     np.random.seed(seed)

def flatten_model(model_weights):
    return tf.concat([tf.reshape(w, [-1]) for w in model_weights], axis=0)


def select_clients(round, num_total_clients, num_selected_clients, client_sampling, data_save_folder):
    '''selects num_clients clients weighted by number of samples from possible_clients
    
    Args:
        num_select_clients: number of clients to select; 
        note that within function, num_clients is set to
        min(num_selected_clients, num_total_clients)
    
    Return:
        indices: an array of indices
    '''
    # this function's randomdess is controlled. 
    # different random seeds at each round 
    np.random.seed(round+4)

    # num_selected_clients <= num_total_clients
    num_selected_clients = min(num_total_clients, num_selected_clients)

    if client_sampling == 0: # uniformly client sampling
        indices = np.random.choice(range(num_total_clients), num_selected_clients, replace=False)
        return indices

    elif client_sampling == 1: # weighted sampling by data_size = num_batches * tti_batch_size
        online_data_size = []
        for client_id in range(num_total_clients):
            count_files = get_folder_files(str(Path(data_save_folder).joinpath("client%d" % client_id))) # return a list of file names
            online_data_size.append(len(count_files) - 1) # assume the same tti_batch_size, num_batches = count_files - 1, as there is one channel_config.pkl

        total_data_size = np.sum(np.asarray(online_data_size))
        pk = [item * 1.0 / total_data_size for item in online_data_size]
        indices = np.random.choice(range(num_total_clients), num_selected_clients, replace=False, p=pk)
        return indices



def log_print(text, color, end = '\n'):
    if color == 'r':
        print(colored(text, 'red'), end = end)
    elif color == 'g':
        print(colored(text, 'green'), end = end)
    elif color == 'b':
        print(colored(text, 'blue'), end = end)
    elif color == 'y':
        print(colored(text, 'yellow'), end = end)
    elif color == 'c':
        print(colored(text, 'cyan'), end = end)
    elif color == 'm':
        print(colored(text, 'magenta'), end = end)
    else:
        print(text, end = end)

def get_folder_files(path: str):
    return [name for name in os.listdir(path)]

def get_subfolder(path: str):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    return all(np.array_equal(a, b) for a, b in zip(list1, list2))



# ---------------------------------------------------------------------------- #
def save_global_model_metrics(model_save_folder, metric_save_folder, i, latest_global_model_parameters, avg_train_loss, avg_test_loss, avg_test_rate):
    global_directory_path = Path(model_save_folder).joinpath('Global')
    if not os.path.exists(global_directory_path):
        os.mkdir(global_directory_path)
    with open(global_directory_path.joinpath('glob_para_round_%d.pkl' % i), 'wb') as file:
        pickle.dump(latest_global_model_parameters, file)

    global_directory_path = Path(metric_save_folder).joinpath('Global')
    if not os.path.exists(global_directory_path):
        os.mkdir(global_directory_path)

    global_metrics = dict()
    global_metrics['avg_train_loss'] = avg_train_loss
    global_metrics['avg_test_loss'] = avg_test_loss
    global_metrics['avg_test_rate'] = avg_test_rate
    with open(global_directory_path.joinpath('glob_metrics_round_%d.pkl' % i), 'wb') as file:
        pickle.dump(global_metrics, file)


def save_global_metrics(metric_save_folder, i, avg_train_loss, avg_test_loss, avg_test_rate):
    global_directory_path = Path(metric_save_folder).joinpath('Global')
    if not os.path.exists(global_directory_path):
        os.mkdir(global_directory_path)

    global_metrics = dict()
    global_metrics['avg_train_loss'] = avg_train_loss
    global_metrics['avg_test_loss'] = avg_test_loss
    global_metrics['avg_test_rate'] = avg_test_rate
    with open(global_directory_path.joinpath('glob_metrics_round_%d.pkl' % i), 'wb') as file:
        pickle.dump(global_metrics, file)


def save_global_model(model_save_folder, i, latest_global_model_parameters):
    global_directory_path = Path(model_save_folder).joinpath('Global')
    if not os.path.exists(global_directory_path):
        os.mkdir(global_directory_path)
    with open(global_directory_path.joinpath('glob_para_round_%d.pkl' % i), 'wb') as file:
        pickle.dump(latest_global_model_parameters, file)


# ---------------------------------------------------------------------------- #
# save the local model weights and losses for the current training round
# /TrainMetrics/client0/local_metrics_round_%d.pkl
def save_local_metrics(metric_save_folder, i, train_loss_dict, test_loss_dict, test_rate_dict):
    for client_id in range(len(train_loss_dict)):
        client_metrics = dict()
        client_metrics['train_loss'] = train_loss_dict[client_id]
        client_metrics['test_loss'] = test_loss_dict[client_id]
        client_metrics['test_rate'] = test_rate_dict[client_id]

        client_directory_path = Path(metric_save_folder).joinpath('Client%d' % client_id)
        if not os.path.exists(client_directory_path):
            os.mkdir(client_directory_path)
        with open(client_directory_path.joinpath('local_metrics_round_%d.pkl' % i), 'wb') as file:
            pickle.dump(client_metrics, file)


def save_local_model(model_save_folder, i, local_model_parameters):
    for client_id in range(len(local_model_parameters)):
        # save local model weights
        client_directory_path = Path(model_save_folder).joinpath('Client%d' % client_id)
        os.makedirs(client_directory_path,exist_ok=True)
        with open(client_directory_path.joinpath('local_para_round_%d.pkl' % i), 'wb') as file:
            pickle.dump(local_model_parameters[client_id], file)


def save_local_model_metrics(model_save_folder, metric_save_folder, i, local_model_parameters, train_loss_dict, test_loss_dict, test_rate_dict):
    for client_id in range(len(train_loss_dict)):
        # save local model weights
        client_directory_path = Path(model_save_folder).joinpath('Client%d' % client_id)
        os.makedirs(client_directory_path,exist_ok=True)
        with open(client_directory_path.joinpath('local_para_round_%d.pkl' % i), 'wb') as file:
            pickle.dump(local_model_parameters[client_id], file)

        client_metrics = dict()
        client_metrics['train_loss'] = train_loss_dict[client_id]
        client_metrics['test_loss'] = test_loss_dict[client_id]
        client_metrics['test_rate'] = test_rate_dict[client_id]

        client_directory_path = Path(metric_save_folder).joinpath('Client%d' % client_id)
        if not os.path.exists(client_directory_path):
            os.mkdir(client_directory_path)
        with open(client_directory_path.joinpath('local_metrics_round_%d.pkl' % i), 'wb') as file:
            pickle.dump(client_metrics, file)
