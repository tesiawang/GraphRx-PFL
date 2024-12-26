import os
import sys
import pickle
import numpy as np
from pathlib import Path

# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from DomainDistance.domain_disc import DiscEstimator
from DomainDistance.create_loss import create_loss
from DomainDistance.create_metric import create_metric
from Data.partition_data import get_client_data_configs, get_all_client_data_paths_from_config
from Data.get_config import get_online_config, BasicConfig 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_clients', type=int, default=6)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_max_epochs', type=int, default=5)
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--early_stop', type=int, default=2)
parser.add_argument('--client_data_dist_type', type=int, default=0)
args = parser.parse_args()

if __name__=='__main__':
    import wandb 
    wandb.login()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    _,cfg = get_online_config()


    # --------------------------- Set config for wandb --------------------------- #
    wandb_config = {**vars(args),**cfg} # one-line code 
    run = wandb.init(project ="Infocom2025", config=wandb_config, name ="DomainDistance")

    # Estimate all pairs of domains
    num_clients = args.num_clients
    # label_dim = 10
    # c = 0.01 # clipping parameter
    lr = args.lr
    num_max_epochs = args.num_max_epochs
    eval_every = args.eval_every
    early_stop = args.early_stop
    loss_func = create_loss('bce')
    metric_func = create_metric('bacc')


    # ------------------------------ Get data paths ------------------------------ #
    data_save_folder = root_path + "/Data/DataFiles/HighSIR_2Nr/OnlineData"
    local_data_configs = get_client_data_configs(dist_type = args.client_data_dist_type, 
                                                 num_total_clients = args.num_clients)
    print(local_data_configs)
    client_data_paths = get_all_client_data_paths_from_config(local_data_configs, data_save_folder)


    # ------------------ Get feature dim of input data dimension ----------------- #
    #[batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 2*num tx antenna + 1]
    link_config = BasicConfig(num_bs_ant=cfg['num_bs_ant'], 
                            fft_size=cfg['fft_size'], 
                            num_bits_per_symbol=cfg['num_bits_per_symbol'],
                            pilot_ofdm_symbol_indices=cfg['pilot_ofdm_symbol_indices'])
    feature_dim = (link_config._num_ofdm_symbols, link_config._fft_size, 2*link_config._num_ut_ant + 2*link_config._num_bs_ant + 1)
    # model_input : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 2*num tx antenna + 1]


    # ------------------------------ Estimate domain distance ------------------------------ #
    disc_estimator = DiscEstimator(num_clients, feature_dim, lr)
    disc_matrix = disc_estimator.estimate_all(num_max_epochs, eval_every, early_stop, loss_func, metric_func, client_data_paths)

    print(disc_matrix)
    pickle.dump(disc_matrix, open(root_path + "/DomainDistance/domain_distance_dist{:d}_interf.pkl".format(args.client_data_dist_type), "wb"))

    # ---------------------------------------------------------------------------- #
    run.finish()
    
