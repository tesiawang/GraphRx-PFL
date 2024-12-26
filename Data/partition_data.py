# Offline and Online Data Partition
# BS = Client
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
import os

# channel_config_list, other_config = get_args() # get run_args
# set_seeds(other_config.random_seed) 
# data_save_folder = "" # sync with generate_data.py

# ---------------------------------------------------------------------------- #
def get_client_data_configs(dist_type, num_total_clients):
    data_config_ids = {}
    # the following data partitioning should be customized based on the paper set-up
    if dist_type == 0: # dist_type = 0 is a simplified version of dist_type = 1
        data_config_ids = {0: [10], # TDL-C, ds 500, mobi 5; ds
                        1: [10], # TDL-C, ds 500, mobi 5; 
                        2: [3], # TDL- B, ds 200, mobi 20 
                        3: [3], # TDL -B , ds 200, mobi 20
                        4: [17], # TDL-E, ds 500, mobi 20
                        5: [17]} # TDL-E, ds 500, mobi 20

    elif dist_type == 5: # introduce quantity divergence
        data_config_ids = {0: [0,1,2], # TDL-B, ds 50, mobi 5,20; ds 200, mobi 5
                           1: [10,11], # TDL-C ds 500 mobi 5-20
                            2: [0], # TDL B, ds 50, mobi 5
                            3: [9], # TDL C, ds 200, mobi 20
                            4: [12,13], # TDL E ds 50, mobi 5-20
                            5: [12,15,16,17]} # TDL E ds 50, mobi 5; TDLE ds 200, 20;  TDLE ds 500, mobi 5-20
    

    assert len(data_config_ids.keys()) == num_total_clients
    return data_config_ids


# ---------------------------------------------------------------------------- #
def get_data_paths(data_save_folder:str, config_idx:list):

    file_names = [name for name in os.listdir(data_save_folder)]
    substr = ["config{:d}_".format(idx) for idx in config_idx]
    # search for matching file names
    data_paths = [str(data_save_folder+'/'+s) for s in file_names if any(sub in s.lower() for sub in substr)]
    data_paths.sort(key=lambda path: next(int(sub[6:-1]) for sub in substr if sub in path)) # Sort data_paths based on the order of config_idx
    return data_paths


def get_all_client_data_paths(client_data_dist_type, num_total_clients, data_save_folder):
    client_data_configs = get_client_data_configs(client_data_dist_type, num_total_clients)
    client_data_paths = dict()
    for client_id in range(num_total_clients):
        client_data_paths[client_id] = get_data_paths(data_save_folder, client_data_configs[client_id])
    return client_data_paths

def get_all_client_data_paths_from_config(client_data_configs, data_save_folder):
    client_data_paths = dict()
    for client_id in range(len(client_data_configs)):
        client_data_paths[client_id] = get_data_paths(data_save_folder, client_data_configs[client_id])
    return client_data_paths

# ---------------------------------------------------------------------------- #
# The final output of running this file should be: offline_config_paths, BS_online_config_paths
