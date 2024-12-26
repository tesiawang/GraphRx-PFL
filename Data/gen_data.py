# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
import sys
import os
import shutil
import time
import argparse
os.environ['PYTHONHASHSEED'] = str(0)
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path) # set root path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from Data.DataGenerator import DataGenerator
from Data.get_config import BasicConfig, get_offline_config, get_online_config, get_cka_eval_config

parser = argparse.ArgumentParser()
parser.add_argument('--generation_type', type=int, default=0, help='Generator with or without inter-cell interference')
parser.add_argument('--seed', type=int, default=53, help='Random seed')
parser.add_argument('--tti_batch_size', type=int, default=32, help='Batch size for each TTI')
parser.add_argument('--num_batches_per_config', type=int, default=200, help='Number of batches per configuration') # on hpc: 400
parser.add_argument('--num_batches_per_snr', type=int, default=10, help='Number of batches per SNR point') # on hpc: 200
run_args = parser.parse_args()

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":

    tf.keras.utils.set_random_seed(run_args.seed)
    np.random.seed(run_args.seed)
    channel_config_list, other_config = get_offline_config() # get run_run_args

    # change the number of antennas
    # other_config['num_bs_ant'] = 4

    if run_args.generation_type == 0:
        data_save_folder = root_path + "/Data/DataFiles/NoneSIR_" + str(other_config['num_bs_ant']) + "Nr" # single data configuration
        # data_save_folder = root_path + "/Data/DataFiles/Eval_CKA"
    elif run_args.generation_type == 1:
        data_save_folder = root_path + "/Data/DataFiles/MidSIR_"+ str(other_config['num_bs_ant']) + "Nr"
    elif run_args.generation_type == 2:
        data_save_folder = root_path + "/Data/DataFiles/HighSIR_"+ str(other_config['num_bs_ant']) + "Nr"
    else:
        raise ValueError("Invalid generator type, please choose from 0, 1, 2")


    os.makedirs(Path(data_save_folder), exist_ok=True)
    # if os.path.exists(Path(data_save_folder)):
    #     shutil.rmtree(Path(data_save_folder))
    #     os.makedirs(Path(data_save_folder))
    # else:
    #     os.makedirs(Path(data_save_folder))

    # ---------------------------- Set up link config ---------------------------- #
    link_config = BasicConfig(num_bs_ant=other_config['num_bs_ant'],
                              fft_size=other_config['fft_size'],
                              num_bits_per_symbol= other_config['num_bits_per_symbol'],
                              pilot_ofdm_symbol_indices=other_config['pilot_ofdm_symbol_indices']) # initialize wireless parameters and communication modules, such as mapper, demapper, etc.
    

    # -------- Generate offline data: generate data for each SNR point ------- #
    # this is the full snr range for the offline data generation
    min_ebNo, max_ebNo, num_ebNo_points = other_config['snr_range'].values()

    if run_args.generation_type == 0:
        generator = DataGenerator(init_config = link_config, # set 'add_interference = True' in this config 
                                    ebNo_dB_range=np.linspace(min_ebNo,max_ebNo,num_ebNo_points),
                                    SIR_dB_range = np.linspace(100,100,1), # set the SIR to 100 dB, which means the interference can be ignored
                                    apply_encoder = True,
                                    add_interference = False)
        
    elif run_args.generation_type == 1:
        min_SIR, max_SIR, num_SIR_points = other_config['mid_SIR'].values()
        generator = DataGenerator(init_config = link_config, 
                                    ebNo_dB_range=np.linspace(min_ebNo, max_ebNo, num_ebNo_points),
                                    SIR_dB_range=np.linspace(min_SIR,max_SIR,num_SIR_points),
                                    apply_encoder= True,
                                    add_interference= True)

    elif run_args.generation_type == 2:
        min_SIR, max_SIR, num_SIR_points = other_config['high_SIR'].values()
        generator = DataGenerator(init_config = link_config, 
                                    ebNo_dB_range=np.linspace(min_ebNo, max_ebNo, num_ebNo_points),
                                    SIR_dB_range=np.linspace(min_SIR, max_SIR, num_SIR_points),
                                    apply_encoder= True,
                                    add_interference= True)
    else:  
        raise ValueError("Invalid generator type, please choose from 0, 1, 2")
    


    # config_id = 0 
    # channel_config = channel_config_list[config_id]
    # print("Offline channel config list: ", channel_config_list)
    # print("Offline SNR range: ", other_config['snr_range'])
    # # print("Offline SIR range: ", other_config['mid_SIR'])

    # new_folder = data_save_folder + "/OfflineData"
    # os.makedirs(Path(new_folder), exist_ok=True)
    # print("Start generating offline data...Data will be saved in %s" % new_folder)

    
    # start = time.time()
    # link_config.set_channel_models(model_type=channel_config['model_type'],
    #                                 PDP_mode=channel_config['PDP_mode'],
    #                                 delay_spread=channel_config['delay_spread'], # this should be in the unit of second
    #                                 min_speed=channel_config['min_speed'],
    #                                 max_speed=channel_config['max_speed'],
    #                                 delta_delay_spread=channel_config['delta_delay_spread'])
    # generator.reset_link_config(link_config)


    # # generate data for each SNR point
    # for snr in np.linspace(min_ebNo, max_ebNo, num_ebNo_points):
    #     generator.reset_snr_range(snr, snr, 1)
    #     channel_realizations_per_snr = generator.receive_data(batch_size=run_args.tti_batch_size, num_batch=run_args.num_batches_per_snr) # 200 batches per SNR point
    #     with open(new_folder + "/config%d_%s%s_ds%d_mobi%d_snr%d.pkl" % (config_id,
    #                                                                     channel_config['model_type'],
    #                                                                     channel_config['PDP_mode'],
    #                                                                     channel_config['delay_spread']*1e9,
    #                                                                     channel_config['max_speed'],
    #                                                                     int(snr)), 'wb') as f:
    #         pickle.dump(channel_realizations_per_snr, f)
    #     config_id += 1
    #     print("Finish generating data for SNR %.2f" % snr)

    # end = time.time()
    # print("Total Time: ", end - start)



    # ---------------------------- Generate online data ---------------------------- #

    new_folder = data_save_folder + "/OnlineData"
    if os.path.exists(Path(new_folder)):
        shutil.rmtree(Path(new_folder))
        os.makedirs(Path(new_folder))
    else:
        os.makedirs(Path(new_folder))
    # os.makedirs(Path(new_folder), exist_ok=True)
    print("Start generating online data...Data will be saved in %s" % new_folder)

    channel_config_list, cfg = get_online_config()
    online_min_ebNo, online_max_ebNo, online_num_ebNo_points = cfg['snr_range'].values()
    print("Online channel config list: ", len(channel_config_list))
    print("Online critical configs: ", cfg['snr_range'])

    start = time.time()
    for config_id in range(len(channel_config_list)):
        channel_realizations_per_config = []
        link_config.set_channel_models(model_type=channel_config_list[config_id]['model_type'],
                                        PDP_mode=channel_config_list[config_id]['PDP_mode'],
                                        delay_spread=channel_config_list[config_id]['delay_spread'], # this should be in the unit of second
                                        min_speed=channel_config_list[config_id]['min_speed'],
                                        max_speed=channel_config_list[config_id]['max_speed'],
                                        delta_delay_spread=channel_config_list[config_id]['delta_delay_spread'])
        

        generator.reset_link_config(link_config)
        generator.reset_snr_range(online_min_ebNo, online_max_ebNo, online_num_ebNo_points) # set the high SNR range for online data generation
        channel_realizations_per_config = generator.receive_data(batch_size=run_args.tti_batch_size, num_batch=run_args.num_batches_per_config) 
        with open(new_folder + "/config%d_%s%s_ds%d_mobi%d.pkl" % (config_id,
                                                                    channel_config_list[config_id]['model_type'],
                                                                    channel_config_list[config_id]['PDP_mode'],
                                                                    channel_config_list[config_id]['delay_spread']*1e9,
                                                                    channel_config_list[config_id]['max_speed']), 'wb') as f:
            pickle.dump(channel_realizations_per_config, f)
        print("Finish generating data for channel config %d" % config_id)

    end = time.time()
    print("Total Time: ", end - start)
        
