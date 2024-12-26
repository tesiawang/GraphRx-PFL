    
import pickle
import numpy as np
from pathlib import Path
import sys

root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path) # set root path
from Data.get_config import get_online_config


channel_config_list, other_config = get_online_config() # get run_args
file_path = root_path + "/Data/config_names.txt"

for config_id in range(len(channel_config_list)):
    
    with open(file_path, 'a') as f:
        f.write("config%d_%s%s_ds%d_mobi%d\n" % (config_id,
                                                channel_config_list[config_id]['model_type'],
                                                channel_config_list[config_id]['PDP_mode'],
                                                channel_config_list[config_id]['delay_spread']*1e9,
                                                channel_config_list[config_id]['max_speed']))