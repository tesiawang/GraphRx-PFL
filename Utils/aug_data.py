import pickle
import numpy as np
from itertools import chain
import os
import sys
from pathlib import Path
os.environ['PYTHONHASHSEED'] = str(0)
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path) # set root path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sionna.utils import expand_to_rank, complex_normal
from Data.get_config import BasicConfig
from Data.partition_data import get_data_paths

'''
The following class is the realization of data augmentation process, independent of the training process.
'''

class DataAugmentor(tf.keras.Model):
    '''
     Simple Data Augmentation for Neural Receivers
    '''
    def __init__(self,
                 config: BasicConfig,
                 data_path: str,
                 load_ratio: float = 0.5,
                 aug_times: int = 64):
        '''
        Args:
            config: basic link configuration
            data_path: each augmentation is focused on one single data path (i.e., multiple batches of one single channel configuration)
            load_ratio: the ratio of the data to be loaded from the data_path
                E.g., the data_path contains 200 batches of data, and load_ratio = 0.1, then 20 batches of data will be loaded
            aug_times: the number of times to augment the data
                E.g., aug_times = 64, after augmentation, we have 20*64 = 1280 batches of data
        '''

        super(DataAugmentor, self).__init__(name='DataAugmentor')
        self._config = config
        self._load_ratio = load_ratio
        self._data_path = data_path
        self._aug_times = aug_times
        self._real_dtype = tf.dtypes.as_dtype(self._dtype).real_dtype


    def load_data_pkl(self):
        data = pickle.load(open(self._data_path, "rb"))
        return data[int((1-self._load_ratio)*len(data)):]
    

    # augment one batch out of the batched data
    def augment_one_batch(self,
                        batch_y_with_interf: tf.Tensor,
                        aug_var_N0: float, # original noise variance in linear scale
                        aug_h_phi: float):
        
        # ------------------- Augment the batched data ------------------- #
        # Rotate the channel
        aug_h_phi = tf.cast(aug_h_phi, batch_y_with_interf.dtype) #tf.complex64
        aug_batch_y =  tf.exp(1j * aug_h_phi) * batch_y_with_interf
        
        # Add noise
        aug_var_N0 = expand_to_rank(aug_var_N0, tf.rank(aug_batch_y), axis=-1) # expand the scalar to the same shape as aug_batch_y
        aug_var_N0 = tf.cast(aug_var_N0, self._real_dtype) # keep the data type real

        aug_batch_noise = complex_normal(tf.shape(aug_batch_y), dtype=aug_batch_y.dtype)
        aug_batch_noise *= tf.cast(tf.sqrt(aug_var_N0), aug_batch_noise.dtype)
        aug_batch_y = aug_batch_y + aug_batch_noise

        # shape of aug_batch_y: 
        # shape of aug_var_N0: the same as aug_batch_y
        return aug_batch_y
    

    def augment_data(self):
        '''
        Augment the data by rotating the channel and adding noise
        '''
        # load the original data to the cpu
        batched_data = self.load_data_pkl()

        # Augment per batch
        augmented_data_list = [] # len = len(batched_data) * aug_times
        for batch_id in range(len(batched_data)):

            # Get one batch of data, and augment it by self._aug_times
            batch_y_with_interf = batched_data[batch_id]['batch_y_with_interf']
            batch_N0 = batched_data[batch_id]['batch_N0']

            # other unchanged data components
            batch_pilots_rg = batched_data[batch_id]['batch_pilots_rg']
            tx_codeword_bits = batched_data[batch_id]['tx_codeword_bits']

            # start augmentation
            for _ in range(self._aug_times):
                # randomly choose the noise power and the channel phase
                # original noise power = N0, augmented noise power = N0/2
                var_range = batch_N0[0] / 2.0
                aug_var_N0 = tf.random.uniform(shape = [1], minval=0, maxval=var_range, dtype=tf.float32) # generate a float scalar uniformly from [0, var_range]
                aug_h_phi = tf.random.uniform(shape = [1], minval=0, maxval=2*np.pi, dtype=tf.float32)

                aug_batch_y = self.augment_one_batch(batch_y_with_interf, aug_var_N0, aug_h_phi)
                aug_batch_N0 = tf.tile(aug_var_N0, [batch_y_with_interf.shape[0]]) # save the noise variance, but not the real noise power realization


                # store the augmented batched_data
                with tf.device('cpu:0'):
                    realization = dict()
                    realization['batch_y_with_interf'] = tf.identity(aug_batch_y)
                    realization['batch_N0'] = tf.identity(aug_batch_N0 + batch_N0)

                    # save the other unchanged data components for training
                    realization['batch_pilots_rg'] = tf.identity(batch_pilots_rg)
                    realization['tx_codeword_bits'] = tf.identity(tx_codeword_bits)

                augmented_data_list.append(realization)
        
        print("Finish data augmentation.")
        return augmented_data_list



if __name__ == "__main__":

    # ------------------ Data Augmentation ------------------ #
    data_save_folder = root_path + "/Data/DataFiles/NoneSIR_2Nr/OfflineData"
    data_paths = get_data_paths(data_save_folder,list(np.linspace(0, 19, 20, dtype=int)))
    aug_data_save_folder = root_path + "/Data/DataFiles/NoneSIR_2Nr/AugData"

    # data_save_folder = root_path + "/Data/DataFiles/Eval_Data/2Nr"
    # data_paths = get_data_paths(data_save_folder, list(np.linspace(0, 2, 3, dtype=int)))
    # aug_data_save_folder = root_path + "/Data/DataFiles/Eval_Data/Aug"

    if not os.path.exists(aug_data_save_folder):
        os.makedirs(aug_data_save_folder)

    for i in range(len(data_paths)):
        augmentor = DataAugmentor(config=BasicConfig(),
                                    data_path=data_paths[i],
                                    load_ratio=0.05,
                                    aug_times=20)
        augmented_data_list = augmentor.augment_data()

        # save the augmented data
        with open(aug_data_save_folder + "/config{:d}.pkl".format(i), "wb") as f:
            pickle.dump(augmented_data_list, f)
    print("Augmented data saved.")