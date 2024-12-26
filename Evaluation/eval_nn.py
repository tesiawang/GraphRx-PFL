# -*- coding: utf-8 -*-
import pickle
import sionna
import tensorflow as tf
import numpy as np
from Data.get_config import BasicConfig
from itertools import chain
import random



# ---------------------------------------------------------------------------- #
#                           Evaluation Of Entire Net                           #
# ---------------------------------------------------------------------------- #

class _EvalEntireNet(tf.keras.Model):
    '''
    Keras model that implements the end-to-end systems.
    '''
    def __init__(self,
                 entire_main_net: tf.keras.Model,
                 config: BasicConfig,
                 eval_data_paths: list):
        

        super(_EvalEntireNet, self).__init__(name='_EvalEntireNet')
        self._entire_main_net = entire_main_net
        self._config = config
        self._removed_null_subc = sionna.ofdm.RemoveNulledSubcarriers(self._config._rg)
        self._rg_demapper = sionna.ofdm.ResourceGridDemapper(config._rg, config._sm)

        self._eval_data_paths = eval_data_paths
        self._batch_id = 0  # initialize it to zero!
        self._batched_data = []
        self._batch_time = []

    def load_data_pkl(self):
        # the total number of batches = 200 = maximum MC iterations
        batched_data = []
        for path in self._eval_data_paths:
            num_batch_per_config = int(200/len(self._eval_data_paths))
            with open(path, "rb") as f:
                data = pickle.load(f)
            batched_data.append(data[-num_batch_per_config:]) # load the last (num_batch_per_config) batches
        batched_data = list(chain.from_iterable(batched_data))

        # Data structure of batched_data: [{},{},{},{}]
        self._batched_data = batched_data # here, we do not explicitly return the batched_data
        print(f"Loaded {len(self._batched_data)} data points for evaluation.")
    

    # The following function must be traced to accelerate the evaluation process
    # 0.3s / batch by using @tf.function
    @tf.function
    def call(self,
             batch_size: int,
             ebno_db: float):

        # set the batch id
        batch_id = self._batch_id % len(self._batched_data) # self._batch_id can be larger than len(self._batched_data)

        # generate noise variance based on a SNR value
        batch_N0 = sionna.utils.ebnodb2no(ebno_db,
                                          self._config._num_bits_per_symbol,
                                          self._config._coderate,
                                          self._config._rg)
        
        # get batch_x_rg
        tx_codeword_bits = self._batched_data[batch_id]['tx_codeword_bits']
        batch_x = self._config._mapper(tx_codeword_bits)
        batch_x_rg = self._config._rg_mapper(batch_x) 

        # get batch_pilots_rg
        batch_pilots_rg = self._batched_data[batch_id]['batch_pilots_rg']

        # reuse the channel and interference data from saved data
        batch_h_freq = self._batched_data[batch_id]['batch_h_freq']
        interf_batch_y = self._batched_data[batch_id]['interf_batch_y'] # interference signal without additive noise
        
        # change the type of interf_batch_y (float32) to complex64
        interf_batch_y = tf.cast(interf_batch_y, tf.complex64)

        # apply the channel, add noise, and add interference ---> obtain the received signal for a given SNR
        batch_y_with_interf = self._config._channel_freq([batch_x_rg, batch_h_freq, batch_N0]) + interf_batch_y

        # reshape data
        batch_N0 = tf.expand_dims(batch_N0, axis=0)
        batch_N0 = tf.tile(batch_N0, [batch_y_with_interf.shape[0]])
        batch_y_with_interf = tf.squeeze(batch_y_with_interf, axis=1)
        batch_y_with_interf = tf.transpose(batch_y_with_interf, [0,2,3,1])
        batch_x_rg = tf.squeeze(batch_x_rg, axis=1)
        batch_x_rg = tf.transpose(batch_x_rg, [0,2,3,1])

        # ---------------------------------------------------------------------------- #
        # compute LLR using neural Rx
        batch_llr = self._entire_main_net([batch_pilots_rg, batch_y_with_interf, batch_N0])
        batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
        batch_llr = self._rg_demapper(batch_llr)
        batch_llr = tf.reshape(batch_llr, [tx_codeword_bits.shape[0], 1, 1, self._config._n])

        # LDPC decoding for bit estimaton
        b_hat = self._config._decoder(batch_llr)

        # get the true bits
        b = self._batched_data[batch_id]['b']

        # read the next batch in the next call
        self._batch_id += 1
        # ---------------------------------------------------------------------------- #
        return b, b_hat


class EvalEntireNet(tf.keras.Model):
    def __init__(self,
                 entire_main_net: tf.keras.Model,
                 config: BasicConfig,
                 ebNo_dB_range: np.ndarray,
                 result_save_path: list,
                 batch_size: int = 32,
                 num_target_block_errors: int = 1000,
                 max_mc_iter: int = 200, 
                 eval_data_paths: list = None):

        super(EvalEntireNet, self).__init__(name='EvalEntireNet')
        self._eval_obj = _EvalEntireNet(entire_main_net=entire_main_net, 
                                        config=config,
                                        eval_data_paths=eval_data_paths)
        self._ebNo_dB_range = ebNo_dB_range
        self._result_save_path = result_save_path
        self._batch_size = batch_size
        self._num_target_block_errors = num_target_block_errors
        self._max_mc_iter = max_mc_iter

    def load_data_pkl(self):
        self._eval_obj.load_data_pkl()

    def eval(self):
        # there is an inner loop in sim_ber() function
        '''
        Simulates until target number of errors is reached and returns BER/BLER.
        The simulation continues with the next SNR point if either num_target_bit_errors bit errors or num_target_block_errors block errors is achieved. 
        Further, it continues with the next SNR point after max_mc_iter batches of size batch_size have been simulated. 

        Early stopping allows to stop the simulation after the first error-free SNR point or after reaching a certain target_ber or target_bler.
        '''
        ber, bler = sionna.utils.sim_ber(self._eval_obj, # call this function to get b, b_hat for each batch
                                      self._ebNo_dB_range, # for each SNR points, run max_mc_iter batches of data
                                      batch_size=self._batch_size,
                                      num_target_block_errors=self._num_target_block_errors,
                                      max_mc_iter=self._max_mc_iter,
                                      verbose = False)
        ber = ber.numpy()
        bler = bler.numpy()
        if self._result_save_path == []:
            return ber, bler
        else:
            with open(self._result_save_path[0], "wb") as file:
                pickle.dump(ber, file)
            
            # with open(self._result_save_path[1], "wb") as file:
            #     pickle.dump(bler, file)
        return ber, bler
