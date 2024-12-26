
# -*- coding: utf-8 -*-
import pickle
import sionna
import tensorflow as tf
import numpy as np
from itertools import chain

tf.get_logger().setLevel('ERROR')

from Data.get_config import BasicConfig
from sionna.ofdm.channel_estimation import BaseChannelEstimator
from Utils.functions import log_print


# ---------------------------------------------------------------------------- #
#                         Evaluation Of Conventional Rx                        #
# ---------------------------------------------------------------------------- #
class _EvalBsRx(tf.keras.Model):
    def __init__(self,
                 perfect_csi: bool,
                 config: BasicConfig,
                 eval_data_paths: list = None,
                 load_ratio: float = 0.7):
        super(_EvalBsRx, self).__init__(name='_EvalBsRx')
        self._perfect_csi = perfect_csi
        self._config = config
        self._removed_null_subc = sionna.ofdm.RemoveNulledSubcarriers(self._config._rg)

        self._load_ratio = load_ratio
        self._eval_data_paths = eval_data_paths
        self._batch_id = 0  
        self._batched_data = []

    def load_data_pkl(self): # TODO: need to change this load_data_pkl to match with eval_nn.py
        batched_data = []
        for path in self._eval_data_paths:
            with open(path, "rb") as f:
                data = pickle.load(f)
            batched_data.append(data[int((1-self._load_ratio)*len(data)):]) # load the last % of the data
        batched_data = list(chain.from_iterable(batched_data))
        # Data structure of batched_data: [{},{},{},{}]
        self._batched_data = batched_data

    @tf.function
    def call(self,
             batch_size: int,
             ebno_db: float):
        
        # set the batch id
        batch_id = self._batch_id % len(self._batched_data) # self._batch_id can be larger than len(self._batched_data)

        # generate the batch of noise based on a SNR value
        batch_N0 = sionna.utils.ebnodb2no(ebno_db,
                                          self._config._num_bits_per_symbol,
                                          self._config._coderate,
                                          self._config._rg)
        
        # ---------------------------------------------------------------------------- #
        # get batch_x_rg
        tx_codeword_bits = self._batched_data[batch_id]['tx_codeword_bits']
        batch_x = self._config._mapper(tx_codeword_bits)
        batch_x_rg = self._config._rg_mapper(batch_x) 

        # # get batch_pilots_rg
        # batch_pilots_rg = self._batched_data[batch_id]['batch_pilots_rg']

        # reuse the channel and interference data from saved data
        batch_h_freq = self._batched_data[batch_id]['batch_h_freq']
        interf_batch_y = self._batched_data[batch_id]['interf_batch_y'] # interference signal without additive noise
        interf_batch_y = tf.cast(interf_batch_y, tf.complex64)

        # apply the channel, add noise, and add interference ---> obtain the received signal for a given SNR
        batch_y_with_interf = self._config._channel_freq([batch_x_rg, batch_h_freq, batch_N0]) + interf_batch_y

        ### Here we do not need to reshape data

        # ------------------------------- Demodulation ------------------------------- #
        if self._perfect_csi:
            batch_h_hat = self._removed_null_subc(batch_h_freq)
            batch_err_var = 0.0
        else:
            batch_h_hat, batch_err_var = self._config._ls_est([batch_y_with_interf, batch_N0]) # why here do not need pilot config

        # TODO: implement different channel estimators and detectors
        batch_x_hat, batch_no_eff = self._config._lmmse_equ([batch_y_with_interf, batch_h_hat, batch_err_var, batch_N0])
        batch_llr = self._config._demapper([batch_x_hat, batch_no_eff])
        b_hat = self._config._decoder(batch_llr)

        # get the true bits
        b = self._batched_data[batch_id]['b']

        # read the next batch in the next call
        self._batch_id += 1

        return b, b_hat


class EvalBsRx(tf.keras.Model):
    def __init__(self,
                 perfect_csi: bool,
                 config: BasicConfig, # need to tune the link-level config
                 ebNo_dB_range: np.ndarray,
                 result_save_path: list,
                 batch_size: int=64,
                 num_target_block_errors: int=1000,
                 max_mc_iter: int=200, 
                 eval_data_paths: list = None,
                 load_ratio: float = 0.7):
        
        super(EvalBsRx, self).__init__(name='EvalBsRx')
        self._eval_obj = _EvalBsRx(perfect_csi=perfect_csi, 
                                   config=config,
                                   eval_data_paths=eval_data_paths,
                                   load_ratio=load_ratio)
        self._ebNo_dB_range = ebNo_dB_range
        self._result_save_path = result_save_path
        
        self._batch_size = batch_size
        self._num_target_block_errors = num_target_block_errors
        self._max_mc_iter = max_mc_iter

    def load_data_pkl(self):
        self._eval_obj.load_data_pkl()

    def eval(self):
        ber, bler = sionna.utils.sim_ber(self._eval_obj,
                                      self._ebNo_dB_range,
                                      batch_size=self._batch_size,
                                      num_target_block_errors=self._num_target_block_errors,
                                      max_mc_iter=self._max_mc_iter,
                                      verbose=False)
        ber = ber.numpy()
        bler = bler.numpy()
        with open(self._result_save_path[0], "wb") as file:
            pickle.dump(ber, file)
        with open(self._result_save_path[1], "wb") as file:
            pickle.dump(bler, file)
        return ber, bler