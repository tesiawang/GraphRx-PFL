# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from Data.get_config import BasicConfig
import sionna
import time

class DataGenerator:
    def __init__(self, 
                 init_config: BasicConfig,
                 ebNo_dB_range: np.ndarray, 
                 SIR_dB_range: np.ndarray,
                 apply_encoder:bool,
                 add_interference) -> None:
        self._config = init_config
        self._ebNo_dB_range = ebNo_dB_range
        self._SIR_db_range = SIR_dB_range
        self._apply_encoder = apply_encoder
        self._add_interference = add_interference
    
    def reset_link_config(self, link_config: BasicConfig) -> None:
        self._config = link_config

    def reset_snr_range(self, min_ebno: float, max_ebno:float, num_ebno_point: int) -> None: # for generating data per snr point
        self._ebNo_dB_range = np.linspace(min_ebno, max_ebno, num_ebno_point)

    @tf.function
    def _receive_data(self,
                      ebno_db,
                      SIR_db,
                      batch_size) -> tuple:

        # -------------------------------- Transmitter ------------------------------- #
        batch_N0 = sionna.utils.ebnodb2no(ebno_db,
                                          self._config._num_bits_per_symbol,
                                          self._config._coderate,
                                          self._config._rg)
        
        if self._apply_encoder == True:
            # use the outer encoder during training 
            b = self._config._binary_source([batch_size, 1, self._config._num_streams_per_tx, self._config._k])
            tx_codeword_bits = self._config._encoder(b)
        else:
            #to reduce the computational complexity, the outer encoder (and decoder) are not used at training
            tx_codeword_bits = self._config._binary_source([batch_size, 1, self._config._num_streams_per_tx, self._config._n])


        batch_x = self._config._mapper(tx_codeword_bits)
        batch_x_rg = self._config._rg_mapper(batch_x) # batch_x_rg contains both the pilot and data symbols
        # batch_pilots_rg = mask*batch_x_rg  # get pilot configurations
        
        # ---------------------------------------------------------------------------- #

        # ---------------------------- Through the Channel --------------------------- #
        cir = self._config._comm_channel_model(batch_size, self._config._rg.num_ofdm_symbols, 1/self._config._rg.ofdm_symbol_duration)
        batch_h_freq = sionna.channel.cir_to_ofdm_channel(self._config._frequencies, *cir, normalize=True) # this is real channel freq response
        batch_y = self._config._channel_freq([batch_x_rg, batch_h_freq, batch_N0])


        # add the interfering signal to the original signal
        if self._add_interference == True:
             # simulate another interfering signal and through the interfering channel 
            interf_codeword_bits = self._config._binary_source([batch_size, 1, self._config._num_streams_per_tx, self._config._n])
            interf_batch_x = self._config._mapper(interf_codeword_bits)
            interf_batch_x_rg = self._config._rg_mapper(interf_batch_x)
            interf_cir = self._config._interf_channel_model(batch_size, self._config._rg.num_ofdm_symbols, 1/self._config._rg.ofdm_symbol_duration)
            interf_batch_h_freq = sionna.channel.cir_to_ofdm_channel(self._config._frequencies, *interf_cir, normalize=True)
            interf_batch_y = self._config._interf_channel_freq([interf_batch_x_rg, interf_batch_h_freq]) # no additive noise

            pi_over_ps = 10**( -SIR_db/10.)
            interf_batch_y = tf.cast(tf.sqrt(pi_over_ps), tf.complex64)  * interf_batch_y
            batch_y_with_interf = batch_y +  interf_batch_y

        else:
            interf_batch_y = tf.zeros(batch_y.shape) # could be set as None 
            batch_y_with_interf = batch_y

        # ---------------------------------------------------------------------------- #
        # batch_h_ls_est, batch_var_ls_est = self._config._ls_est([batch_y, batch_N0])

        # ----------------Shape of the following parameters: --------------- #
        # shape of batch_x_rg (complex): [batch_size, 1, 1, num_ofdm_symbols, fft_size]
        # shape of batch_y (complex): [batch_size, 1, 1, num_ofdm_symbols, fft_size]
        # shape of batch_N0 (float): [batch_size]
        # shape of batch_h_freq (complex): [batch_size, NUM_RX, 1, NUM_TX, 1, num_ofdm_symbols, fft_size]
        # shape of batch_h_ls_est (complex): [batch_size, NUM_RX, 1, NUM_TX, 1, num_ofdm_symbols, efficient_fft_size]
        # shape of batch_var_ls_est (float): [1, 1, 1, 1, 1, num_ofdm_symbols, efficient_fft_size]


        # ------------------------------ Extract pilots ------------------------------ #
        batch_N0 = tf.expand_dims(batch_N0, axis=0)
        batch_N0 = tf.tile(batch_N0, [batch_y_with_interf.shape[0]]) # save the noise variance, but not the real noise power realization
        batch_y_with_interf = tf.squeeze(batch_y_with_interf, axis=1)
        batch_y_with_interf = tf.transpose(batch_y_with_interf, [0,2,3,1])
        batch_x_rg = tf.squeeze(batch_x_rg, axis=1)
        batch_x_rg = tf.transpose(batch_x_rg, [0,2,3,1])

        mask = np.zeros((self._config._num_ofdm_symbols, self._config._fft_size, self._config._num_ut_ant), dtype=np.bool8)
        for idx in self._config._pilot_ofdm_symbol_indices:
            mask[idx,:,:] = True
        mask = tf.constant(mask, dtype=tf.complex64)
        batch_pilots_rg = mask*batch_x_rg  # get pilot configurations
        # ---------------------------------------------------------------------------- #

        return batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits, batch_h_freq, interf_batch_y, b


        # shape of batch_pilots_rg (complex): [batch_size, num_ofdm_symbols, fft_size, num_tx_ant]
        # shape of batch_y (complex): [batch_size, num_ofdm_symbols, fft_size, num_rx_ant] 
        # shape of batch_N0 (float): [batch_size]
        # shape of tx_codeword_bits (float): [batch_size, NUM_TX, num_streams_per_tx, n]
        # shape of batch_h_freq (complex): [batch_size, NUM_RX, 1, NUM_TX, 1, num_ofdm_symbols, fft_size], fft_size = number of subcarriers???
        # shape of interf_batch_y (complex): [batch_size, 1, num_rx_ant, num_ofdm_symbols, fft_size]
        # shape of b (float): [batch_size, 1, 1, k]
    

    def receive_data(self, batch_size:int, num_batch:int) -> tuple:
        # generate data on the full SNR range
        channel_realizations_per_config = []

        # to avoid expensive retracing: move the loop in the class function
        for _ in range(num_batch):
            # start = time.time()
            # set one SNR point for a batch of data!!!
            ebno_db = float(np.random.choice(self._ebNo_dB_range, 1)) # randomly choose an SNR value
            SIR_db = float(np.random.choice(self._SIR_db_range, 1)) # randomly choose an SIR value
            batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits, batch_h_freq, interf_batch_y, b = self._receive_data(ebno_db=ebno_db,
                                                                                                                                  SIR_db = SIR_db,
                                                                                                                                  batch_size=batch_size)
            # end = time.time()
            # print("Time to generate one batch: ", end - start)
            with tf.device('cpu:0'):
                realization = dict()
                realization['batch_pilots_rg'] = tf.identity(batch_pilots_rg)
                realization['batch_y_with_interf'] = tf.identity(batch_y_with_interf)
                realization['batch_N0'] = tf.identity(batch_N0)
                realization['tx_codeword_bits'] = tf.identity(tx_codeword_bits)
                realization['batch_h_freq'] = tf.identity(batch_h_freq)
                realization['interf_batch_y'] = tf.identity(interf_batch_y)
                realization['b'] = tf.identity(b)

            channel_realizations_per_config.append(realization)

        return channel_realizations_per_config