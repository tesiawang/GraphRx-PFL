# -*- coding: utf-8 -*-
import time
import pickle
import sionna
import tensorflow as tf
import numpy as np
from pathlib import Path
from Data.get_config import BasicConfig
from itertools import chain


class TestMainNet:
    def __init__(self,
                 entire_main_net: tf.keras.Model,
                 test_data_paths: str,
                 load_ratio: float = 0.3,
                 config: BasicConfig = BasicConfig()
                 ):

        # Construct the main demapper
        self._entire_main_net = entire_main_net
        self._rg_demapper = sionna.ofdm.ResourceGridDemapper(config._rg, config._sm)
        self._decoder = config._decoder
        self._n = config._n
        self._test_data_paths  = test_data_paths
        self._test_loss = 0.
        self._test_ber = 0.
        self._load_ratio = load_ratio
        self._config = config
        self.num_ber_batches = 150 # reduce test batches
        # self._record_result = record_result

    @tf.function
    def test_one_batch(self,
                        batch_pilots_rg: np.ndarray,
                        batch_y_with_interf: np.ndarray,
                        batch_N0: np.ndarray,
                        tx_codeword_bits: np.ndarray,
                       ):
        # shape of batch_pilots_rg (complex): [batch_size, num_ofdm_symbols, fft_size, 1]
        # shape of batch_y (complex): [batch_size, num_ofdm_symbols, fft_size, 1]
        # shape of batch_N0 (float): [batch_size]
        # shape of tx_codeword_bits (float): [batch_size, NUM_TX, num_streams_per_tx, n]
        batch_llr = self._entire_main_net([batch_pilots_rg, batch_y_with_interf, batch_N0])
        batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
        batch_llr = self._rg_demapper(batch_llr)
        batch_llr = tf.reshape(batch_llr, [tx_codeword_bits.shape[0], 1, 1, self._n])
        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(tx_codeword_bits, batch_llr)
        batch_loss = tf.reduce_mean(batch_loss)

        # --------------------------- compute the BMD rate --------------------------- #
        # BMD_rate = tf.constant(1.0, tf.float32) - batch_loss/tf.math.log(2.)
        # print(f"BMD Rate: {BMD_rate:.2E} bit")
        
        # ------------------------- compute the weighted loss ------------------------ #
        # batch_loss_weighted = batch_loss * tf.math.log(1+1/batch_N0[0])/tf.math.log(2)
        return batch_loss


    @tf.function
    def test_one_batch_ber(self, test_ebno_db,
                            batch_pilots_rg,
                            batch_y_with_interf,
                            tx_codeword_bits,
                            batch_h_freq,
                            interf_batch_y):

        # generate new noise variance based on a single SNR value
        batch_N0 = sionna.utils.ebnodb2no(test_ebno_db,
                                          self._config._num_bits_per_symbol,
                                          self._config._coderate,
                                          self._config._rg)
        # get the new y with the test SNR
        batch_x = self._config._mapper(tx_codeword_bits)
        batch_x_rg = self._config._rg_mapper(batch_x) 
        interf_batch_y = tf.cast(interf_batch_y, tf.complex64)  # change the type of interf_batch_y (float32) to complex64

        # TODO: is it reasonable??? shared channel and interference data
        batch_y_with_interf = self._config._channel_freq([batch_x_rg, batch_h_freq, batch_N0]) + interf_batch_y

        # reshape data
        batch_N0 = tf.expand_dims(batch_N0, axis=0)
        batch_N0 = tf.tile(batch_N0, [batch_y_with_interf.shape[0]])
        batch_y_with_interf = tf.squeeze(batch_y_with_interf, axis=1)
        batch_y_with_interf = tf.transpose(batch_y_with_interf, [0,2,3,1])
        batch_x_rg = tf.squeeze(batch_x_rg, axis=1)
        batch_x_rg = tf.transpose(batch_x_rg, [0,2,3,1])

        # compute b_hat using neural Rx's output
        batch_llr = self._entire_main_net([batch_pilots_rg, batch_y_with_interf, batch_N0])
        batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
        batch_llr = self._rg_demapper(batch_llr)
        batch_llr = tf.reshape(batch_llr, [tx_codeword_bits.shape[0], 1, 1, self._n])
        b_hat = self._config._decoder(batch_llr) # LDPC decoding for bit estimaton

        # ---------------------------------------------------------------------------- #
        return b_hat


    def load_data_pkl(self):
        batched_data = []
        for path in self._test_data_paths:
            with open(path, "rb") as f:
                data = pickle.load(f)
            batched_data.append(data[int((1-self._load_ratio)*len(data)):]) # load the last 30% of the data
        batched_data = list(chain.from_iterable(batched_data))
        # Data structure of batched_data: [{},{},{},{}]
        return batched_data



    def test(self, batched_data):
        
        # test_rate = 0.
        if batched_data is None:
            batched_data = self.load_data_pkl()
        
        # shuffle the data
        np.random.shuffle(batched_data)

        # ------------- compute average batch loss over all test batches ------------- #
        test_loss = 0.
        for batch_id in range(len(batched_data)):
            # Load batch data
            batch_pilots_rg = batched_data[batch_id]['batch_pilots_rg']
            batch_y_with_interf = batched_data[batch_id]['batch_y_with_interf']
            batch_N0 = batched_data[batch_id]['batch_N0']
            tx_codeword_bits = batched_data[batch_id]['tx_codeword_bits']

            # time_start = time.time()
            batch_loss = self.test_one_batch(batch_pilots_rg,
                                            batch_y_with_interf,
                                            batch_N0,
                                            tx_codeword_bits)
            # time_end = time.time()
            # print("Time for testing one batch loss: {:.3f}".format(time_end-time_start))
            test_loss += batch_loss
            # test_rate += BMD_rate

        self._test_loss =  test_loss/len(batched_data)  # this is the average batch loss on one batch
        # self._test_rate = test_rate/len(batched_data) # this is the average BMD rate over all the batches

        # ---------- compute ber over 200 batches on a single high-SNR point --------- #
        test_ebno_db = 4 # only high SNR point at 4 dB
        test_ber = 0.
        for batch_id in range(self.num_ber_batches):
            real_batch_id = batch_id % len(batched_data)
            batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits, batch_h_freq, interf_batch_y, b = batched_data[real_batch_id].values()
            # time_start = time.time()
            b_hat = self.test_one_batch_ber(test_ebno_db,
                                            batch_pilots_rg,
                                            batch_y_with_interf,
                                            tx_codeword_bits,
                                            batch_h_freq,
                                            interf_batch_y)
            # time_end = time.time()
            # print("Time for testing one batch ber: {:.3f}".format(time_end-time_start))
            batch_ber = sionna.utils.compute_ber(b, b_hat)
            test_ber += batch_ber
        self._test_ber = test_ber/self.num_ber_batches

        return self._test_loss, self._test_ber
    
    
    def set_test_data_paths(self, new_test_data_paths):
        self._test_data_paths = new_test_data_paths
    

    def set_model_weights(self, weights):
        self._entire_main_net.set_weights(weights)



# add generalization tests, i.e., how the local model performs on other clients' data
def multi_client_global_model_test(test_obj: TestMainNet,
                                    latest_global_model_parameters, 
                                    client_data_paths):
    
    num_total_clients = len(client_data_paths)
    test_loss_dict = dict()
    test_ber_dict = dict()
    avg_test_loss = 0.
    avg_test_ber = 0.

    for client_id in range(num_total_clients):

        test_obj.set_test_data_paths(client_data_paths[client_id])
        batched_test_data = test_obj.load_data_pkl()

        # set the model weights as the current local model weights
        test_obj.set_model_weights(latest_global_model_parameters)
        test_loss, test_ber = test_obj.test(batched_test_data)
        print("Finish testing for client {:d}: test loss: {:.4f}, test BER: {:.6f}".format(client_id, test_loss, test_ber))

        # add the test loss and test BER to the dict
        test_loss_dict[client_id] = test_loss
        test_ber_dict[client_id] = test_ber
        
    # calculate the average test loss and test BER
    avg_test_loss = sum(test_loss_dict.values())/num_total_clients
    avg_test_ber  = sum(test_ber_dict.values())/num_total_clients
    
    return test_loss_dict, test_ber_dict, avg_test_loss, avg_test_ber



def multi_client_local_model_test(test_obj:TestMainNet,
                                  local_weights, 
                                  client_data_paths):
    
    num_total_clients = len(client_data_paths)  
    test_loss_dict = dict()
    test_ber_dict = dict()
    avg_test_loss = 0.
    avg_test_ber = 0.

    for client_id in range(num_total_clients):

        test_obj.set_test_data_paths(client_data_paths[client_id])
        batched_test_data = test_obj.load_data_pkl()

        # set the model weights as the current local model weights
        test_obj.set_model_weights(local_weights[client_id])
        test_loss, test_ber = test_obj.test(batched_test_data)
        print("Finish testing for client {:d}: test loss: {:.4f}, test BER: {:.6f}".format(client_id, test_loss, test_ber))
        
        # add the test loss and test BER to the dict
        test_loss_dict[client_id] = test_loss
        test_ber_dict[client_id] = test_ber

    # calculate the average test loss and test BER
    avg_test_loss = sum(test_loss_dict.values())/num_total_clients
    avg_test_ber  = sum(test_ber_dict.values())/num_total_clients
        

    return test_loss_dict, test_ber_dict, avg_test_loss, avg_test_ber
        