# -*- coding: utf-8 -*-
import time
import pickle
import sionna
import tensorflow as tf
import numpy as np
from Data.get_config import BasicConfig

class CoresetSelector:
    def __init__(self,
                 entire_main_net: tf.keras.Model,
                 config: BasicConfig,
                 buffer_size: int,
                 lr: float = 0.001):
        
        # Construct the main demapper
        self._entire_main_net = entire_main_net
        self._rg_demapper = sionna.ofdm.ResourceGridDemapper(config._rg, config._sm)
        self._n = config._n
        self._config = config
        self._buffer_size = buffer_size # in batches

        # Training parameters
        self._lr = lr
        # self._optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # Adam optimizer


    # ---------------------------------------------------------------------------- #
    #                         Batch-wise coreset selection                         #
    # ---------------------------------------------------------------------------- #
    @tf.function
    def compute_batch_grad(self, batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits):

        with tf.GradientTape() as tape:
            batch_llr = self._entire_main_net([batch_pilots_rg, batch_y_with_interf, batch_N0])
            batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
            batch_llr = self._rg_demapper(batch_llr)
            batch_llr = tf.reshape(batch_llr, [tx_codeword_bits.shape[0], 1, 1, self._n])
            batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(tx_codeword_bits, batch_llr)
            batch_loss = tf.reduce_mean(batch_loss)
    
        
        grads = tape.gradient(batch_loss, self._entire_main_net.trainable_variables)
        grads = [tf.reshape(_g, [-1]) for _g in grads]
        grads = tf.concat(grads, axis=0)
        return grads

    
    def select_coreset_batch_wise(self, batched_data):

        all_batch_grads = []
        for batch_id in range(len(batched_data)):
            # unpack the dictionary
            batch_pilots_rg = batched_data[batch_id]['batch_pilots_rg']
            batch_y_with_interf = batched_data[batch_id]['batch_y_with_interf']
            batch_N0 = batched_data[batch_id]['batch_N0']
            tx_codeword_bits = batched_data[batch_id]['tx_codeword_bits']
            grads = self.compute_batch_grad(batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits)
            all_batch_grads.append(grads)

        avg_all_batch_grads = tf.reduce_mean(all_batch_grads, axis=0)
        all_batch_sim = np.zeros(len(batched_data))
        all_batch_diversity = np.zeros(len(batched_data))
        all_batch_metric = np.zeros(len(batched_data))

        for i in range(len(batched_data)):

            # compute batch affinity
            batch_sim = - tf.losses.cosine_similarity(all_batch_grads[i], avg_all_batch_grads, axis=0)
            batch_diversity = 0.

            # compute sample diversity
            for j in range(len(batched_data)):
                if j != i:
                    batch_diversity += (tf.losses.cosine_similarity(all_batch_grads[i], all_batch_grads[j], axis=0))
            batch_diversity = batch_diversity / (len(batched_data) - 1)

            all_batch_sim[i] = batch_sim
            all_batch_diversity[i] = batch_diversity
            all_batch_metric[i] = batch_sim + batch_diversity
        
        # sort the batch by the metric and record the top-k batch IDs
        sorted_batch_ids = np.argsort(all_batch_metric) # descending order
        return sorted_batch_ids[::-1][:self._buffer_size]
        

    # ---------------------------------------------------------------------------- #
    #                         Sample-wise coreset selection                        #
    # ---------------------------------------------------------------------------- #
    
    @tf.function
    def fast_adapt_one_iter(self, batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits, 
                            update_weights=True,
                            sample_grads=None,
                            one_batch_recorder=None):
        '''
        Update the model with one mini-batch of data
        return the per-sample metric (batch affinity & sample diversity)
        '''

        with tf.GradientTape() as tape:
            batch_llr = self._entire_main_net([batch_pilots_rg, batch_y_with_interf, batch_N0])
            batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
            batch_llr = self._rg_demapper(batch_llr)
            batch_llr = tf.reshape(batch_llr, [tx_codeword_bits.shape[0], 1, 1, self._n])
            batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(tx_codeword_bits, batch_llr)
            sample_losses = tf.reduce_mean(batch_loss, axis=[1, 2, 3])
            batch_loss = tf.reduce_mean(batch_loss)
        
        for sample_id in range(sample_losses.shape[0]):
            # get the gradient for one data point
            g = tape.gradient(sample_losses[sample_id], self._entire_main_net.trainable_variables)
            flatten_g = [tf.reshape(_g, [-1]) for _g in g]
            sample_grads[sample_id] = tf.concat(flatten_g, axis=0)
        
        grads = tape.gradient(batch_loss, self._entire_main_net.trainable_variables)

        # we can just compute the grads above, but do not update the weights
        if update_weights == True:
            self._optimizer.apply_gradients(zip(grads, self._entire_main_net.trainable_variables))

        # compute batch affinity & sample diversity in one mini-batch
        # ...

        return one_batch_recorder

    

    def select_coreset_sample_wise(self, batched_data):
        # set the buffer size upper bound as the multiplies of 'batch_size'
        all_sample_recorder = [] # len = len(batched_data) * batch_size

        for batch_id in range(len(batched_data)):
            # unpack the dictionary
            batch_pilots_rg = batched_data[batch_id]['batch_pilots_rg']
            batch_y_with_interf = batched_data[batch_id]['batch_y_with_interf']
            batch_N0 = batched_data[batch_id]['batch_N0']
            tx_codeword_bits = batched_data[batch_id]['tx_codeword_bits']
            
            # update the model parameters with one batch for one iteration
            one_batch_recorder = np.zeros(batch_y_with_interf.shape[0]) # batch_size
            sample_grads = [None] * batch_y_with_interf.shape[0] # batch_size

            # record the per-point grads, avg grads.
            one_batch_recorder = self.fast_adapt_one_iter(batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits, 
                                                         update_weights=True,
                                                         sample_grads=sample_grads,
                                                         one_batch_recorder=one_batch_recorder)
            all_sample_recorder.append(one_batch_recorder)

            # given the all_sample_recorder, select the top-k samples to store in the buffer
            # ...