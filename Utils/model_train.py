# -*- coding: utf-8 -*-
import time
import pickle
import sionna
import tensorflow as tf
import numpy as np
from Data.get_config import BasicConfig
from Utils.functions import log_print, flatten_model
from itertools import chain
import wandb
from sionna.utils import expand_to_rank, complex_normal



# noinspection PyCallingNonCallable
class TrainEntireMainNet:
    def __init__(self,
                 entire_main_net: tf.keras.Model,
                 lr: float,
                 train_data_paths: list,
                 load_ratio: float,
                 config: BasicConfig,
                 apply_aug: bool = False,
                 aug_times: int = 1):
        
        # Construct the main demapper
        self._entire_main_net = entire_main_net
        self._rg_demapper = sionna.ofdm.ResourceGridDemapper(config._rg, config._sm)
        self._n = config._n
        self._config = config
        self._Es = 1.425

        # Training parameters
        self._lr = lr
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # Adam optimizer

        # other
        self._train_data_paths = train_data_paths
        self._train_one_batch_time = 0.
        self._epoch_loss_list = []
        self._avg_batch_loss_list = []
        self._load_ratio = load_ratio
        self._apply_aug = apply_aug
        self._aug_times = aug_times
        # self._real_dtype = tf.dtypes.as_dtype(self._dtype).real_dtype


    def set_train_data_paths(self, train_data_paths):
        self._train_data_paths = train_data_paths
        
        
    def set_model_weights(self, weights):
        self._entire_main_net.set_weights(weights)

    
    def load_data_pkl(self):
        batched_data = []
        with tf.device('CPU:0'):
            for path in self._train_data_paths:
                with open(path, "rb") as f:
                    data = pickle.load(f) # this step does: copy the data pickled in cpu to gpu memory; all the data samples are on the gpus...

                # if the data is on the gpu when pickled, then the data is still on the gpu after unpickled
                batched_data.append(data[:int(self._load_ratio*len(data))]) # [[{}],[{}],[{}]]
        batched_data = list(chain.from_iterable(batched_data))
        # Data structure of batched_data: [{},{},{},{}]
        return batched_data


    # ---------------------------------------------------------------------------- #
    #                         Normal batch-training process                        #
    # ---------------------------------------------------------------------------- #
    @tf.function
    def train_one_batch(self,
                        batch_pilots_rg: np.ndarray,
                        batch_y_with_interf: np.ndarray,
                        batch_N0: np.ndarray,
                        tx_codeword_bits: np.ndarray,
                        update_weights = True,
                        ebno_linear = 1.0): # default value if not provided
        # shape of batch_pilots_rg (complex): [batch_size, num_ofdm_symbols, fft_size, 1]
        # shape of batch_y (complex): [batch_size, num_ofdm_symbols, fft_size, num_bs_ant]
        # shape of batch_N0 (float): [batch_size]
        # shape of tx_codeword_bits (float): [batch_size, NUM_TX, num_streams_per_tx, n]
        
        with tf.GradientTape() as tape:
            batch_llr = self._entire_main_net([batch_pilots_rg, batch_y_with_interf, batch_N0])
            batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
            batch_llr = self._rg_demapper(batch_llr)
            batch_llr = tf.reshape(batch_llr, [tx_codeword_bits.shape[0], 1, 1, self._n])
            batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(tx_codeword_bits, batch_llr) # use the labels to compute loss
            batch_loss = tf.reduce_mean(batch_loss)

            ### Use the shannon achievable rate to weight the loss
            # batch_loss_weighted = batch_loss * tf.math.log(1+ebno_linear)/tf.math.log(2.)

        # Computing and applying gradientss
        grads = tape.gradient(batch_loss, self._entire_main_net.trainable_variables)
        if update_weights == True:
            self._optimizer.apply_gradients(zip(grads, self._entire_main_net.trainable_variables))

        return batch_loss, grads


    # augment and train one batch out of the batched data
    @tf.function
    def augment_and_train_one_batch(self,
                                    batch_y_with_interf: tf.Tensor,
                                    batch_pilots_rg: tf.Tensor,
                                    tx_codeword_bits: tf.Tensor,
                                    batch_N0: tf.Tensor,
                                    var_range: tf.Tensor):
        
        aug_var_N0 = tf.random.uniform(shape = [1], minval=0, maxval=var_range, dtype=tf.float32) # generate a float scalar uniformly from [0, var_range]
        aug_h_phi = tf.random.uniform(shape = [1], minval=0, maxval=2*np.pi, dtype=tf.float32)

        aug_batch_N0 = tf.tile(aug_var_N0, [batch_y_with_interf.shape[0]]) # save the noise variance, but not the real noise power realization
        aug_batch_N0 +=  batch_N0

        # ------------------- Augment the batched data ------------------- #
        # Rotate the channel
        aug_h_phi = tf.cast(aug_h_phi, batch_y_with_interf.dtype) #tf.complex64
        aug_batch_y =  tf.exp(1j * aug_h_phi) * batch_y_with_interf
        
        # Add noise
        aug_var_N0 = expand_to_rank(aug_var_N0, tf.rank(aug_batch_y), axis=-1) # expand the scalar to the same shape as aug_batch_y
        aug_var_N0 = tf.cast(aug_var_N0, tf.float32)
        aug_batch_noise = complex_normal(tf.shape(aug_batch_y), dtype=aug_batch_y.dtype)
        aug_batch_noise *= tf.cast(tf.sqrt(aug_var_N0), aug_batch_noise.dtype)
        aug_batch_y = aug_batch_y + aug_batch_noise

        # ---------------------------------------------------------------------------- #
        batch_loss, _ = self.train_one_batch(batch_pilots_rg,
                                             aug_batch_y,
                                             aug_batch_N0,
                                             tx_codeword_bits,
                                             update_weights=True)
        return batch_loss


    def train(self, num_epochs, batched_data):

        epoch_loss_list = []
        if batched_data is None:
            batched_data = self.load_data_pkl()

        # First load all the channel config data into one list, then load batch by batch.
        # Data structure of batched_data: [{},{},{},{}]
        np.random.shuffle(batched_data)

        for epoch in range(num_epochs):
            # time_start = time.time()
            epoch_loss = 0.
            
            # batched_data[batch_id] is a dictionary with keys, then we unpack the dictionary to get the values for one single batch
            for batch_id in range(len(batched_data)): # E.g., 200 Batches of data
                
                # unpack the dictionary
                batch_pilots_rg = batched_data[batch_id]['batch_pilots_rg']
                tx_codeword_bits = batched_data[batch_id]['tx_codeword_bits']
                batch_y_with_interf = batched_data[batch_id]['batch_y_with_interf']
                batch_N0 = batched_data[batch_id]['batch_N0']

                # train the original true batch
                ebno_linear = self._Es/(batch_N0[0]* self._config._num_bits_per_symbol * self._config._coderate)
                batch_loss, _ = self.train_one_batch(batch_pilots_rg,
                                                    batch_y_with_interf,
                                                    batch_N0,
                                                    tx_codeword_bits,
                                                    update_weights=True,
                                                    ebno_linear=ebno_linear)
                # print("Original batch loss: ", batch_loss.numpy())
                
                epoch_loss += batch_loss

                # ---------------------------------------------------------------------------- #
                if self._apply_aug:
                    # start augmentation 
                    for a in range(self._aug_times - 1):
                        # randomly choose the noise power and the channel phase
                        # original noise power = N0, augmented noise power = N0/2
                        var_range = batch_N0[0] / 2.0
                        batch_loss = self.augment_and_train_one_batch(batch_y_with_interf,
                                                                      batch_pilots_rg,
                                                                      tx_codeword_bits,
                                                                      batch_N0,
                                                                      var_range)
                        epoch_loss += batch_loss

            
            # ---------------------------------------------------------------------------- #
            # Memory Use after one epoch 
            # print(" --------------After one epoch ----------")
            # print("CPU memory current usage:{} GB, peak usage {} GB".format(tf.config.experimental.get_memory_info("CPU:0")['current']/1e9,
            #                                                                            tf.config.experimental.get_memory_info("CPU:0")['peak']/1e9))
            # print("GPU memory current usage:{} GB, peak usage {} GB".format(tf.config.experimental.get_memory_info("GPU:0")['current']/1e9,
            #                                                                            tf.config.experimental.get_memory_info("GPU:0")['peak']/1e9))
            # print(" ----------------------------------------")
            

            epoch_loss_list.append(epoch_loss/(len(batched_data)*self._aug_times)) # average batch loss over one batch
            if num_epochs > 0:
                log_print("====> Training Process, Epoch %d, Train Loss: %.4f" % (epoch, epoch_loss_list[-1]), color='g')

        self._epoch_loss_list = epoch_loss_list
        
        # return the trained weights in the last epoch
        return self._entire_main_net.get_weights().copy(), self._epoch_loss_list[-1]


    # ---------------------------------------------------------------------------- #
    #                           PFLGraph Training Process                          #
    # ---------------------------------------------------------------------------- #
    @tf.function
    def train_graph_one_batch(self, batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits, lam, flat_weighted_model, flat_local_model, round, penalty):
        
        # with open('/home/pFedRx/weights_before_batch1_run2.pkl', 'wb') as f:
        #     pickle.dump(self._entire_main_net.get_weights(), f)

        with tf.GradientTape() as tape:
            batch_llr = self._entire_main_net([batch_pilots_rg, batch_y_with_interf, batch_N0])
            batch_llr = sionna.utils.insert_dims(batch_llr, 2, axis=1)
            batch_llr = self._rg_demapper(batch_llr)
            batch_llr = tf.reshape(batch_llr, [tx_codeword_bits.shape[0], 1, 1, self._n])
            batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(tx_codeword_bits, batch_llr)
            batch_loss = tf.reduce_mean(batch_loss)

            if round > 0: 
                # compute the loss
                # local_model is weighted initialized as the weighted model at each round
                # Thus, in the beginning of each round, local model = weighted model
                # Then, the local model is updated batch by batch, but regularized by the weighted model
                
                # here the negative sign is taken in 'tf.keras.losses.cosine_similarity': un-normed version
                # batch_loss is on the scale of 0.7 in the beginning
                if penalty == 0:
                    batch_loss = batch_loss # without any penalty term
                elif penalty == 1:
                    batch_loss += lam * tf.keras.losses.cosine_similarity(flat_weighted_model, flat_local_model)
                elif penalty == 2:
                    # the input flat_weighted_model =  the (flat-)normalized weighted model
                    batch_loss += lam * (-tf.tensordot(flat_weighted_model,flat_local_model,axes =1)/tf.norm(flat_local_model))
                elif penalty == 3:
                    # the input flat_weighted model  = the normalized model updates
                    batch_loss += lam * (-tf.tensordot(flat_weighted_model, flat_local_model,axes =1)/tf.norm(flat_local_model))
                else:
                    raise ValueError("penalty term indicator should be 0, 1, 2, 3")
                
        # compute grads wrt the loss
        grads = tape.gradient(batch_loss, self._entire_main_net.trainable_variables)
        
        # update the self._entire_main_net
        self._optimizer.apply_gradients(zip(grads, self._entire_main_net.trainable_variables))

        return batch_loss, grads
    

    @tf.function
    def augment_and_train_graph_one_batch(self,
                                          batch_pilots_rg: tf.Tensor,
                                          batch_y_with_interf: tf.Tensor,
                                          batch_N0: tf.Tensor,
                                          tx_codeword_bits: tf.Tensor,
                                          lam,
                                          flat_weighted_model: tf.Tensor,
                                          flat_local_model: tf.Tensor,
                                          round,
                                          penalty,
                                          var_range):
                                        
        aug_var_N0 = tf.random.uniform(shape = [1], minval=0, maxval=var_range, dtype=tf.float32) # generate a float scalar uniformly from [0, var_range]
        aug_h_phi = tf.random.uniform(shape = [1], minval=0, maxval=2*np.pi, dtype=tf.float32)
        aug_batch_N0 = tf.tile(aug_var_N0, [batch_y_with_interf.shape[0]]) # save the noise variance, but not the real noise power realization
        new_batch_N0 = aug_batch_N0 + batch_N0

        # ------------------- Augment the batched data ------------------- #
        # Rotate the channel
        aug_h_phi = tf.cast(aug_h_phi, batch_y_with_interf.dtype) #tf.complex64
        aug_batch_y =  tf.exp(1j * aug_h_phi) * batch_y_with_interf
        
        # Add noise
        aug_var_N0 = expand_to_rank(aug_var_N0, tf.rank(aug_batch_y), axis=-1) # expand the scalar to the same shape as aug_batch_y
        aug_var_N0 = tf.cast(aug_var_N0, tf.float32)
        aug_batch_noise = complex_normal(tf.shape(aug_batch_y), dtype=aug_batch_y.dtype)
        aug_batch_noise *= tf.cast(tf.sqrt(aug_var_N0), aug_batch_noise.dtype)
        aug_batch_y = aug_batch_y + aug_batch_noise

        # ---------------------------------------------------------------------------- #     
        batch_loss, _ = self.train_graph_one_batch(batch_pilots_rg,
                                                    aug_batch_y,
                                                    new_batch_N0,
                                                    tx_codeword_bits,
                                                    lam=lam,
                                                    flat_weighted_model=flat_weighted_model,
                                                    flat_local_model=flat_local_model,
                                                    round=round,
                                                    penalty=penalty) 
        return batch_loss
    

    def train_graph(self,round, local_iter, lam, weighted_model, record_batch_loss_every, batched_data, penalty, initial_para=None):
        '''
        Client local training with or without the penalty term
        penalty = 0: no penalty term
        penalty = 1: model cosine similarity as the penalty term
        penalty = 2: normalized mdoel cosine similarity as the penalty term
        penalty = 3: normalized "model update" (compared to the initial model) cosine similarity as the penalty term
        '''
        batch_loss_list = list()
        avg_batch_loss_list = list()

        if batched_data is None:
            batched_data = self.load_data_pkl()

        np.random.shuffle(batched_data)
        num_batch = len(batched_data)
        
        if penalty == 3: # only need to flatten the inital_para for once
            flat_initial_para = flatten_model(initial_para)

        for iter_id in range(local_iter): # NOTE: be careful of this 'local_iter', it should be set in proportion to the number of local batches
            batch_pilots_rg = batched_data[iter_id % num_batch]['batch_pilots_rg']
            batch_y_with_interf = batched_data[iter_id % num_batch]['batch_y_with_interf']
            batch_N0 = batched_data[iter_id % num_batch]['batch_N0']
            tx_codeword_bits = batched_data[iter_id % num_batch]['tx_codeword_bits']
        
            # get the current local model weights
            local_model = self._entire_main_net.get_weights()
            if penalty == 0 or penalty == 1:
                flat_local_model = flatten_model(local_model)
                flat_weighted_model = flatten_model(weighted_model) # here we consider the unnormalized weighted model
            elif penalty == 2:
                # the input flat_weighted_model = the (flat-)normalized weighted model
                flat_local_model = flatten_model(local_model)
                flat_weighted_model = weighted_model # normalized weighted model
            elif penalty == 3:
                # the input flat_weighted model = the normalized model updates
                flat_local_model = flatten_model(local_model) - flat_initial_para
                flat_weighted_model = weighted_model # normalized weighted updates
            else:
                raise ValueError("penalty should be 0, 1, 2, 3")
            
            # train the un-augmented batch
            batch_loss, _ = self.train_graph_one_batch(batch_pilots_rg,
                                                        batch_y_with_interf,
                                                        batch_N0,
                                                        tx_codeword_bits,
                                                        lam=lam,
                                                        flat_weighted_model=flat_weighted_model,
                                                        flat_local_model=flat_local_model,
                                                        round=round,
                                                        penalty=penalty)
            batch_loss_list.append(batch_loss.numpy())
            # print("Original batch loss: ", batch_loss.numpy())

            # ---------------------------------------------------------------------------- #
            if self._apply_aug:
                # print("Apply Augmentation")
                # start augmentation
                for _ in range(self._aug_times - 1):
                    # randomly choose the noise power and the channel phase
                    # original noise power = N0, augmented noise power = N0/2
                    var_range = batch_N0[0] / 2.0
                    batch_loss = self.augment_and_train_graph_one_batch(batch_pilots_rg,
                                                                        batch_y_with_interf,
                                                                        batch_N0,
                                                                        tx_codeword_bits,
                                                                        lam,flat_weighted_model,flat_local_model,round, penalty,
                                                                        var_range)
                    batch_loss_list.append(batch_loss.numpy())
            
            # ---------------------------------------------------------------------------- #
            # save and return batch loss list
            if (iter_id+1) % record_batch_loss_every == 0 and iter_id > 0:
                record_len = len(batch_loss_list)
                true_num_batch = record_batch_loss_every * self._aug_times
                avg_batch_loss_list.append(sum(batch_loss_list[(record_len-true_num_batch):]) / true_num_batch)
                log_print("====> FedGraph Local Training Process, Local Iter: %d, Loss: %.8f" % (iter_id, avg_batch_loss_list[-1]),
                         color='g')
            
        self._avg_batch_loss_list = avg_batch_loss_list
        return self._entire_main_net.get_weights().copy(), self._avg_batch_loss_list[-1]


    
    # ---------------------------------------------------------------------------- #
    #                                FedRep Training                               #
    # ---------------------------------------------------------------------------- #
    def train_head(self, glob_layers, num_epochs, batched_data):
        for layer_idx in range(len(self._entire_main_net.layers)):
            if layer_idx in glob_layers: # glob_layers = rep_layers
                self._entire_main_net.layers[layer_idx].trainable = False
            else:
                self._entire_main_net.layers[layer_idx].trainable = True

        print("First train personalized classifer layers:")
        # batched_data = none
        trained_weights, train_loss = self.train(num_epochs=num_epochs, batched_data=batched_data)

        # return the trained weights in the last epoch
        return trained_weights, train_loss


    def train_rep(self, glob_layers, num_epochs, batched_data):
        for layer_idx in range(len(self._entire_main_net.layers)):
            if layer_idx not in glob_layers: # glob_layers = rep_layers
                self._entire_main_net.layers[layer_idx].trainable = False
            else:
                self._entire_main_net.layers[layer_idx].trainable = True

        print("Then train shared representation layers:")
        trained_weights, train_loss = self.train(num_epochs=num_epochs, batched_data=batched_data)

        # return the trained weights in the last epoch
        return trained_weights, train_loss


    # ---------------------------------------------------------------------------- #
    #                                Dittos Training                               #
    # ---------------------------------------------------------------------------- #
    def train_ditto(self, local_iter, lam, latest_w_glob, local_lr, record_batch_loss_every, batched_data):
        batch_loss_list = list()  # save the batch loss of local iterations
        avg_batch_loss_list = list()

        if batched_data is None:
            batched_data = self.load_data_pkl()

        np.random.shuffle(batched_data)
        num_batch = len(batched_data)

        for iter_id in range(local_iter):
            
            # each iteration of local training is a modified gradient descent: load one batch data
            batch_pilots_rg = batched_data[iter_id % num_batch]['batch_pilots_rg']
            batch_y_with_interf = batched_data[iter_id % num_batch]['batch_y_with_interf']
            batch_N0 = batched_data[iter_id % num_batch]['batch_N0']
            tx_codeword_bits = batched_data[iter_id % num_batch]['tx_codeword_bits']
            batch_loss, grads = self.train_one_batch(batch_pilots_rg,
                                                    batch_y_with_interf,
                                                    batch_N0,
                                                    tx_codeword_bits,
                                                    update_weights=False)
            
            # update local model parameters, with a common lambda for all clients
            w_local = self._entire_main_net.get_weights()
            for layer_id in range(len(grads)):
                eff_grad = grads[layer_id] + lam * (w_local[layer_id] - latest_w_glob[layer_id])
                w_local[layer_id] = w_local[layer_id] - local_lr * eff_grad

            # update the model weights
            self._entire_main_net.set_weights(w_local)
            batch_loss_list.append(batch_loss.numpy())

            # save and return batch loss list
            if (iter_id+1) % record_batch_loss_every == 0 and iter_id > 0:
                record_len = len(batch_loss_list)
                avg_batch_loss_list.append(sum(batch_loss_list[(record_len-record_batch_loss_every):]) / record_batch_loss_every)
                log_print("====> Ditto Local Training Process, Local Iter: %d, Loss: %.8f" % (iter_id, avg_batch_loss_list[-1]), color='g')

        self._avg_batch_loss_list = avg_batch_loss_list
        return self._entire_main_net.get_weights().copy(), avg_batch_loss_list[-1]


    
