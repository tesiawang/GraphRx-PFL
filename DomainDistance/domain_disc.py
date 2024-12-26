import os
import sys
import pickle
import numpy as np
from itertools import combinations, chain
from pathlib import Path
import wandb

# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import sionna as sn

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from DomainDistance.mlp_model import MLPDiscriminator


class DiscEstimator:
    def __init__(self, num_clients, feature_dim, lr, label_dim=None, c=0.01):
        '''
        If no label is used, we only need to specify the feature_dim.
        clip parameter c is only required when calculating the Wasserstein distance.
        '''
        self._feature_dim = feature_dim
        self._model = MLPDiscriminator(feature_dim, label_dim, c)
        self._num_clients = num_clients
        self._lr = lr
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # Adam optimizer


    def load_data_pkl(self, data_paths, mode="train"):
        train_eval_split = 0.8
        batched_data = []
        if mode == "train":
            for path in data_paths:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                batched_data.append(data[:int(train_eval_split*len(data))])
            batched_data = list(chain.from_iterable(batched_data))
            # Data structure of batched_data: [{},{},{},{}]
        elif mode == "eval":
            for path in data_paths:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                batched_data.append(data[int(train_eval_split*len(data)):])
            batched_data = list(chain.from_iterable(batched_data))

        else:
            raise ValueError("mode must be either 'train' or 'eval'")
        return batched_data


    def get_model_input(self, one_batch_data):
            '''
            # Input shapes: One batch data as a dictionary
            #   shape of batch_pilots_rg (complex): [batch_size, num_ofdm_symbols, fft_size, NUM_TX_ANTENNAS]
            #   shape of batch_y (complex): [batch_size, num_ofdm_symbols, fft_size, NUM_RX_ANTENNAS]
            #   shape of batch_N0 (float): [batch_size]
            '''
            batch_pilots_rg = one_batch_data['batch_pilots_rg']
            batch_y_with_interf = one_batch_data['batch_y_with_interf']
            batch_N0 = one_batch_data['batch_N0']

            # change dims of batch_N0
            batch_N0 = sn.utils.log10(batch_N0)
            batch_N0 = sn.utils.insert_dims(batch_N0, 3, 1)
            batch_N0 = tf.tile(batch_N0, [1, batch_y_with_interf.shape[1],batch_y_with_interf.shape[2], 1]) # [64,14,276,1]
            
            # here is very important: 
            # model_input : [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 2*num tx antenna + 1]
            model_input = tf.concat([tf.math.real(batch_y_with_interf),
                                    tf.math.imag(batch_y_with_interf),
                                    tf.math.real(batch_pilots_rg),
                                    tf.math.imag(batch_pilots_rg),
                                    batch_N0], axis=-1)
            # # the product of all the dimensions except the first one is the feature dimension
            # feature_dim = model_input.shape[1:]
            return model_input
        

    # @tf.function
    def train_one_pair_model(self, loss_func, metric_func, batch_size, data_path_i, data_path_j):
        ''' 
        This function trains the MLP discriminator to distinguish between two domains i and j. ONLY FOR ONE EPOCH
        we assign client i with domain 0 and client j with domain 1.
        '''
        total_examples, total_loss, total_metric = 0, 0, 0

        batched_data_i = self.load_data_pkl(data_path_i, "train") 
        batched_data_j = self.load_data_pkl(data_path_j, "train")
        num_batch_i = len(batched_data_i)
        num_batch_j = len(batched_data_j)

        # update the model over all the batches for one epoch
        for iter_id in range(max(num_batch_i,num_batch_j)):
            
            # ---------------------------------------------------------------------------- #
            # update the discriminator with domain 0 (client i) data
            input_i = self.get_model_input(batched_data_i[iter_id % num_batch_i])
        
            # update the discriminator with domain 1 (client j) data
            input_j = self.get_model_input(batched_data_j[iter_id % num_batch_j])

            with tf.GradientTape() as tape:
                logits_i = self._model(input_i)
                domain_i = tf.ones_like(logits_i) * int(0) 
                loss_i = loss_func(domain_i, logits_i) # the order is critical for this function: loss_func(y_true, y_pred)

                logits_j = self._model(input_j)
                domain_j = tf.ones_like(logits_j) * int(1)
                loss_j =loss_func(domain_j, logits_j)   

                true_batch_loss = (loss_i + loss_j)/2.0

            grads = tape.gradient(true_batch_loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

            # ---------------------------------------------------------------------------- #
            # compute metric which is binary accuracy
            metric_i = metric_func(logits_i, domain_i) # mean-metric of one batch
            metric_j = metric_func(logits_j, domain_j) # mean-metric of one batch
           
            total_examples += batch_size * 2
            total_loss += (loss_i + loss_j) * batch_size
            total_metric += (metric_i + metric_j) * batch_size
        
        avg_loss_ij = total_loss / total_examples
        avg_metric_ij = total_metric / total_examples

        # the avg_metric_ij's range is already [0,1]
        return avg_loss_ij, avg_metric_ij
    
    
    # @tf.function
    def eval_one_pair_model(self, loss_func, metric_func, batch_size, data_path_i, data_path_j):
        total_examples, total_loss, total_metric = 0, 0, 0
        batched_data_i = self.load_data_pkl(data_path_i, "eval")
        batched_data_j = self.load_data_pkl(data_path_j, "eval")
        num_batch_i = len(batched_data_i)
        num_batch_j = len(batched_data_j)

        # update the model over all the batches for once
        for iter_id in range(max(num_batch_i,num_batch_j)):
            
            # ---------------------------------------------------------------------------- #
            # update the discriminator with domain 0 (client i) data
            input_i = self.get_model_input(batched_data_i[iter_id % num_batch_i])
        
            # update the discriminator with domain 1 (client j) data
            input_j = self.get_model_input(batched_data_j[iter_id % num_batch_j])
        
            # compute the loss and metric (binary accuracy)
            logits_i = self._model(input_i)
            domain_i = tf.ones_like(logits_i) * int(0) 
            loss_i = loss_func(domain_i, logits_i)

            logits_j = self._model(input_j)
            domain_j = tf.ones_like(logits_j) * int(1)
            loss_j =loss_func(domain_j, logits_j)   

            metric_i = metric_func(logits_i, domain_i)
            metric_j = metric_func(logits_j, domain_j)
           
            total_examples += batch_size * 2
            total_loss += (loss_i + loss_j) * batch_size
            total_metric += (metric_i + metric_j) * batch_size
        
        avg_loss_ij = total_loss / total_examples
        avg_metric_ij = total_metric / total_examples

        # the avg_metric_ij's range is already [0,1]
        return avg_loss_ij, avg_metric_ij
    
    

    def estimate_all(self, num_max_epochs, eval_every, early_stop, loss_func, metric_func,client_data_paths):
        '''
        This function estimates the domain distance between all pairs of clients.
        This function calls the 'train_one_pair_model' function for each pair of clients.
        '''
        # intialization
        disc_matrix = np.zeros((self._num_clients, self._num_clients))
        batch_size = 32

        # build the self._model
        self._model.build((batch_size, *self._feature_dim))

        # Iterate every pair of clients:
        for i, j in combinations(range(self._num_clients),2):
            print("Estimating domain divergence between client ", i, " and client ", j)
            
            best_metric = - np.inf
            epochs_no_improve = 0
            # client_i, client_j = i,j 

            for e in range(num_max_epochs):
                print("Epoch {}:".format(e))
                # training pair-wise discriminator
                train_loss, train_metric = self.train_one_pair_model(loss_func, metric_func, batch_size, client_data_paths[i], client_data_paths[j])
                print("===> train_loss: ", train_loss.numpy(), "train_metric: ", train_metric.numpy())
                wandb.log({f"train_loss_{i}{j}": train_loss.numpy(), f"train_metric_{i}{j}": train_metric.numpy(), f"epoch_{i}{j}":e}, commit=False)

                # evaluation
                if e % eval_every == 0:
                    eval_loss, eval_metric = self.eval_one_pair_model(loss_func, metric_func, batch_size, client_data_paths[i], client_data_paths[j])
                    print("===> eval_loss: ", eval_loss.numpy(), "eval_metric: ", eval_metric.numpy())
                    wandb.log({f"eval_loss_{i}{j}": eval_loss.numpy(), f"eval_metric_{i}{j}": eval_metric.numpy(), f"epoch_{i}{j}":e}, commit=False)
                
                # early stopping
                if eval_metric > best_metric:
                    best_metric = eval_metric
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve > early_stop:
                        break
                wandb.log({f"best_metric_{i}{j}": best_metric.numpy(), f"epoch_{i}{j}":e}, commit=True) 
            
            final_metric = best_metric
            print("The estimated divergence between client ", i, " and client ", j, " is: ", final_metric.numpy())
            # metric = metric - 1
            disc_matrix[i,j] = final_metric
            disc_matrix[j,i] = final_metric

        return disc_matrix
