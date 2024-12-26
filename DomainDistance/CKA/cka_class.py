import tensorflow as tf
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import numpy as np
import pickle
from itertools import chain

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def load_data_pkl(data_paths):
    batched_data = []
    for path in data_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        batched_data.append(data[:]) 
    batched_data = list(chain.from_iterable(batched_data))
    # Data structure of batched_data: [{},{},{},{}]
    return batched_data

class CKA:
    def __init__(self,
                 model1: tf.keras.Model,
                 model2: tf.keras.Model,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None):
        """
        :param model1: (tf.keras.Model) Neural Network 1
        :param model2: (tf.keras.Model) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2
        
        self.model1_features = []
        self.model2_features = []

        self.model1_layers = model1_layers # [0,1,3]
        self.model2_layers = model2_layers

        # build the model
        # x_ = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_ut_ant), dtype=np.complex64)
        # y_ = np.zeros((1, link_config._num_ofdm_symbols, link_config._fft_size, link_config._num_bs_ant), dtype=np.complex64)
        # n0_ = np.zeros(1, dtype=np.float32)
        # entire_main_net([x_,y_,n0_])
        

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = tf.ones([N, 1])
        result = tf.linalg.trace(tf.matmul(K, L))
        ### pytorch version
        # result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        # result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()

        result += ((tf.transpose(ones) @ K @ ones @ tf.transpose(ones) @ L @ ones) / ((N - 1) * (N - 2))).numpy()
        result -= ((tf.transpose(ones) @ K @ L @ ones) * 2 / (N - 2)).numpy()
        final_result = (1 / (N * (N - 3)) * result).numpy().reshape([-1])
        assert not np.isnan(self.hsic_matrix).any(), "HSIC tensor element computation resulted in NANs"
        
        # finish checking this inner function: the final result is a real scalar
        return final_result

    def compare(self,
                eval_data_path1: List,
                eval_data_path2: List = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param eval_data_path1: 
        :param eval_data_path2: If given, model 2 will run on this dataset. (default = None)
        """

        if eval_data_path2 is None:
            print("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            eval_data_path2 = eval_data_path1

        N = len(self.model1_layers) 
        M = len(self.model2_layers) 
        self.hsic_matrix = np.zeros([N, M, 3])

        eval_batched_data1 = load_data_pkl(eval_data_path1)
        eval_batched_data2 = load_data_pkl(eval_data_path2)

        num_batches = min(len(eval_batched_data1), len(eval_batched_data2))

        # Measure the sim between each pair of examples
        # Get one data sample's input
        for batch_id in range(num_batches):

            self.model1_features = []
            self.model2_features = []
            
            batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits, batch_h_freq, interf_batch_y, b= eval_batched_data1[batch_id].values()
            batch_pilots_rg = tf.identity(batch_pilots_rg).gpu()
            batch_y_with_interf = tf.identity(batch_y_with_interf).gpu()
            batch_N0 = tf.identity(batch_N0).gpu()
            model1_fea = self.model1([batch_pilots_rg, batch_y_with_interf, batch_N0])

            batch_pilots_rg, batch_y_with_interf, batch_N0, tx_codeword_bits, batch_h_freq, interf_batch_y, b= eval_batched_data2[batch_id].values()
            batch_pilots_rg = tf.identity(batch_pilots_rg).gpu()
            batch_y_with_interf = tf.identity(batch_y_with_interf).gpu()
            batch_N0 = tf.identity(batch_N0).gpu()
            model2_fea = self.model2([batch_pilots_rg, batch_y_with_interf, batch_N0])

            self.model1_features = model1_fea
            self.model2_features = model2_fea

            for i, feat1 in enumerate(self.model1_features):
                
                X = tf.reshape(feat1, [feat1.shape[0], -1]) # num_batch * num_flatten_features
                K = X @ tf.transpose(X) # num_batch * num_batch
                K = tf.linalg.set_diag(K, tf.zeros(K.shape[0], dtype=K.dtype))
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, feat2 in enumerate(self.model2_features):
                    Y = tf.reshape(feat2, [feat2.shape[0], -1])
                    L = Y @ tf.transpose(Y)
                    L = tf.linalg.set_diag(L, tf.zeros(L.shape[0], dtype=L.dtype))

                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (tf.math.sqrt(self.hsic_matrix[:, :, 0]) *
                                                        tf.math.sqrt(self.hsic_matrix[:, :, 2]))
        
        # Here the sqrt is the root cause of nan. In the normal case, the HSIC tensor elements should be positive.
        # After normalization, hsic_matrix has the shape of [N,M]
        print("Successfully computed HSIC matrix.") 
        print(self.hsic_matrix)
        print(np.diag(self.hsic_matrix))
        assert not tf.reduce_any(tf.math.is_nan(self.hsic_matrix)), "HSIC computation resulted in NANs"
        return self.hsic_matrix


    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        # ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        # ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        # if title is not None:
        #     ax.set_title(f"{title}", fontsize=18)
        # else:
        #     ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        # plt.show()

