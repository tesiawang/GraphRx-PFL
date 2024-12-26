
import pickle
import numpy as np
from pathlib import Path
import argparse
import os
import sys
import matplotlib.pyplot  as plt
import random

# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import sionna as sn
import scipy.io as sio

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
# from Evaluation.eval_utils import load_ber_from_file

eval_result_save_folder = root_path + "/Evaluation/Ablation_Aug/Dist0_HighSIR"
min_ebNo = -4
max_ebNo = 15
num_ebNo_points = 20
num_total_clients = 6
plot_each = 1
plot_avg = 1


# --------------------- plot BER for each client locally --------------------- #
if __name__ == "__main__":
    ebno_dbs = np.linspace(min_ebNo,max_ebNo,num_ebNo_points)

    if plot_each == 1:
        # --------------------------- Plot for each client --------------------------- #
        aug_list = [1,2,4,8,16] # four different augmentation times
        ebno_dbs = np.linspace(min_ebNo,max_ebNo,num_ebNo_points)

        for client_id in range(num_total_clients):
            res_list = list()
            for aug_time in aug_list:
                ber = pickle.load(open(eval_result_save_folder + "/aug{:d}_bs{:d}.pkl".format(aug_time, client_id), 'rb'))
                res_list.append(ber)

            plt.figure(figsize=(10,6))
            plt.semilogy(ebno_dbs, res_list[0], 's-', c=f'C3', label=f'Aug=1 (160 batches of data)')
            plt.semilogy(ebno_dbs, res_list[1], 'x--', c=f'C1', label=f'Aug = 2')
            plt.semilogy(ebno_dbs, res_list[2], '>-', c=f'C2', label=f'Aug = 4')
            plt.semilogy(ebno_dbs, res_list[3], 'o-', c=f'C0', label=f'Aug = 8')
            plt.semilogy(ebno_dbs, res_list[4], 'o-', c=f'C4', label=f'Aug = 16')

            plt.xlabel(r"$E_b/N_0$ (dB)")
            plt.ylabel("Bit Error Rate (BER)")
            plt.grid(which="both")
            plt.ylim((1e-6, 1.0))
            plt.xlim((0,15))
            plt.legend()
            plt.tight_layout()
            plt.savefig(eval_result_save_folder + "/vary_aug_ber_bs{:d}.png".format(client_id))

        print("The BER is plotted!")


    if plot_avg == 1:
        # # ------------------- Plot average BER for the whole system ------------------ #
        aug_list = [1,2,4,8,16]
        ebno_dbs = np.linspace(min_ebNo,max_ebNo,num_ebNo_points)

        alg_ber_list = []
        avg_ber_dict = {}
        for aug_time in aug_list:
            avg_ber = 0.
            for client_id in range(num_total_clients):
                ber = pickle.load(open(eval_result_save_folder + "/aug{:d}_bs{:d}.pkl".format(aug_time, client_id), 'rb'))
                avg_ber += ber 

            avg_ber /= num_total_clients # average BER achieved by this algorithm
            alg_ber_list.append(avg_ber)
            name = "aug{:d}_ber".format(aug_time)
            avg_ber_dict[name] = avg_ber
            
        sio.savemat(eval_result_save_folder + "/avg_ber.mat", avg_ber_dict)
         # --------------------------- Plot the average BER --------------------------- #
        plt.figure(figsize=(10,6))

        # load avg_ber from the mat file
        avg_ber_dict = sio.loadmat(eval_result_save_folder + "/avg_ber.mat")
                    
        plt.semilogy(ebno_dbs, alg_ber_list[0], 's-', c=f'C3', label=f'Aug=1 (160 batches of data)')
        plt.semilogy(ebno_dbs, alg_ber_list[1], 'x--', c=f'C1', label=f'Aug=2 (160 batches of data)')
        plt.semilogy(ebno_dbs, alg_ber_list[2], '>-', c=f'C2', label=f'Aug=4 (160 batches of data)')
        plt.semilogy(ebno_dbs, alg_ber_list[3], 'o-', c=f'C0', label=f'Aug=8 (160 batches of data)')
        plt.semilogy(ebno_dbs, alg_ber_list[4], 'o-', c=f'C4', label=f'Aug=16')

        plt.ylabel("Bit Error Rate")
        plt.grid(which="both")
        plt.ylim((1e-6, 1.0))
        plt.xlim((0, 15))
        plt.legend()
        plt.tight_layout()
        plt.savefig(eval_result_save_folder + "/ber_avg.png")
        print("The average BER is plotted!")

