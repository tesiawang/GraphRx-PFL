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

parser = argparse.ArgumentParser()
parser.add_argument("--eval_result_save_folder", type=str, default=root_path + "/Evaluation/NewOnlineEvalResults_Interf/Dist5/local", required=False)
parser.add_argument("--min_ebNo", type=float, default=-4, required=False)
parser.add_argument("--max_ebNo", type=float, default=15, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=20, required=False)
parser.add_argument("--num_total_clients", type=int, default=6, required=False)
parser.add_argument("--plot_each", type=int, default=0, required=False)
parser.add_argument("--plot_avg", type=int, default=1, required=False)
run_args = parser.parse_args()

# ---------------------------------------------------------------------------- #

def load_ber_from_file(eval_result_save_folder: str, 
                       config_id: int,
                       fl_method: str):
    new_file_name = fl_method + '_ber_c%d.pkl' % config_id
    with open(Path(eval_result_save_folder).joinpath(new_file_name),'rb') as f:
        ber = pickle.load(f)
    return ber

def load_bler_from_file(eval_result_save_folder: str, 
                        config_id: int,
                        fl_method: str):
    new_file_name = fl_method + '_bler_c%d.pkl' % config_id
    with open(Path(eval_result_save_folder).joinpath(new_file_name),'rb') as f:
        bler = pickle.load(f)
    return bler


# --------------------- plot BER for each client locally --------------------- #
if __name__ == "__main__":
    os.makedirs(run_args.eval_result_save_folder, exist_ok=True)
    if run_args.plot_each == 1:
        # --------------------------- Plot for each client --------------------------- #
        alg_list = ["fedgraph_new","fedgraph_old", "ditto","fedrep","localonly","lmmse","ideal","fedavg"] # 
        ebno_dbs = np.linspace(run_args.min_ebNo,run_args.max_ebNo,run_args.num_ebNo_points)

        for client_id in range(run_args.num_total_clients):
            res_list = list()
            for alg in alg_list:
                ber = load_ber_from_file(eval_result_save_folder=run_args.eval_result_save_folder,
                                        config_id=client_id,
                                        fl_method=alg)
                res_list.append(ber)

            plt.figure(figsize=(10,6))
            plt.semilogy(ebno_dbs, res_list[0], 's-', c=f'C3', label=f'Our Method ')
            plt.semilogy(ebno_dbs, res_list[1], 'x--', c=f'C1', label=f'pFedGraph')
            plt.semilogy(ebno_dbs, res_list[2], '>-', c=f'C2', label=f'Ditto')
            plt.semilogy(ebno_dbs, res_list[3], 'o-', c=f'C0', label=f'FedRep')
            plt.semilogy(ebno_dbs, res_list[4], '^--', c=f'C4', label=f'LocalAdapt')
            plt.semilogy(ebno_dbs, res_list[5], 'v-', c=f'C5', label=f'LMMSE')
            plt.semilogy(ebno_dbs, res_list[6], '<-', c=f'C9', label=f'Ideal')
            plt.semilogy(ebno_dbs, res_list[7], '>-.', c=f'C8', label=f'FedAvg')
            
            plt.xlabel(r"$E_b/N_0$ (dB)")
            plt.ylabel("Bit Error Rate (BER)")
            plt.grid(which="both")
            plt.ylim((1e-6, 1.0))
            plt.xlim((0,15))
            plt.legend()
            plt.tight_layout()
            plt.savefig(run_args.eval_result_save_folder + "/ber_c{:d}.png".format(client_id))

        print("The BER is plotted!")


    if run_args.plot_avg == 1:
        # # ------------------- Plot average BER for the whole system ------------------ #
        alg_list = ["fedgraph_new","fedgraph_old", "ditto","fedrep", "localonly", "lmmse","ideal","fedavg"] # "localonly", 
        ebno_dbs = np.linspace(run_args.min_ebNo,run_args.max_ebNo,run_args.num_ebNo_points)

        alg_ber_list = []
        std_list = []  
        avg_ber_dict = {}
        for alg in alg_list:
            avg_ber = 0.
            one_point_std = 0.
            one_point_ber_list = []
            for client_id in range(run_args.num_total_clients):
                ber = load_ber_from_file(eval_result_save_folder=run_args.eval_result_save_folder,
                                        config_id=client_id,
                                        fl_method=alg)
                avg_ber += ber 
                one_point_ber_list.append(ber[10]) # ber[10]: the 10-th point ber for client i

            avg_ber /= run_args.num_total_clients # average BER achieved by this algorithm
            one_point_std = np.std(one_point_ber_list) # BER variance achieved by this algorithm

            std_list.append(one_point_std)
            alg_ber_list.append(avg_ber)
            avg_ber_dict[alg] = avg_ber
            
        # sio.savemat(run_args.eval_result_save_folder + "/avg_ber.mat", avg_ber_dict)
        
        # --------------------------- Plot the standard deviation --------------------------- #
        sio.savemat(run_args.eval_result_save_folder + "/std_ber.mat", {"std": std_list})
        plt.figure(figsize=(10,6))
        plt.bar(alg_list, std_list)
        plt.xlabel("FL Methods")
        plt.ylabel("Standard Deviation of One-Point BER")
        plt.grid(which="both")
        plt.tight_layout()
        plt.savefig(run_args.eval_result_save_folder + "/a_std_bar.png")
        print("The standard deviation of one-point BER is plotted!")



        # --------------------------- Plot the average BER --------------------------- #
        # plt.figure(figsize=(10,6))

        # # load avg_ber from the mat file
        # avg_ber_dict = sio.loadmat(run_args.eval_result_save_folder + "/avg_ber.mat")
                    
        # plt.semilogy(ebno_dbs, alg_ber_list[0], 's-', c=f'C3', label=f'Our Method')
        # plt.semilogy(ebno_dbs, alg_ber_list[1], 'x--', c=f'C1', label=f'pFedGraph')
        # plt.semilogy(ebno_dbs, alg_ber_list[2], '>-', c=f'C2', label=f'Ditto')
        # plt.semilogy(ebno_dbs, alg_ber_list[3], 'o-', c=f'C0', label=f'FedRep')
        # plt.semilogy(ebno_dbs, alg_ber_list[4], '^--', c=f'C4', label=f'Localonly')
        # plt.semilogy(ebno_dbs, alg_ber_list[5], 'v-', c=f'C5', label=f'LMMSE')
        # plt.semilogy(ebno_dbs, alg_ber_list[6], '<-', c=f'C9', label=f'Ideal')
        # plt.semilogy(ebno_dbs, alg_ber_list[7], '>-.', c=f'C8', label=f'FedAvg')
        # plt.xlabel(r"$E_b/N_0$ (dB)")
        # plt.ylabel("Bit Error Rate")
        # plt.grid(which="both")
        # plt.ylim((1e-6, 1.0))
        # plt.xlim((0, 15))
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(run_args.eval_result_save_folder + "/ber_avg.png")
        # print("The average BER is plotted!")

