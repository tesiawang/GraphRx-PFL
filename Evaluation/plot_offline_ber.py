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
import sionna as sn

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
# from Evaluation.eval_utils import load_ber_from_file

parser = argparse.ArgumentParser()
parser.add_argument("--eval_result_save_folder", type=str, default=root_path + "/Evaluation/OfflineEvalResults/HighSIR", required=False)
parser.add_argument("--min_ebNo", type=float, default=0, required=False)
parser.add_argument("--max_ebNo", type=float, default =12, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=13, required=False)
run_args = parser.parse_args()

if __name__ == "__main__":
    # ------------------ Plot BER and BLER for the global model ------------------ #
    ber_list = list()
    bler_list = list()  
    snr_db_list = list()
    alg_list = ["nnrx", "lmmse", "ideal"]
    for alg in alg_list:
        with open(run_args.eval_result_save_folder + "/"+ alg + "_ber.pkl", "rb") as file:
            ber = pickle.load(file)
        ber_list.append(ber)
        with open(run_args.eval_result_save_folder + "/"+ alg + "_bler.pkl", "rb") as file:
            bler = pickle.load(file)
        bler_list.append(bler)
        # nr_db_list.append(np.linspace(run_args.min_ebNo,run_args.max_ebNo,run_args.num_ebNo_points))

    ebno_dbs = np.linspace(run_args.min_ebNo,run_args.max_ebNo,run_args.num_ebNo_points)
    plt.figure(figsize=(10,6))
    plt.semilogy(ebno_dbs, ber_list[0], 's-.', c=f'C2', label=f'Pretrained neural receiver')
    plt.semilogy(ebno_dbs, ber_list[1], 'x--', c=f'C1', label=f'LMMSE detection')
    plt.semilogy(ebno_dbs, ber_list[2], 'o-', c=f'C0', label=f'Upper Bound (Perfect CSI)')
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(which="both")
    plt.ylim((1e-5, 1.05))
    plt.xlim((0,12))
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_args.eval_result_save_folder + "/ber.png")
    # sn.utils.plotting.plot_ber(snr_db=snr_db_list,
    #                            ber=ber_list,
    #                            legend=['DeepRx','LMMSERx', 'IdealRx'],
    #                            save_fig=True,
    #                            path=run_args.eval_result_save_folder + "/ber.png")


    plt.figure(figsize=(10,6))
    plt.semilogy(ebno_dbs, bler_list[0], 's-.', c=f'C2', label=f'Pretrained neural receiver')
    plt.semilogy(ebno_dbs, bler_list[1], 'x--', c=f'C1', label=f'LMMSE detection')
    plt.semilogy(ebno_dbs, bler_list[2], 'o-', c=f'C0', label=f'Upper Bound (Perfect CSI)')    

    #
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.ylim((1e-5, 1.1))
    plt.xlim((-4,8))
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_args.eval_result_save_folder + "/bler.png")

