# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path
import argparse
import os
import sys
import matplotlib.pyplot  as plt

# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set root path
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)


parser = argparse.ArgumentParser()
# path config
parser.add_argument("--eval_result_save_folder", type=str, default=root_path + "/Evaluation/OnlineEvalResults/2Nr_Traindist4_Results/local_loss", required=False)
# config the FL method
parser.add_argument("--fl_method", type=str, default="fedrep", required=False)
parser.add_argument("--seed", type=int, default=0, required=False)
parser.add_argument("--num_rounds", type=int, default=20, required=False)
parser.add_argument("--num_total_clients", type=int, default=6, required=False)
parser.add_argument("--exp_id", type=int, default=8, required=False)


run_args = parser.parse_args()
# do not shutil the folder; keep the previous results
if not os.path.exists(run_args.eval_result_save_folder):
    os.makedirs(run_args.eval_result_save_folder,exist_ok=False)

# ---------------------------------------------------------------------------- #
#                                 Plot online metrics                          #
# ---------------------------------------------------------------------------- #

metric_path = ""
if run_args.fl_method == "fedavg":
    metric_path = root_path + "/OnlineAdapt/FedAvgs/TrainMetrics_EXP{:d}".format(run_args.exp_id)
elif run_args.fl_method == "fedgraph":
    metric_path = root_path+"/OnlineAdapt/FedGraphs/TrainMetrics_EXP{:d}".format(run_args.exp_id)   
elif run_args.fl_method == "ditto":
    metric_path = root_path + "/OnlineAdapt/Dittos/TrainMetrics_EXP{:d}".format(run_args.exp_id)
elif run_args.fl_method == "fedrep":
    metric_path = root_path + "/OnlineAdapt/FedReps/TrainMetrics_EXP{:d}".format(run_args.exp_id)
elif run_args.fl_method == "fedavgFT":
    metric_path = root_path + "/OnlineAdapt/FedAvgFT/TrainMetrics_EXP{:d}".format(run_args.exp_id)
elif run_args.fl_method == "localonly":
    metric_path = root_path + "/OnlineAdapt/LocalOnly/TrainMetrics"
else:
    raise ValueError("Invalid FL method!")

# fedavg_client5_loss = 0.
# fedrep_client5_loss = 0.

if run_args.fl_method != "localonly":
   
    # ------------------------ load global average metrics ----------------------- #
    glob_round_train_loss = list()
    glob_round_test_loss = list()
    glob_round_test_rate = list()
    
    for r in range(run_args.num_rounds):
        with open(Path(metric_path).joinpath("Global","glob_metrics_round_%d.pkl" % r), 'rb') as f:
            glob_metrics = pickle.load(f)
        glob_round_train_loss.append(glob_metrics['avg_train_loss'])
        glob_round_test_loss.append(glob_metrics['avg_test_loss'])
        glob_round_test_rate.append(glob_metrics['avg_test_rate'])

    # plot the loss curve
    plt.figure()
    plt.plot(np.linspace(1,run_args.num_rounds,num=run_args.num_rounds), glob_round_train_loss, 'o-', label="avg train loss")
    plt.plot(np.linspace(1,run_args.num_rounds,num=run_args.num_rounds), glob_round_test_loss, 'o-', label="avg test loss")
    # plt.ylim([0.190,0.210])
    plt.legend()
    plt.savefig(run_args.eval_result_save_folder + "/" + run_args.fl_method + "_EXP{:d}_glob_round_loss.png".format(run_args.exp_id))


    # print(np.array(glob_round_train_loss))
    # print(np.array(glob_round_test_loss))
    # with open(root_path + "/best_train_loss.pkl", 'wb') as f:
    #     pickle.dump(glob_round_train_loss, f)
    # with open(root_path + "/best_test_loss.pkl", 'wb') as f:
    #     pickle.dump(glob_round_test_loss, f)

    # plt.figure()
    # plt.plot(np.linspace(1,run_args.num_rounds,num=run_args.num_rounds), glob_round_test_rate, 'o-', label="avg test rate")
    # plt.legend()
    # plt.savefig(run_args.eval_result_save_folder + "/" + run_args.fl_method + "_glob_round_rate.png")

    # -------------------------- load the client metrics ------------------------- #
    
    for client_id in range(run_args.num_total_clients):
        round_train_loss = list()
        round_test_loss = list()
        round_test_rate = list()
        for r in range(run_args.num_rounds):
            with open(Path(metric_path).joinpath("Client%d" % client_id,"local_metrics_round_%d.pkl" % r), 'rb') as f:
                local_metrics = pickle.load(f)
            round_train_loss.append(local_metrics['train_loss'])
            round_test_loss.append(local_metrics['test_loss'])
            round_test_rate.append(local_metrics['test_rate'])

        # plot the loss curve
        plt.figure()
        plt.plot(np.linspace(1,run_args.num_rounds,num=run_args.num_rounds), round_train_loss, 'o-', label="train loss")
        plt.plot(np.linspace(1,run_args.num_rounds,num=run_args.num_rounds), round_test_loss, 'o-', label="test loss")
        # plt.ylim([0.190,0.220])
        plt.legend()
        plt.savefig(run_args.eval_result_save_folder + "/" + run_args.fl_method + "_EXP{:d}_client{:d}_round_loss.png".format(run_args.exp_id, client_id))


    # ---------------------------------------------------------------------------- #

elif run_args.fl_method == "localonly":
    #TODO: add the average metrics for localonly methods
    for client_id in range(run_args.num_total_clients):
        with open(Path(metric_path).joinpath("Client%d" % client_id,"all_epoch_train_loss.pkl"), 'rb') as f:
            all_epoch_train_loss = pickle.load(f)
        with open(Path(metric_path).joinpath("Client%d" % client_id,"all_epoch_test_loss.pkl"), 'rb') as f:
            all_epoch_test_loss = pickle.load(f)
        with open(Path(metric_path).joinpath("Client%d" % client_id,"all_epoch_test_rate.pkl"), 'rb') as f:
            all_epoch_test_rate = pickle.load(f)

        plt.figure()
        plt.plot(all_epoch_train_loss, label="train loss")
        plt.plot(all_epoch_test_loss, label="test loss")
        plt.legend()
        plt.savefig(Path(run_args.eval_result_save_folder).joinpath("localonly_client%d_loss.png" % client_id))
        plt.close()

        plt.figure()
        plt.plot(all_epoch_test_rate, label="test rate")
        plt.legend()
        plt.savefig(Path(run_args.eval_result_save_folder).joinpath("localonly_client%d_rate.png" % client_id))
        plt.close()
else:
    raise ValueError("Invalid FL method!")
