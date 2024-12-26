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
parser.add_argument("--eval_result_save_folder", type=str, default=root_path + "/Evaluation/OnlineEvalResults/Loss_2Nr_TDLA", required=False)
parser.add_argument("--pretrained_metrics", type=str, default=root_path + "/OfflinePretrain/TrainMetrics_NoneSIR_2Nr", required=False)
parser.add_argument("--num_epochs", type=int, default=100, required=False)

run_args = parser.parse_args()
# do not shutil the folder; keep the previous results
if not os.path.exists(run_args.eval_result_save_folder):
    os.makedirs(run_args.eval_result_save_folder,exist_ok=False)


# ---------------------------------------------------------------------------- #
#                             Plot offline metrics                             #
# ---------------------------------------------------------------------------- #
# load the epoch train loss
with open(Path(run_args.pretrained_metrics).joinpath("all_epoch_train_loss.pkl"), 'rb') as f:
    all_epoch_train_loss = pickle.load(f)
with open(Path(run_args.pretrained_metrics).joinpath("all_epoch_test_loss.pkl"), 'rb') as f:
    all_epoch_test_loss = pickle.load(f)
with open(Path(run_args.pretrained_metrics).joinpath("all_epoch_test_rate.pkl"), 'rb') as f:
    all_epoch_test_rate = pickle.load(f)


# plot the loss curve
plt.figure()
plt.plot(np.linspace(1,run_args.num_epochs,num=run_args.num_epochs), all_epoch_train_loss[:run_args.num_epochs], '-', label="avg train loss")
plt.plot(np.linspace(1,run_args.num_epochs,num=run_args.num_epochs), all_epoch_test_loss[:run_args.num_epochs], '-', label="avg test loss")
plt.ylim([0.,0.7])
plt.legend()
plt.savefig(run_args.eval_result_save_folder + "/loss.png")

plt.figure()
plt.plot(np.linspace(1,run_args.num_epochs,num=run_args.num_epochs), all_epoch_test_rate[:run_args.num_epochs], 'o-')
plt.savefig(run_args.eval_result_save_folder + "/test_rate.png")

