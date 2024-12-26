#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=4:gpu_type=RTX6000
#PBS -o /rds/general/user/tw223/home/pFedRx/Logs_HPC/
#PBS -e /rds/general/user/tw223/home/pFedRx/Logs_HPC/
export WANDB_API_KEY='4e58f7ae7908a3a92c7b6f2e7bbba0a3142cca70'
cd $HOME/pFedRx/OfflinePretrain
sir_patterns=("NoneSIR" "HighSIR" "MidSIR")
for i in {0..2};do
    CUDA_VISIBLE_DEVICES=$i singularity run --nv $HOME/hpc_sionna_container_latest.sif python pretrain_entire_main_net.py --seed 42 \
                                                                                --num_epochs 20 \
                                                                                --save_model_every 2 \
                                                                                --SIR_pattern ${sir_patterns[$i]} \
                                                                                --lr 0.001 &
done
wait