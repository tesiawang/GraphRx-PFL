#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -o /rds/general/user/tw223/home/pFedRx/Logs_HPC/
#PBS -e /rds/general/user/tw223/home/pFedRx/Logs_HPC/
export WANDB_API_KEY='4e58f7ae7908a3a92c7b6f2e7bbba0a3142cca70'
singularity run --nv $HOME/hpc_sionna_container_latest.sif python $HOME/pFedRx/OnlineAdapt/Algo/fedgraph.py \
                                                                                        --seed 53 \
                                                                                        --load_pretrained 1 \
                                                                                        --client_data_dist_type 4 \
                                                                                        --save_model_every 10 \
                                                                                        --local_iter 1600 \
                                                                                        --record_batch_loss_every 400 \
                                                                                        --lr 0.001 \
                                                                                        --clients_per_round 6 \
                                                                                        --num_total_clients 6 \
                                                                                        --train_load_ratio 0.8 \
                                                                                        --test_load_ratio 0.2 \
                                                                                        --num_rounds 30 \
                                                                                        --weighted_initial 1 \
                                                                                        --penalty 0 \
                                                                                        --dist_metric "l2" \
                                                                                        --hyper_c 0.6 \
                                                                                        --directed_graph 1
singularity run --nv $HOME/hpc_sionna_container_latest.sif python3 -u $HOME/pFedRx/Evaluation/online_eval.py \
                          --num_total_clients 6 \
                          --num_rounds 30 \
                          --fl_method "fedgraph_new" \
                          --eval_baseline 1 \
                          --client_data_dist_type 4 >> $HOME/pFedRx/online_eval4.txt 2>&1