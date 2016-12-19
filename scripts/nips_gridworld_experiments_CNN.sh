#!/bin/bash
# Script for running icml gridworld experiments with CNN networks
# 8x8 map
THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ./data/gridworld_8_state_channel.mat --output ./nips16results/gridworld/grid8_CNN.pk --epochs 20 --model cnn --stepsize 0.01 --imsize 8 --reg 0.0 --batchsize 128 | tee -a ./nips16results/gridworld/out_grid8_CNN.txt ; 
THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ./data/gridworld_8_state_channel.mat --output ./nips16results/gridworld/grid8_CNN.pk --epochs 20 --model cnn --stepsize 0.002 --imsize 8 --reg 0.0 --batchsize 128 --warmstart ./nips16results/gridworld/grid8_CNN.pk | tee -a ./nips16results/gridworld/out_grid8_CNN.txt ; 
# 16x16 map
#THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ./data/gridworld_16_state_channel.mat --output ./nips16results/gridworld/grid16_CNN.pk --epochs 20 --model cnn --stepsize 0.01 --imsize 16 --reg 0.0 --batchsize 128 | tee -a ./nips16results/gridworld/out_grid16_CNN.txt ; 
#THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ./data/gridworld_16_state_channel.mat --output ./nips16results/gridworld/grid16_CNN.pk --epochs 20 --model cnn --stepsize 0.002 --imsize 16 --reg 0.0 --batchsize 128 --warmstart ./nips16results/gridworld/grid16_CNN.pk | tee -a ./nips16results/gridworld/out_grid16_CNN.txt ; 
# 28x28 map
THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ./data/gridworld_28_state_channel.mat --output ./nips16results/gridworld/grid28_CNN.pk --epochs 20 --model cnn --stepsize 0.01 --imsize 28 --reg 0.0 --batchsize 128 | tee -a ./nips16results/gridworld/out_grid28_CNN.txt ; 
THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ./data/gridworld_28_state_channel.mat --output ./nips16results/gridworld/grid28_CNN.pk --epochs 20 --model cnn --stepsize 0.002 --imsize 28 --reg 0.0 --batchsize 128 --warmstart ./nips16results/gridworld/grid28_CNN.pk | tee -a ./nips16results/gridworld/out_grid28_CNN.txt ; 
