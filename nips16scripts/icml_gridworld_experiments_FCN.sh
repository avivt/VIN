#!/bin/bash
# Script for running icml gridworld experiments with FCN networks
# 16x16 map
THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ~/Data/LearnTraj/icml16/gridworld_16.mat --output ./nips16results/gridworld/grid16_FCN.pk --epochs 30 --model fcn --stepsize 0.01 --imsize 16 --reg 0.0 --batchsize 12 --statebatchsize 10 | tee -a ./nips16results/gridworld/out_grid16_FCN.txt ; 
THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ~/Data/LearnTraj/icml16/gridworld_16.mat --output ./nips16results/gridworld/grid16_FCN.pk --epochs 30 --model fcn --stepsize 0.005 --imsize 16 --reg 0.0 --batchsize 12 --statebatchsize 10 --warmstart ./nips16results/gridworld/grid16_FCN.pk | tee -a ./nips16results/gridworld/out_grid16_FCN.txt ; 
THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ~/Data/LearnTraj/icml16/gridworld_16.mat --output ./nips16results/gridworld/grid16_FCN.pk --epochs 30 --model fcn --stepsize 0.002 --imsize 16 --reg 0.0 --batchsize 12 --statebatchsize 10 --warmstart ./nips16results/gridworld/grid16_FCN.pk | tee -a ./nips16results/gridworld/out_grid16_FCN.txt ; 
THEANO_FLAGS='floatX=float32,device=gpu' python NN_run_training.py --input ~/Data/LearnTraj/icml16/gridworld_16.mat --output ./nips16results/gridworld/grid16_FCN.pk --epochs 30 --model fcn --stepsize 0.001 --imsize 16 --reg 0.0 --batchsize 12 --statebatchsize 10 --warmstart ./nips16results/gridworld/grid16_FCN.pk | tee -a ./nips16results/gridworld/out_grid16_FCN.txt ; 


