#!/bin/bash
python run_model.py --dataset_name="isotropic" --model="NonEquivResNet" --batch_size=16 --seed=0 --hidden_dim=128 --learning_rate=0.0001 --kernel_size=3 
python run_model.py --dataset_name="isotropic" --model="EquivResNet" --batch_size=8 --seed=0 --hidden_dim=16 --learning_rate=0.0001 --kernel_size=3 --equiv_last=False --separable=False --reflection=True
python run_model.py --dataset_name="isotropic" --model="Relaxed_EquivResNet" --batch_size=2 --seed=0 --hidden_dim=16 --learning_rate=0.0001 --kernel_size=3 --equiv_last=False --separable=False --reflection=True

python run_model.py --dataset_name="channel" --model="NonEquivResNet" --batch_size=8 --seed=0 --hidden_dim=64 --learning_rate=0.001 --kernel_size=3
python run_model.py --dataset_name="channel" --model="EquivResNet" --batch_size=4 --seed=0 --hidden_dim=16 --learning_rate=0.0005 --kernel_size=3 --equiv_last=False --separable=False --reflection=True
python run_model.py --dataset_name="channel" --model="Relaxed_EquivResNet" --batch_size=2 --seed=0 --hidden_dim=16 --learning_rate=0.0001 --kernel_size=3 --equiv_last=False --separable=False --reflection=True

