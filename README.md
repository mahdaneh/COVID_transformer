# COVID_transformer [under development]

A Vision Transformer (ViT)–based deep learning project for classifying chest X-ray (CXR) images 
into three categories: **COVID-19**, **Normal**, and **Non-COVID**.

## Overview

This repository implements a Vision Transformer and compare it with Resnet18 and a Resnet18 backbone with VIT head model on chest xray images. 
It supports:

- Training a ViT model on chest X-ray datasets  
- Classifying images into *COVID-19*, *Normal*, and *Non-COVID pneumonia*  
- Custom training with cosine annealing and warm-up learning rate schedules

## Key Notes
- AdamW optimizer with weight decay (AdamW provides better regularization than Adam.)
- Ensure dataset class balance to avoid biased predictions.
- Warm-up + cosine annealing learning rate strategy  
- Configurable training via JSON files in `configs/`  
- Modular code structure for data, model, and utilities  
- Real-world CXR dataset support  
- Scripts for training and evaluating the model


Train each model using its corresponding config file, available in `configs/` folder:

``python train_QU_EX_COVID.py --config configs/your_config.json``


### Key hyperparameters:

Learning rate

Weight decay (AdamW)

Batch size

Epochs

Image size

## Evaluation

Use utilities in util.py to compute:

Accuracy

Precision / Recall / F1

Confusion matrix

Loss & metric curves


