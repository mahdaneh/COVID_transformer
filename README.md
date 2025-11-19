# COVID_transformer [To be Continued]

A Vision Transformer (ViT)–based deep learning project for classifying chest X-ray (CXR) images 
into three categories: **COVID-19**, **Normal**, and **Non-COVID**.

## Overview

This repository implements a Vision Transformer and compare it with Resnet18 and a Resnet18 backbone with VIT head model on chest xray images. 

### Data
Covid-QU-EU dataset is used for training and evaluation purposes. You can read about this data set [here](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)


### Notes
- AdamW optimizer with weight decay (AdamW provides better regularization than Adam.)
- Ensure dataset class balance to avoid biased predictions.
- Warm-up + cosine annealing learning rate strategy with small base learning rate

Train each model using its corresponding config file, available in `configs/` folder. For example:

``python run_train_eval.py --config configs/your_config.json``


3 models-- VIT, Resnet18, and VIT with Res18 backbone (frozen) are trained and evaluated on Covid-QU-EX dataset.  

![Model Comparison](docs/comparison.png)


### Evaluation

Use utilities in util.py to compute:

- Accuracy

- Precision / Recall / F1

- Confusion matrix

- Loss & metric curves


