# COVID_transformer [To be Continued]

A Vision Transformer (ViT)–based deep learning project for classifying chest X-ray (CXR) images
into three categories: **COVID-19**, **Normal**, and **Non-COVID**.

## Overview

This repository implements a Vision Transformer and compare it with Resnet18 and a Resnet18 backbone with VIT head
models on chest xray images.

### Data

Covid-QU-EU dataset is used for training and evaluation purposes. You can read about this data
set [here](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)
![Sample frequencies per class across datasets](docs/data_summary.png)

### Notes

- AdamW optimizer with weight decay (AdamW provides better regularization than Adam.)
- Ensure dataset class balance to avoid biased predictions.
- Warm-up + cosine annealing learning rate strategy with small base learning rate

Train each model using its corresponding config file, available in `configs/` folder. For example:

``python run_train_eval.py --config configs/your_config.json``

3 models including VIT, Resnet18, and VIT head with Res18 backbone (frozen) are trained and evaluated on Covid-QU-EX
dataset.

![Training curves of the models](docs/comparison.png)

### Evaluation

Use utilities in util.py to compute:

| Model Name   | Acc    | Precision   | Recall   | f_score   |
|--------------|--------|-------------|----------|-----------|
| Res18        | 89.9   | 89.9        | 89.8     | 89.7      |
| VIT          | 72.7   | 73.3        | 73.1     | 73        |
| Res18+VIT    | 68.6   | 69.5        | 68.7     | 68.8      |


| Res18                                 | VIT                                | Res18+VIT                              |
|---------------------------------------|------------------------------------|----------------------------------------|
| ![](docs/confusion_matrix_Resnet.png) | ![](docs/confusion_matrix_VIT.png) | ![](docs/confusion_matrix_Res_VIT.png) |


