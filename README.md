# COVID_transformer 

This project implements a Vision Transformer (ViT) for COVID-19 detection from chest X-rays, and benchmarks performance against:

- **ResNet-18** (standard CNN)  
- **ResNet-18 + ViT head** (where the ResNet backbone is frozen and the ViT head is trained)  

The goal is to explore the effectiveness of transformer-based models on medical imaging, particularly in distinguishing COVID-19 cases in CXR data.

## Overview

This repository implements a Vision Transformer and compare it with Resnet18 and a Resnet18 backbone with VIT head
models on chest xray images.

### Dataset 
- The repository uses the **COVID-QU-Ex** dataset, which contains 33,920 chest X-ray images [here](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu).
- This is a multiclass dataset with classes: COVID-19, Normal, and Pneumonia/Non-COVID.
- The dataset comprises three sets: Train, Val, Test. 

![Sample frequencies per class across datasets](docs/data_summary.png)

### Training
3 models including VIT, Resnet18, and VIT head with Res18 backbone (frozen) are trained and evaluated on Covid-QU-EX
dataset.
#### Notes
- AdamW optimizer with weight decay (AdamW provides better regularization than Adam.)
- Ensure dataset class balance to avoid biased predictions.
- Warm-up + cosine annealing learning rate strategy with small base learning rate

Train each model using its corresponding config file, available in `configs/` folder. For example:

``python run_train_eval.py --config configs/VIT_QU_EX.json``
To train other models, you could find other config files in ``config`` folder.

![Training curves of the models](docs/comparison.png)

### Evaluation

On "Test" set, we compare the three models by acc, precision, recall, f_score. The confusion
matrices als show the models performance for classes.  

| Model Name   | Acc    | Precision   | Recall   | f_score   |
|--------------|--------|-------------|----------|-----------|
| Res18        | 89.9   | 89.9        | 89.8     | 89.7      |
| VIT          | 72.7   | 73.3        | 73.1     | 73        |
| Res18+VIT    | 68.6   | 69.5        | 68.7     | 68.8      |


| Res18                                 | VIT                                | Res18+VIT                              |
|---------------------------------------|------------------------------------|----------------------------------------|
| ![](docs/confusion_matrix_Resnet.png) | ![](docs/confusion_matrix_VIT.png) | ![](docs/confusion_matrix_Res_VIT.png) |


