# Transformers for COVID-19 Detection from Chest X-Rays


This is a self-study project exploring **Vision Transformer (ViT)** architectures for COVID-19 detection from chest X-rays, and benchmarking their performance against a standard CNN architecture (ResNet-18).

We use a base ViT model with 16×16 patches, pretrained on ImageNet-1k, and fine-tuned on the **COVID-QU-Ex dataset**.

---

## Platform

* **Python:** 3.8
* **PyTorch:** 1.10.0, **Torchvision:** 0.11.1
* **Experiment Tracking:** Trackio, MLflow
* **GPU:** NVIDIA RTX 4060 Laptop GPU

---

## Dataset

* **COVID-QU-Ex:** 33,920 chest X-ray images ([Kaggle link](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu))
* **Classes:** COVID-19, Normal, Pneumonia/Non-COVID
* **Splits:** Train, Validation, Test

![Sample frequencies per class across datasets](docs/data_summary.png)

---

### Training Notes

Training strategies used:

* **Optimizer:** AdamW (better regularization than Adam)
* **Learning rate schedule:** Warm-up + Cosine Annealing
* **Class balance:** Ensured balanced batches to avoid biased predictions

Train models using the corresponding config file in the `configs/` folder. Example:


## Evaluation

Models are compared on the **Test set** using accuracy, precision, recall, and F1-score. Confusion matrices illustrate per-class performance.

**Results: ViT vs ResNet-18**

| Model Name | Accuracy | Precision | Recall | F1-score | Parameters |
| ---------- | -------- | --------- | ------ | -------- | ---------- |
| ResNet-18  | 82.56    | 82.66     | 82.41  | 82.37    | 11.24M     |
| ViT_b_16   | 87.51    | 87.45     | 87.37  | 87.33    | 85.8M      |

![F1-score per class](docs/f1_score_perclass.png) ![Precision per class](docs/Precision_perclass.png) ![Recall per class](docs/Recall_perclass.png)

**Confusion Matrices:**

| ResNet-18                             | ViT_b_16                                   |
| ------------------------------------- | ------------------------------------------ |
| ![](docs/confusion_matrix_Resnet.png) | ![](docs/confusion_matrix_TV_vit_b_16.png) |

---

### Key Insights

* ViT-based models **outperform ResNet-18** on all metrics.
* ViTs have **significantly more parameters**, making them harder to train on resource-constrained devices (e.g., laptop GPUs).
* This study demonstrates the effectiveness of **transformer architectures for medical image classification** and provides a benchmark for future experiments.
