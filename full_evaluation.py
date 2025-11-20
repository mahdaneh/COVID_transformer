import json

import matplotlib.pyplot as plt
import mlflow
import torch
from ptflops import get_model_complexity_info
from sklearn.metrics import *
from torch.utils.data import DataLoader

import Operations as ops
import dataset_repo as d_repo


def load_model(model_path):
    model_config_path = model_path.replace("weights", "configs").replace(
        "_checkpoint_99.pth", ".json"
    )
    with open(model_config_path) as f:
        config_dict = json.load(f)

    model_state_dict = torch.load(model_path)
    # with torch.device("meta"):
    model = ops.build_model(config_dict)
    model.load_state_dict(model_state_dict, assign=True)
    model.eval()
    return model


def clac_flops(model):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model,
            (3, 224, 224),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
    return 2 * macs, params


def model_evaluation():
    data_config = "configs/full_evaluation_models.json"
    with open(data_config) as f:
        test_data_dict = json.load(f)
    class_names = test_data_dict["DATA"]["label names"]
    test_dataset = d_repo.Covid_QU_Ex(
        test_data_dict["DATA"],
        training=True,
        mode="train",
    )
    # Load your test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_data_dict["DATA"]["batch size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    # List of models to evaluate

    model_path = [
        ("VIT", "weights/VIT_QU_EX_deep_checkpoint_99.pth"),
        ("Resnet", "weights/Resnet_QU_EX_checkpoint_99.pth"),
        ("Res_VIT", "weights/Res_VIT_deep_checkpoint_99.pth"),
    ]

    experiment_name = "model_comparison_TestSet"
    mlflow.set_experiment(experiment_name)

    for name, path in model_path:
        with mlflow.start_run(run_name=name, nested=True):
            # Load model
            model = load_model(path)
            flops, params = clac_flops(model)
            log, y_pred, y_true = ops.evaluation(model, test_loader)

            metrics_dict = {}
            p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            for i, cls in enumerate(class_names):
                metrics_dict["precision_class_%s" % cls] = p[i]
                metrics_dict["recall_class_%s" % cls] = r[i]
                metrics_dict["f1_class_%s" % cls] = f1[i]
                metrics_dict["support_class_%s" % cls] = s[i]
            mlflow.log_metrics(metrics_dict)

            p, r, f1, s = precision_recall_fscore_support(
                y_true, y_pred, average="macro"
            )
            mlflow.log_metric("precision_macro", p)
            mlflow.log_metric("recall_macro", r)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("accuracy", accuracy)
            print(f"accuracy {accuracy}")

            mlflow.log_text(str(flops), "calc.txt")
            mlflow.log_text(str(params), "calc.txt")
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            disp.plot()
            plt.savefig("docs/confusion_matrix_%s.png" % name)


if __name__ == "__main__":
    model_evaluation()
