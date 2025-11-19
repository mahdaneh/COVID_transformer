import json
import logging
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn


import dataset_repo as d_repo
import util as util
from network import VIT
import Operations as op
import pdb
import argparse


def train_eval(config_file, args):
    weights_folder = Path("weights/")
    run_name = config_file.removeprefix("configs/").removesuffix(".json")

    with open(config_file, "r") as cf:
        config = json.load(cf)
    epochs = config["INFO"]["epochs"]
    accumulation_step = config["INFO"]["accumulation step"]
    log_filename = Path(config_file.replace("configs", "logs").replace(".json", ".log"))

    print("Logging to {}".format(log_filename))
    net_name = config["INFO"]["net name"]
    wb_project_name = config["INFO"]["wandb project"]

    wandb.init(
        project=wb_project_name,
        name=run_name,
        config=config,
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=log_filename,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Logging started")
    logger.info(", ".join(f"{key}: {value}" for key, value in config.items()))

    train_dataset = d_repo.Covid_QU_Ex(
        config["DATA"],
        training=True,
        mode="train",
    )
    val_dataset = d_repo.Covid_QU_Ex(
        config["DATA"],
        training=False,
        mode="val",
    )

    logger.info(
        f"{len(train_dataset)} Training samples; {len(val_dataset)} val samples"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s ", device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["INFO"]["batch size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["INFO"]["batch size"],
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    if net_name == "VIT":
        network = VIT(config["VIT"]).to(device)
    elif net_name == "ResNet":
        network = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )  # or pretrained=True in older versions

        # Replace the final fully connected (fc) layer:
        in_features = network.fc.in_features
        fc_layers = torch.nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1),
        )
        network.fc = fc_layers

    elif net_name == "resnet_VIT":
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(*list(resnet.children())[:4])
        backbone.requires_grad_(False)  # to freeze backbone
        vit = VIT(config["VIT"]).to(device)
        network = nn.Sequential(backbone, vit)

    # network = torch.compile(network)
    network.to(device)
    wandb.watch(network, log="all")
    logger.info("%s model on device: %s", net_name, next(network.parameters()).is_cuda)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["INFO"]["LR"])
    scheduler = util.WrmUpCosinScheduler(
        optimizer, config["INFO"]["warmup epochs"], epochs, config["INFO"]["LR"]
    )

    info_log = op.train_eval(
        weights_folder,
        run_name,
        train_loader,
        val_loader,
        network,
        optimizer,
        scheduler,
        epochs,
        accumulation_step,
        logger,
        wandb,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args_def = {
        "config": {"type": str, "default": ""},
        "restart": {"type": bool, "default": True},
        "start_epoch": {"type": int, "default": 1},
    }
    for k, v in args_def.items():
        if v != args_def[k]["default"]:
            parser.add_argument(
                f"--{k}", default=args_def[k]["default"], type=args_def[k]["type"]
            )

    args = parser.parse_args()

    train_eval(args.config, args)
