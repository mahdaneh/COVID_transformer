import json
import logging
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import dataset_repo as d_repo
import util as util
from network import VIT, SimpleCNN
import Operations as op

CONFIG_FILE = (
    "configs/Res_VIT.json"  #'configs/Resnet_QU_EX.json'#"configs/VIT_QU_EX.json"
)
WEIGHTS_FOLDER = Path("weights/")
LOG_FOLDER = Path("logs/")


def main():
    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)
    epochs = config["INFO"]["epochs"]
    net_name = config["INFO"]["net name"]
    wandb.init(project="COVID", name=CONFIG_FILE, config=config)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=CONFIG_FILE.replace("configs", "logs").replace(".json", ".log"),
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
            weights=models.ResNet18_Weights.DEFAULT
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
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone = nn.Sequential(*list(resnet.children())[:4])
        vit = VIT(config["VIT"]).to(device)
        # module_dict = nn.ModuleDict({"backbone":backbone,
        #     "VIT": vit})
        network = nn.Sequential(backbone, vit)

    network.to(device)
    wandb.watch(network, log="all")
    logger.info("%s model on device: %s", net_name, next(network.parameters()).is_cuda)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["INFO"]["LR"])
    scheduler = util.WrmUpCosinScheduler(optimizer, 20, epochs, config["INFO"]["LR"])

    with logging_redirect_tqdm():
        info_log = op.train_eval(
            WEIGHTS_FOLDER,
            train_loader,
            val_loader,
            network,
            optimizer,
            scheduler,
            epochs,
        )
        logger.info(info_log)
        wandb.log(info_log)


if __name__ == "__main__":
    main()
