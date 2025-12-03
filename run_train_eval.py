import argparse
import json
import logging
from pathlib import Path

import torch
# import wandb
import trackio as wandb
from torch.utils.data import DataLoader

import Operations as op
import dataset_repo as d_repo
import util as util


def train_eval(config_file, args):
    weights_folder = Path("weights/")
    run_name = config_file.removeprefix("configs/").removesuffix(".json")

    with open(config_file, "r") as cf:
        config = json.load(cf)
    epochs = config["INFO"]["epochs"]
    accumulation_step = config["INFO"]["accumulation step"]
    log_filename = Path(config_file.replace("configs", "logs").replace(".json", ".log"))

    print("Logging to {}".format(log_filename))

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
        training=False,
        mode="test",
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

    network = op.build_model(config)
    # network = torch.compile(network)
    network.to(device)

    # wandb.watch(network, log="all")
    logger.info("model on device: %s", next(network.parameters()).is_cuda)

    optimizer = torch.optim.AdamW(network.parameters(), lr=config["INFO"]["LR"])
    scheduler = util.WrmUpCosinScheduler(
        optimizer, config["INFO"]["warmup epochs"], epochs, config["INFO"]["LR"]
    )

    info_log = op.train_valid(
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
        "resume_epoch": {"type": int, "default": 1},
    }
    for k, v in args_def.items():
        if v != args_def[k]["default"]:
            parser.add_argument(
                f"--{k}", default=args_def[k]["default"], type=args_def[k]["type"]
            )

    args = parser.parse_args()

    train_eval(args.config, args)
