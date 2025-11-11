import json
import logging
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import dataset_repo as d_repo
import util as util
from network import VIT, SimpleCNN

CONFIG_FILE = "VIT_QU_EX_config.json"
WEIGHTS_FOLDER = Path("weights/")


def main():
    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)

    net_name = config["INFO"]["net name"]
    wandb.init(project="COVID", name=CONFIG_FILE, config=config)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=CONFIG_FILE.replace("config.json", "_loging.log"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logger.info("Logging started")
    logger.info(", ".join(f"{key}: {value}" for key, value in config.items()))

    train_dataset = d_repo.Covid_QU_Ex(
        config["DATA"],
        image_size=config[net_name]["image size"][1:],
        training=True,
        mode="train",
    )
    val_dataset = d_repo.Covid_QU_Ex(
        config["DATA"],
        image_size=config[net_name]["image size"][1:],
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
    else:
        network = SimpleCNN().to(device)

    wandb.watch(network, log="all")
    logger.info("%s model on device: %s", net_name, next(network.parameters()).is_cuda)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["INFO"]["LR"])
    scheduler = util.WrmUpCosinScheduler(
        optimizer, 20, config["INFO"]["epochs"], config["INFO"]["LR"]
    )

    CE_loss = torch.nn.CrossEntropyLoss()
    with logging_redirect_tqdm():
        pbar = tqdm(range(config["INFO"]["epochs"]))
        for epoch in pbar:
            tr_acc, tr_loss, val_acc, val_loss = 0, 0, 0, 0
            network.train()
            for b, data in enumerate(train_loader):
                images = data[0].to(device)
                labels = data[1].to(device)

                optimizer.zero_grad()
                prediction = network(images)
                loss = CE_loss(prediction, labels)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                class_prediction = torch.argmax(prediction, dim=1)

                tr_acc += torch.mean(
                    torch.where(class_prediction == labels, 1.0, 0.0)
                ).item()
            scheduler.step()
            info_log = {
                "Epoch": epoch,
                "LR": scheduler.get_last_lr(),
                "tr_loss": tr_loss / (b + 1),
                "tr_acc": tr_acc / (b + 1),
            }

            pbar.set_description(str(info_log))

            torch.save(
                network.state_dict(),
                WEIGHTS_FOLDER.joinpath(
                    CONFIG_FILE.replace(".json", "_checkpoint_%d.pth" % epoch)
                ),
            )
            if epoch % 5 == 0:  # every 5 epochs
                network.eval()
                with torch.no_grad():
                    for eval_b, data in enumerate(val_loader):
                        images, labels = data[0].to(device), data[1].to(device)
                        prediction = network(images)
                        loss = CE_loss(prediction, labels)
                        val_loss += loss.item()
                        class_prediction = torch.argmax(prediction, dim=1)
                        val_acc += torch.mean(
                            torch.where(class_prediction == labels, 1.0, 0.0)
                        ).item()

                info_log |= {
                    "eval_loss": val_loss / (eval_b + 1),
                    "eval_acc": val_acc / (eval_b + 1),
                }
                pbar.set_description(str(info_log))

            logger.info(info_log)
            wandb.log(info_log)

        # end of full training


if __name__ == "__main__":
    main()
