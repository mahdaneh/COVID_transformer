from torchvision import models
from tqdm import tqdm

from network import *


def train_eval(
        weights_fldr,
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
):
    CE_loss = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        tr_acc, tr_loss, val_acc, val_loss = 0, 0, 0, 0
        network.train()
        optimizer.zero_grad()
        for b, data in enumerate(train_loader):
            images = data[0].to(device)
            labels = data[1].to(device)

            prediction = network(images)
            loss = CE_loss(prediction, labels)
            loss = loss / accumulation_step
            loss.backward()
            if (b + 1) % accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            tr_loss += accumulation_step * loss.item()
            class_prediction = torch.argmax(prediction, dim=1)

            tr_acc += torch.mean(
                torch.where(class_prediction == labels, 1.0, 0.0)
            ).item()
            pbar.set_description(
                "epoch %d batch %d: tr accuracy %f" % (epoch, b, tr_acc / (b + 1))
            )

        # the last batch
        if (b + 1) % accumulation_step != 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        info_log = {
            "Epoch": epoch,
            "LR": scheduler.get_last_lr()[-1].item(),
            "tr_loss": tr_loss / (b + 1),
            "tr_acc": tr_acc / (b + 1),
        }

        pbar.set_description(str(info_log))

        torch.save(
            network.state_dict(),
            weights_fldr.joinpath("%s_checkpoint_%d.pth" % (run_name, epoch)),
        )

        if epoch % 5 == 0 or (epochs - epoch) < 5:  # every 5 epochs
            log, _, _ = evaluation(network, val_loader, loss=CE_loss)
            info_log |= log

        print(str(info_log))
        logger.info(info_log)
        wandb.log(info_log)
    return info_log


def evaluation(
        network, val_loader, device=torch.device("cuda"), loss=torch.nn.CrossEntropyLoss()
):
    val_loss = 0
    val_acc = 0
    predictions = []
    true_labels = []
    network.eval()
    with torch.no_grad():
        for eval_b, data in enumerate(val_loader):
            images, labels = data[0].to(device), data[1].to(device)
            prediction = network(images)
            val_loss += loss(prediction, labels).item()

            class_prediction = torch.argmax(prediction, dim=1)

            predictions.extend(class_prediction.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            val_acc += torch.mean(
                torch.where(class_prediction == labels, 1.0, 0.0)
            ).item()
        info_log = {
            "eval_loss": val_loss / (eval_b + 1),
            "eval_acc": val_acc / (eval_b + 1),
        }
        return info_log, predictions, true_labels


def build_model(config):
    net_name = config["INFO"]["net name"]
    if net_name == "VIT":
        network = VIT(config["VIT"])
    elif net_name == "ResNet":
        network = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
        for child in list(backbone.children())[:2]:
            for p in child.parameters():
                p.requires_grad = False
        # backbone.requires_grad_(False)  # to freeze backbone
        vit = VIT(config["VIT"])
        network = nn.Sequential(backbone, vit)
    elif net_name == "tv_vit_b_16":
        network = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise NotImplementedError

    return network


