import torch

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


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
        print(str(info_log))
        logger.info(info_log)
        wandb.log(info_log)
    return info_log
