import torch

from tqdm import tqdm
def train_eval(WEIGHTS_FOLDER, train_loader,
              val_loader, network, optimizer, scheduler, epochs ):

    CE_loss = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pbar = tqdm(range(epochs))
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
            WEIGHTS_FOLDER.joinpath("_checkpoint_%d.pth" % epoch))
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
    return info_log
