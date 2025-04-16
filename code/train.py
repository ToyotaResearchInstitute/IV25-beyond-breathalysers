import argparse
import os
import pickle
import warnings
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader

from dataset import VisuomotorDataset, fetch_participants
from model import load_model
from utils import RunningLoss, load_config_as_box, set_random_seed


warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils.checkpoint: the use_reentrant parameter")
warnings.filterwarnings("ignore", category=UserWarning, message="None of the inputs have requires_grad=True")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Only reconstruction head is pre-trained. Classification and forecasting heads must be fine-tuned.",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.",
)

def main():
    log_wandb = True
    positive_sample_size = 4
    negative_sample_size = 6

    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--model_name", type=str, help="Override model")
    parser.add_argument("--train_exp", nargs="+", help="Override training experiments to use")
    parser.add_argument("--name", type=str, help="Override exp name")
    parser.add_argument("--cfg", type=str, help="Override config file")

    args = parser.parse_args()

    # load YAML config
    if args.cfg:
        cfg = load_config_as_box(f"{args.cfg}")
    else:
        cfg = load_config_as_box(f"code/configs/base.yaml")

    # make sure that cfg.results.path exists, if not, create it
    cfg.results.path = os.path.expanduser(cfg.results.path)
    if not os.path.exists(cfg.results.path):
        print(f"Making path {cfg.results.path}")
        os.makedirs(cfg.results.path)

    # override if provided
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.seed:
        cfg.train.random_seed = args.seed
    if args.learning_rate:
        cfg.train.learning_rate = args.learning_rate
    if args.train_exp:
        cfg.train.exp_to_use = args.train_exp
    if args.name:
        cfg.name = args.name
    if args.model_name:
        cfg.model.name = args.model_name

    # train on GPU if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # fix random seed
    set_random_seed(cfg.train.random_seed)

    # get participant list
    participant_list = fetch_participants(data_path=cfg.data.path)

    num_v1_subjects = len([p for p in participant_list if p[0] == "P"])
    num_runs = num_v1_subjects // positive_sample_size

    # different hold-out sets
    for ii in range(num_runs):
        all_ids = participant_list.copy()
        pids = {}
        v1_subjects = deque(p for p in all_ids if p[0] == "P")
        v1_subjects.rotate(-ii * positive_sample_size)
        v1_subjects = [_ for _ in v1_subjects]
        v2_subjects = deque(p for p in all_ids if p[0] == "7")
        v2_subjects.rotate(-ii * negative_sample_size)
        v2_subjects = [_ for _ in v2_subjects]

        pids["val"] = [v1_subjects.pop(0) for jj in range(positive_sample_size)]
        pids["val"] += [v2_subjects.pop(0) for jj in range(negative_sample_size)]
        pids["train"] = v1_subjects + v2_subjects
        # rely on multi-round x-validation, treating average val performance as an effective estimate of test
        print(f"Run {ii+1}/{num_runs}, val on {pids['val']}")
        pids["test"] = pids["val"]
        cfg["pids"] = pids

        # set up data
        sd_train = VisuomotorDataset(pids, "train", cfg)
        sd_val = VisuomotorDataset(pids, "val", cfg, norm_data=sd_train.get_norm_data())
        sd_test = VisuomotorDataset(pids, "test", cfg, norm_data=sd_train.get_norm_data())

        # check things are working
        _tr = sd_train[0]
        _va = sd_val[0]
        _te = sd_test[0]
        
        # dataloaders
        dl_train = DataLoader(sd_train, cfg.train.batch_size, shuffle=True, drop_last=False, num_workers=8)
        dl_val = DataLoader(sd_val, batch_size=1, shuffle=False, drop_last=False)
        dl_test = DataLoader(sd_test, batch_size=1, shuffle=False, drop_last=False)

        exp_name = f"base-{cfg.name}-split-{ii:02d}"
        output_dir = os.path.join(cfg.results.path, exp_name)
        print(f"Making path {output_dir}")
        os.makedirs(output_dir)

        if log_wandb:
            wandb.init(
                project="IV25-beyond-breathalysers",
                name=f"{exp_name}",
                dir=output_dir,
                config=cfg,
            )

        # run experiment
        model = load_model(cfg)
        model.to(device).float()

        class_counts = torch.tensor([30, 20]).to(device)
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        criterion = nn.CrossEntropyLoss(weight=weights)

        current_lr = cfg.train.learning_rate
        optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)
        if cfg.train.scheduler.use:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.train.scheduler.step_size, gamma=cfg.train.scheduler.gamma
            )
        else:
            pass

        rl = RunningLoss()

        # train & val loop
        num_epochs = cfg.train.epochs
        for epoch in range(num_epochs):
            model.train()
            rl.reset()
            count = 0

            for inputs, labels, _ in dl_train:
                optimizer.zero_grad()

                # prepare inputs and labels
                inputs = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))
                labels = labels.view(-1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                if log_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            f"epoch_{epoch}_first_sample": wandb.plot.line_series(
                                xs=np.array(list(range(inputs.shape[-1]))),
                                ys=[
                                    inputs[0, 1, 0, :].cpu().numpy(),
                                ],
                                xname="Time steps",
                            ),
                        }
                    )

                # forward pass
                outputs = model(inputs)

                # compute loss and optimize
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # compute stats
                _, predicted = torch.max(outputs, 1)
                
                rl.update_train(loss.item(), predicted, labels)

            # validation
            model.eval()

            with torch.no_grad():
                for inputs, labels, _ in dl_val:
                    # prepare inputs and labels
                    inputs = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))
                    labels = labels.view(-1)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward pass, no gradients
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    # compute statistics
                    _, predicted = torch.max(outputs, 1)
                    rl.update_val(loss.item(), predicted, labels)

            tpr_train, fpr_train, F1_train = rl.get_tpr_fpr_F1("train")
            tpr_val, fpr_val, F1_val = rl.get_tpr_fpr_F1("val")

            print(
                f"Epoch [{epoch+1:04d}/{num_epochs}], Loss: {rl.avg_train_loss:.4f}, Accuracy: {rl.avg_train_acc:.2f}%, TPR: {tpr_train:.2f}, FPR: {fpr_train:.2f} F1: {F1_train:.2f} | Val: {rl.avg_val_loss:.4f}, Accuracy: {rl.avg_val_acc:.2f}%, TPR: {tpr_val:.2f}, FPR: {fpr_val:.2f}  F1: {F1_val:.2f} | LR: {current_lr:.6f}"
            )
            if log_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": rl.avg_train_loss,
                        "train_acc": rl.avg_train_acc,
                        "train_tpr": tpr_train,
                        "train_fpr": fpr_train,
                        "val_loss": rl.avg_val_loss,
                        "val_acc": rl.avg_val_acc,
                        "val_tpr": tpr_val,
                        "val_fpr": fpr_val,
                        "learning_rate": current_lr,
                    }
                )

            if cfg.train.scheduler.use:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = cfg.train.learning_rate

        # run test
        model.eval()
        results = {}
        with torch.no_grad():
            for inputs, labels, (exp, pid) in dl_test:
                # prepare inputs and labels
                inputs = inputs.view(-1, inputs.size(2), inputs.size(3), inputs.size(4))
                labels = labels.view(-1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass, no gradients
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                # compute statistics
                _, predicted = torch.max(outputs, 1)
                exp = exp[0]
                pid = pid[0]
                if exp not in results:
                    results[exp] = {}
                if pid not in results[exp]:
                    results[exp][pid] = {
                        "outputs": outputs.cpu().numpy(),
                        "predicted": predicted.cpu().numpy(),
                        "labels": labels.cpu().numpy(),
                    }
                else:
                    raise ValueError

        if log_wandb:
            # save the model
            model_save_path = f"{wandb.run.dir}/model.pth"
            torch.save(model.state_dict(), model_save_path)
            results_save_path = f"{wandb.run.dir}/results.pkl"
            # saving dictionary as pickle file
            with open(results_save_path, "wb") as file:
                pickle.dump(results, file)
            wandb.finish()


if __name__ == "__main__":
    main()
