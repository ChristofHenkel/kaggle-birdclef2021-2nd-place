import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
import cv2
from copy import copy
import os
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import SequentialSampler, DataLoader

cv2.setNumThreads(0)

sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("losses")
sys.path.append("utils")


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_train_dataloader(train_ds, cfg):
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataloader(val_ds, cfg):
    sampler = SequentialSampler(val_ds)
    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    )
    return scheduler


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_model(cfg, train_dataset):
    Net = importlib.import_module(cfg.model).Net
    return Net(cfg)


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("-s", "--seed", type=int, default=-1, help="seed")
parser_args, _ = parser.parse_known_args(sys.argv)


cfg = copy(importlib.import_module(parser_args.config).cfg)

if parser_args.seed > -1:
    cfg.seed = parser_args.seed

os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)

cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device


def run_eval(model, val_dataloader, cfg, pre="val"):

    model.eval()
    torch.set_grad_enabled(False)

    val_data = defaultdict(list)

    for data in tqdm(val_dataloader):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        for key, val in output.items():
            val_data[key] += [output[key]]

    for key, val in output.items():
        value = val_data[key]
        if len(value[0].shape) == 0:
            val_data[key] = torch.stack(value)
        else:
            val_data[key] = torch.cat(value, dim=0)

    if cfg.save_val_data:
        torch.save(val_data, f"{cfg.output_dir}/{pre}_data_seed{cfg.seed}.pth")

    if "loss" in val_data:
        val_losses = val_data["loss"].cpu().numpy()
        val_loss = np.mean(val_losses)
        print(f"Mean {pre}_loss", np.mean(val_losses))

    else:
        val_loss = 0.0

    print("EVAL FINISHED")

    return val_loss


if __name__ == "__main__":

    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)

    device = "cuda:%d" % cfg.gpu
    cfg.device = device

    set_seed(cfg.seed)

    train_df = pd.read_csv(cfg.train_df)
    val_df = pd.read_csv(cfg.val_df)

    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader = get_val_dataloader(val_dataset, cfg)

    model = get_model(cfg, train_dataset)
    model.to(device)

    total_steps = len(train_dataset)

    params = model.parameters()
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=0)

    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch)

        cfg.curr_epoch = epoch

        print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                cfg.curr_step += cfg.batch_size

                data = next(tr_it)

                model.train()
                torch.set_grad_enabled(True)

                batch = batch_to_device(data, device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())

                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if cfg.curr_step % cfg.batch_size == 0:
                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

        if cfg.val:
            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                val_loss = run_eval(model, val_dataloader, cfg)
            else:
                val_score = 0

        if cfg.epochs > 0:
            checkpoint = create_checkpoint(
                model, optimizer, epoch, scheduler=scheduler, scaler=scaler
            )

            torch.save(checkpoint, f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}.pth")

    if cfg.epochs > 0:
        checkpoint = create_checkpoint(model, optimizer, epoch, scheduler=scheduler, scaler=scaler)

        torch.save(checkpoint, f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}.pth")
