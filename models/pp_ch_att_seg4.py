import timm
from torch import nn
import torch
import torchaudio as ta
from torch.cuda.amp import autocast
import numpy as np
from model_utils import GeM, Mixup


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg

        self.n_classes = cfg.n_classes

        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.window_size,
            win_length=cfg.window_size,
            hop_length=cfg.hop_size,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            pad=0,
            n_mels=cfg.mel_bins,
            power=cfg.power,
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)

        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.in_chans,
        )

        if "efficientnet" in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.global_pool = GeM()

        self.head = nn.Linear(backbone_out, self.n_classes)
        self.bn0 = nn.InstanceNorm2d(1)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self.mixup = Mixup(mix_beta=cfg.mix_beta)
        self.att = nn.Sequential(nn.Linear(backbone_out, 512), nn.ReLU(), nn.Linear(512, 1))

        if cfg.pretrained_weights is not None:
            sd = torch.load(cfg.pretrained_weights, map_location="cpu")["model"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)
            print("weights loaded from", cfg.pretrained_weights)

    def forward(self, batch):

        if self.training:
            x = batch["input"]
            y = batch["target"]
            bs, time = x.shape

        else:
            x = batch["input"]
            bs, parts, time = x.shape
            x = x.reshape(parts, time)
            y = batch["target"]
            y = y[0]

        with autocast(enabled=False):
            x = self.wav2img(x)  # (8, 256, 1876)
            if self.cfg.mel_norm:
                x = (x + 80) / 80

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]
        x = self.bn0(x)

        weight = batch["weight"]

        if self.training:
            if np.random.random() <= self.cfg.mixup:
                x, y, weight = self.mixup(x, y, weight)
            if np.random.random() <= self.cfg.mixup2:
                x, y, weight = self.mixup(x, y, weight)

        x = self.backbone(x)  # (8, 512, 59, 8)
        x = x.mean(3)  # pool freq
        x = x.permute(0, 2, 1)  # bs, time, feats

        att_weights = torch.softmax(self.att(x), dim=1)
        x2 = (x * att_weights).sum(1)

        logits = self.head(x2)  # (batch_size, classes)

        loss = self.loss_fn(logits, y)

        loss = (loss.mean(dim=1) * weight) / weight.sum()
        loss = loss.sum()

        return {"loss": loss, "logits": logits.sigmoid(), "target": y}
