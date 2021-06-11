from torch.utils.data import Dataset
import torch
import numpy as np
import librosa
import ast


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = None
val_collate_fn = None


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.bird2id = {bird: id_ for id_, bird in enumerate(cfg.birds)}
        self.df = df.copy()

        if self.mode == "train":
            self.data_folder = cfg.train_data_folder
            self.df = self.df[self.df["rating"] >= self.cfg.min_rating]
        elif self.mode == "val":
            self.data_folder = cfg.val_data_folder
        elif self.mode == "test":
            self.data_folder = cfg.test_data_folder

        self.fns = self.df["filename"].unique()

        self.df = self.setup_df()

        self.aug_audio = cfg.train_aug
        print(self.aug_audio)

    def setup_df(self):
        df = self.df.copy()

        if self.mode == "train":

            df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
            label = df["primary_label"].values
            labels = (label == "bird").astype(float)
            if self.cfg.label_smoothing > 0:
                labels = np.clip(labels + self.cfg.label_smoothing, 0, 1)

        else:

            df["weight"] = 1
            label = df["birds"].values
            labels = 1 - (label == "nocall").astype(float)

        df[[f"t{i}" for i in range(self.cfg.n_classes)]] = labels

        if self.mode != "train":
            df = df.groupby("filename")

        return df

    def __getitem__(self, idx):

        if self.mode == "train":
            row = self.df.iloc[idx]
            fn = row["primary_label"] + "/" + row["filename"]
            label = row[[f"t{i}" for i in range(self.cfg.n_classes)]].values
            weight = row["weight"]
            wav_len = row["length"]
            parts = 1
        else:
            fn = self.fns[idx]
            row = self.df.get_group(fn)
            label = row[[f"t{i}" for i in range(self.cfg.n_classes)]].values
            weight = 1
            wav_len = None
            parts = label.shape[0]

        if self.mode == "train":
            wav_len_sec = wav_len / self.cfg.sample_rate
            duration = self.cfg.wav_crop_len
            max_offset = wav_len_sec - duration
            max_offset = max(max_offset, 1)
            offset = np.random.randint(max_offset)
        else:
            offset = 0.0
            duration = None

        wav = self.load_one(fn, offset, duration)

        # PAD here?:
        if self.mode == "train":
            if wav.shape[0] < (self.cfg.wav_crop_len * self.cfg.sample_rate):
                pad = self.cfg.wav_crop_len * self.cfg.sample_rate - wav.shape[0]
                wav = np.pad(wav, (0, pad))

        if self.aug_audio:
            wav = self.aug_audio(samples=wav, sample_rate=self.cfg.sample_rate)

        wav_tensor = torch.tensor(wav)  # (n_samples)
        if parts > 1:
            n_samples = wav_tensor.shape[0]
            wav_tensor = wav_tensor[: n_samples // parts * parts].reshape(
                parts, n_samples // parts
            )

        feature_dict = {
            "input": wav_tensor,
            "target": torch.tensor(label.astype(np.float32)),
            "weight": torch.tensor(weight),
        }
        return feature_dict

    def __len__(self):
        return len(self.fns)

    def load_one(self, id_, offset, duration):
        fp = self.data_folder + id_
        try:
            wav, sr = librosa.load(fp, sr=None, offset=offset, duration=duration)
        except:
            print("FAIL READING rec", fp)

        return wav

    def birds2target(self, birds):
        birds = birds.split()
        target = [self.bird2id.get(item, None) for item in birds if not item == "nocall"]
        target = [item for item in target if item is not None]
        return target

    def secondary2target(self, secondary_label):
        birds = ast.literal_eval(secondary_label)
        target = [self.bird2id.get(item) for item in birds if not item == "nocall"]
        return target
