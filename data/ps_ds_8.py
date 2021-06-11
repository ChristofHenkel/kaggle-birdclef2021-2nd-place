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
        self.df = df.copy()

        self.bird2id = {bird: idx for idx, bird in enumerate(cfg.birds)}
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

    def setup_df(self):
        df = self.df.copy()

        if self.mode == "train":

            df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

            labels = np.eye(self.cfg.n_classes)[df["target"].astype(int).values]
            label2 = df["secondary_labels"].apply(lambda x: self.secondary2target(x)).values
            for i, t in enumerate(label2):
                labels[i, t] = 1
        else:
            targets = df["birds"].apply(lambda x: self.birds2target(x)).values
            labels = np.zeros((df.shape[0], self.cfg.n_classes))
            for i, t in enumerate(targets):
                labels[i, t] = 1

        df[[f"t{i}" for i in range(self.cfg.n_classes)]] = labels

        if self.mode != "train":
            df = df.groupby("filename")

        return df

    def __getitem__(self, idx):

        if self.mode == "train":
            row = self.df.iloc[idx]
            if row["filename"][:4] == "2020":
                fn = row["filename"]
            else:
                fn = row["primary_label"] + "/" + row["filename"]

            label = row[[f"t{i}" for i in range(self.cfg.n_classes)]].values
            weight = row["weight"]
            fold = row["fold"]

            if self.cfg.label_smoothing > 0:
                label = np.clip(label + self.cfg.label_smoothing, 0, 1)

            wav_len = row["length"]
            parts = 1
        else:
            fn = self.fns[idx]
            row = self.df.get_group(fn)
            label = row[[f"t{i}" for i in range(self.cfg.n_classes)]].values
            wav_len = None
            parts = label.shape[0]
            fold = -1
            weight = 1

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

        expected_length = self.cfg.wav_crop_len * self.cfg.sample_rate
        if wav.shape[0] < expected_length:
            pad = self.cfg.wav_crop_len * self.cfg.sample_rate - wav.shape[0]

            wav_orig = wav.copy()

            l = wav.shape[0]

            if pad >= l:
                while wav.shape[0] <= expected_length:
                    wav = np.concatenate([wav, wav_orig], axis=0)
            else:
                max_offset = l - pad
                offset = np.random.randint(max_offset)
                wav = np.concatenate([wav, wav_orig[offset : offset + pad]], axis=0)

            wav = wav[:expected_length]

        if self.mode == "train":
            if self.aug_audio:
                wav = self.aug_audio(samples=wav, sample_rate=self.cfg.sample_rate)
        else:
            if self.cfg.val_aug:
                wav = self.cfg.val_aug(samples=wav, sample_rate=self.cfg.sample_rate)

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
            "fold": torch.tensor(fold),
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
        target = [self.bird2id.get(item) for item in birds if not item == "nocall"]
        return target

    def secondary2target(self, secondary_label):
        birds = ast.literal_eval(secondary_label)
        target = [self.bird2id.get(item) for item in birds if not item == "nocall"]
        return target
