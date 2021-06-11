from default_config import basic_cfg
import audiomentations as AA

cfg = basic_cfg
cfg.train_df = cfg.data_dir + "train_meta_4folded_v1.csv"
cfg.val_df = cfg.data_dir + "train_soundscape_labels_v2.csv"

# dataset
cfg.min_rating = 2.0

cfg.wav_crop_len = 30  # seconds

cfg.lr = 0.0005
cfg.epochs = 15
cfg.batch_size = 16
cfg.batch_size_val = 1
cfg.dataset = "ps_ds_2"
cfg.model = "ps_model_3"
cfg.backbone = "tf_efficientnetv2_s_in21k"

cfg.num_workers = 32

cfg.save_val_data = True
cfg.mixed_precision = True

cfg.mixup = True
cfg.mix_beta = 1

cfg.mel_norm = True

cfg.train_aug = AA.Compose(
    [
        AA.AddBackgroundNoise(
            sounds_path="input/ff1010bird_nocall/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.5
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/train_soundscapes/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/aicrowd2020_noise_30sec/noise_30sec", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
    ]
)
