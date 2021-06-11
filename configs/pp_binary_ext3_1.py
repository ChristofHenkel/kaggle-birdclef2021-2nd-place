from default_config import basic_cfg
import numpy as np

cfg = basic_cfg
cfg.train_df = cfg.data_dir + "binary_ext3.csv"
cfg.val_df = cfg.data_dir + "train_soundscape_labels_v2.csv"

# dataset
cfg.min_rating = 0.0
cfg.dataset = "pp_wav_ds_v8"
cfg.wav_crop_len = 10  # seconds

cfg.lr = 0.0004
cfg.epochs = 15
cfg.batch_size = 32
cfg.batch_size_val = 1
cfg.model = "pp_ch_att_seg4"
cfg.backbone = "tf_efficientnet_b0_ns"

cfg.num_workers = 32
cfg.save_val_data = True

cfg.mixed_precision = True
cfg.mix_beta = 1.0
cfg.mixup = 0.5
cfg.mixup2 = 0.5

cfg.birds = np.array(["bird"])
cfg.n_classes = len(cfg.birds)

cfg.mel_norm = True
