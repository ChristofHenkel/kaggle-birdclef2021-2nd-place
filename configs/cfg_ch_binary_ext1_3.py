from default_config import basic_cfg
import numpy as np

cfg = basic_cfg
cfg.train_df = cfg.data_dir + "binary_ext1.csv"
cfg.val_df = cfg.data_dir + "train_soundscape_labels_v2.csv"

# dataset
cfg.min_rating = 4.0
cfg.dataset = "wav_ds_v8"
cfg.wav_crop_len = 10  # seconds

cfg.lr = 0.0001
cfg.epochs = 20
cfg.batch_size = 32
cfg.batch_size_val = 1
cfg.model = "ch_att_seg2"
cfg.backbone = "seresnext26t_32x4d"

cfg.num_workers = 32

cfg.mixed_precision = True
cfg.mix_beta = 1.0
cfg.birds = np.array(["bird"])
cfg.n_classes = len(cfg.birds)
