from default_config import basic_cfg

cfg = basic_cfg
cfg.train_df = cfg.data_dir + "train_meta_4folded_v1.csv"
cfg.val_df = cfg.data_dir + "train_soundscape_labels_v2.csv"

# dataset
cfg.min_rating = 0.0

cfg.wav_crop_len = 30  # seconds

cfg.lr = 0.001
cfg.epochs = 11
cfg.warmup = 1
cfg.batch_size = 16
cfg.batch_size_val = 1
cfg.dataset = "ps_ds_2"
cfg.model = "ps_model_9"
cfg.backbone = "eca_nfnet_l0"

cfg.num_workers = 32

cfg.save_val_data = True
cfg.mixed_precision = True

cfg.mixup = 0.5
cfg.mixup2 = 0.5
cfg.mix_beta = 1

cfg.mel_norm = True

cfg.window_size = 1024
cfg.hop_size = 320
cfg.fmin = 50
cfg.fmax = 14000
cfg.mel_bins = 128
cfg.top_db = None
