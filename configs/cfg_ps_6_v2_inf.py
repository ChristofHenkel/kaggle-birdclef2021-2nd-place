from default_config import basic_cfg

cfg = basic_cfg
cfg.train_df = cfg.data_dir + "train_meta_4folded_v1.csv"
cfg.val_df = cfg.data_dir + "train_soundscape_labels_v2.csv"

# dataset
cfg.min_rating = 2.0

cfg.wav_crop_len = 30  # seconds

cfg.lr = 0.0001
cfg.epochs = 20
cfg.batch_size = 32
cfg.batch_size_val = 1
cfg.dataset = "ps_ds_2_inf"
cfg.model = "ps_model_3_inf2"
cfg.backbone = "resnet34"

cfg.num_workers = 32

cfg.save_val_data = True
cfg.mixed_precision = True

cfg.mixup = True
cfg.mix_beta = 1
