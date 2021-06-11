import os
from types import SimpleNamespace
import numpy as np

cfg = SimpleNamespace(**{})

# paths
cfg.data_folder = ''
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.data_dir = "input/"
cfg.train_data_folder = cfg.data_dir + "train_short_audio/"
cfg.val_data_folder = cfg.data_dir + "train_soundscapes/"
cfg.output_dir = f"output/models/{os.path.basename(__file__).split('.')[0]}"

# dataset
cfg.dataset = "base_ds"
cfg.min_rating = 0
cfg.n_classes = 398
cfg.batch_size = 32
cfg.val_df = None
cfg.batch_size_val = 1
cfg.train_aug = None
cfg.val_aug = None
cfg.test_augs = None
cfg.wav_len_val = 5  # seconds

# audio
cfg.window_size = 2048
cfg.hop_size = 512
cfg.sample_rate = 32000
cfg.fmin = 16
cfg.fmax = 16386
cfg.power = 2
cfg.mel_bins = 256
cfg.top_db = 80.0

# img model
cfg.backbone = "resnet18"
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.train = True
cfg.val = True
cfg.in_chans = 1

cfg.alpha = 1
cfg.eval_epochs = 1
cfg.eval_train_epochs = 1
cfg.warmup = 0

cfg.mel_norm = False

cfg.label_smoothing = 0

cfg.remove_pretrained = []

# training
cfg.lr = 1e-4
cfg.epochs = 10
cfg.seed = -1
cfg.save_val_data = True

# ressources
cfg.mixed_precision = True
cfg.gpu = 0
cfg.num_workers = 4
cfg.drop_last = True

cfg.mixup = 0
cfg.mixup2 = 0
cfg.mix_beta = 0.5

cfg.label_smoothing = 0

cfg.mixup_2x = False

cfg.birds = np.array(['acafly', 'acowoo', 'aldfly', 'ameavo', 'amecro', 'amegfi',
       'amekes', 'amepip', 'amered', 'amerob', 'amewig', 'amtspa',
       'andsol1', 'annhum', 'astfly', 'azaspi1', 'babwar', 'baleag',
       'balori', 'banana', 'banswa', 'banwre1', 'barant1', 'barswa',
       'batpig1', 'bawswa1', 'bawwar', 'baywre1', 'bbwduc', 'bcnher',
       'belkin1', 'belvir', 'bewwre', 'bkbmag1', 'bkbplo', 'bkbwar',
       'bkcchi', 'bkhgro', 'bkmtou1', 'bknsti', 'blbgra1', 'blbthr1',
       'blcjay1', 'blctan1', 'blhpar1', 'blkpho', 'blsspa1', 'blugrb1',
       'blujay', 'bncfly', 'bnhcow', 'bobfly1', 'bongul', 'botgra',
       'brbmot1', 'brbsol1', 'brcvir1', 'brebla', 'brncre', 'brnjay',
       'brnthr', 'brratt1', 'brwhaw', 'brwpar1', 'btbwar', 'btnwar',
       'btywar', 'bucmot2', 'buggna', 'bugtan', 'buhvir', 'bulori',
       'burwar1', 'bushti', 'butsal1', 'buwtea', 'cacgoo1', 'cacwre',
       'calqua', 'caltow', 'cangoo', 'canwar', 'carchi', 'carwre',
       'casfin', 'caskin', 'caster1', 'casvir', 'categr', 'ccbfin',
       'cedwax', 'chbant1', 'chbchi', 'chbwre1', 'chcant2', 'chispa',
       'chswar', 'cinfly2', 'clanut', 'clcrob', 'cliswa', 'cobtan1',
       'cocwoo1', 'cogdov', 'colcha1', 'coltro1', 'comgol', 'comgra',
       'comloo', 'commer', 'compau', 'compot1', 'comrav', 'comyel',
       'coohaw', 'cotfly1', 'cowscj1', 'cregua1', 'creoro1', 'crfpar',
       'cubthr', 'daejun', 'dowwoo', 'ducfly', 'dusfly', 'easblu',
       'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eletro',
       'eucdov', 'eursta', 'fepowl', 'fiespa', 'flrtan1', 'foxspa',
       'gadwal', 'gamqua', 'gartro1', 'gbbgul', 'gbwwre1', 'gcrwar',
       'gilwoo', 'gnttow', 'gnwtea', 'gocfly1', 'gockin', 'gocspa',
       'goftyr1', 'gohque1', 'goowoo1', 'grasal1', 'grbani', 'grbher3',
       'grcfly', 'greegr', 'grekis', 'grepew', 'grethr1', 'gretin1',
       'greyel', 'grhcha1', 'grhowl', 'grnher', 'grnjay', 'grtgra',
       'grycat', 'gryhaw2', 'gwfgoo', 'haiwoo', 'heptan', 'hergul',
       'herthr', 'herwar', 'higmot1', 'hofwoo1', 'houfin', 'houspa',
       'houwre', 'hutvir', 'incdov', 'indbun', 'kebtou1', 'killde',
       'labwoo', 'larspa', 'laufal1', 'laugul', 'lazbun', 'leafly',
       'leasan', 'lesgol', 'lesgre1', 'lesvio1', 'linspa', 'linwoo1',
       'littin1', 'lobdow', 'lobgna5', 'logshr', 'lotduc', 'lotman1',
       'lucwar', 'macwar', 'magwar', 'mallar3', 'marwre', 'mastro1',
       'meapar', 'melbla1', 'monoro1', 'mouchi', 'moudov', 'mouela1',
       'mouqua', 'mouwar', 'mutswa', 'naswar', 'norcar', 'norfli',
       'normoc', 'norpar', 'norsho', 'norwat', 'nrwswa', 'nutwoo',
       'oaktit', 'obnthr1', 'ocbfly1', 'oliwoo1', 'olsfly', 'orbeup1',
       'orbspa1', 'orcpar', 'orcwar', 'orfpar', 'osprey', 'ovenbi1',
       'pabspi1', 'paltan1', 'palwar', 'pasfly', 'pavpig2', 'phivir',
       'pibgre', 'pilwoo', 'pinsis', 'pirfly1', 'plawre1', 'plaxen1',
       'plsvir', 'plupig2', 'prowar', 'purfin', 'purgal2', 'putfru1',
       'pygnut', 'rawwre1', 'rcatan1', 'rebnut', 'rebsap', 'rebwoo',
       'redcro', 'reevir1', 'rehbar1', 'relpar', 'reshaw', 'rethaw',
       'rewbla', 'ribgul', 'rinkin1', 'roahaw', 'robgro', 'rocpig',
       'rotbec', 'royter1', 'rthhum', 'rtlhum', 'ruboro1', 'rubpep1',
       'rubrob', 'rubwre1', 'ruckin', 'rucspa1', 'rucwar', 'rucwar1',
       'rudpig', 'rudtur', 'rufhum', 'rugdov', 'rumfly1', 'runwre1',
       'rutjac1', 'saffin', 'sancra', 'sander', 'savspa', 'saypho',
       'scamac1', 'scatan', 'scbwre1', 'scptyr1', 'scrtan1', 'semplo',
       'shicow', 'sibtan2', 'sinwre1', 'sltred', 'smbani', 'snogoo',
       'sobtyr1', 'socfly1', 'solsan', 'sonspa', 'soulap1', 'sposan',
       'spotow', 'spvear1', 'squcuc1', 'stbori', 'stejay', 'sthant1',
       'sthwoo1', 'strcuc1', 'strfly1', 'strsal1', 'stvhum2', 'subfly',
       'sumtan', 'swaspa', 'swathr', 'tenwar', 'thbeup1', 'thbkin',
       'thswar1', 'towsol', 'treswa', 'trogna1', 'trokin', 'tromoc',
       'tropar', 'tropew1', 'tuftit', 'tunswa', 'veery', 'verdin',
       'vigswa', 'warvir', 'wbwwre1', 'webwoo1', 'wegspa1', 'wesant1',
       'wesblu', 'weskin', 'wesmea', 'westan', 'wewpew', 'whbman1',
       'whbnut', 'whcpar', 'whcsee1', 'whcspa', 'whevir', 'whfpar1',
       'whimbr', 'whiwre1', 'whtdov', 'whtspa', 'whwbec1', 'whwdov',
       'wilfly', 'willet1', 'wilsni1', 'wiltur', 'wlswar', 'wooduc',
       'woothr', 'wrenti', 'y00475', 'yebcha', 'yebela1', 'yebfly',
       'yebori1', 'yebsap', 'yebsee1', 'yefgra1', 'yegvir', 'yehbla',
       'yehcar1', 'yelgro', 'yelwar', 'yeofly1', 'yerwar', 'yeteup1',
       'yetvir', 'rocpig1'])

basic_cfg = cfg
