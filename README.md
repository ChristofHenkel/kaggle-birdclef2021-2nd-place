# kaggle-birdclef21
Our writeup for this solution can be found on [kaggle](https://www.kaggle.com/c/birdclef-2021/discussion/243463).

## Training
Download the Birdclef2021 dataset from [kaggle](https://www.kaggle.com/c/birdclef-2021/data) and extract the contents to the ./input/ folder.

For the binary classifier training additional data from [DCASE](http://dcase.community/challenge2018/task-bird-audio-detection) is used. The original datasets have been published under the Creative Commons Attribution licence [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). To allow for training of our binary classifier, we resampled and converted the provided data and uploaded to a kaggle dataset [here](https://www.kaggle.com/ilu000/2ndplacebirdclef2021-binary-data). Prior to binary classifier training, please extract the folders "bird" and "nocall" to ./input/train_short_audio/.

For the other models, additional data from [DCASE](http://dcase.community/challenge2018/task-bird-audio-detection) and from the BirdClef2020 challenge hosted by [Aicrowd](https://www.aicrowd.com/challenges/lifeclef-2020-bird-monophone) is used as background noise. The data has been resampled and converted and uploaded to a kaggle dataset [here](https://www.kaggle.com/christofhenkel/birdclef2021-background-noise). Prior to training, please extract the folders "aicrowd2020_noise_30sec", "ff1010bird_nocall" and "train_soundscapes" to ./input/ (note: train_soundscapes/nocall is identical to the data above and may be merged). For potential additional diversity, you can also extract 30s bird segments from old validation data containing same birds as 2021 data.

Training can be initialized with:

```sh
# -C flag is used to specify a config file
# replace NAME_OF_CONFIG with an appropiate config file name such as cfg_ps_6_v2
python train.py -C NAME_OF_CONFIG

```

After training, the last checkpoint (model weights) will be saved to the folder ./output/NAME_OF_CONFIG/


## Inference
Inference is published in a kaggle kernel [here](https://www.kaggle.com/ilu000/2nd-place-birdclef2021-inference/). Weights from our trained models are provided in a kaggle dataset linked to the inference kernel [here](https://www.kaggle.com/ilu000/2ndplacebirdclef2021-models).
