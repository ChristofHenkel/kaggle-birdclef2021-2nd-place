set -e

for SEED in {0..5}
do
python train.py --seed=$SEED "$@"
done
