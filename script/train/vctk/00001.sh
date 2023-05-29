#!/bin/sh
python src/train/nvcnet/vctk/00001.py \
ml.exp_id=1 \
ml.fast_dev_run=False \
ml.art_output_dir=./data/outputs \
ml.log_output_dir=./log/outputs \
model.target=nvcnet \
dataset=vctk \
dataset.data_dir=data/vctk/train/data \
dataset.dataset_metadata_train_file=./data/vctk/train/metadata_train.csv \
dataset.dataset_metadata_val_file=./data/vctk/val/metadata_train.csv \
dataset.dataset_metadata_test_file=./data/vctk/val/metadata_train.csv \
dataset.speaker_list_file=./src/dataset/vctk/spk/list_of_speakers.txt
