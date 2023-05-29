#!/bin/sh
python src/train/nvcnet/vc/00001.py \
ml.exp_id=0 \
ml.fast_dev_run=false \
ml.art_output_dir=./data/outputs \
ml.log_output_dir=./log/outputs \
ml.batch_size=12 \
ml.epochs=500 \
model.target=nvcnet \
dataset=vc \
dataset.data_dir=/data/karanovc \
dataset.dataset_metadata_train_file=/nvc_net/results/val_karanovc_spk_list.txt \
dataset.dataset_metadata_val_file=/nvc_net/results/val_karanovc_spk_list.txt \
dataset.dataset_metadata_test_file=/nvc_net/results/val_karanovc_spk_list.txt \
dataset.speaker_list_file=/nvc_net/results/karanovc_spk_list.txt \
model.label_smooth=true \
model.adv_loss_type=mse \
model.sisnr_loss.is_use=true \
model.sisnr_loss.lambda_snr=5 \
model.spk_loss.is_use=true \
model.spk_loss.type=mse \
model.spk_loss.lambda_spk=5