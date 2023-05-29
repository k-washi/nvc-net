# NVC-Net

[NVC-Net](https://github.com/sony/ai-research-code/tree/master/nvcnet ) のpytorch version

- BCE lossをmseに変更 (GとDの乖離を減らす)
- 信号の一致度をとるためsnr lossを追加
- spk embeddingの一貫性保証のため、cycle consistency embedding lossを追加
- label smoothingを追加


# 実行環境作成

```
docker-compose up -d
```

```
poetry install
source .venv/bin/activate
pip install -e .
```

# データ形式

```
    音声データセットdir
     |-spk1
     |  |-wav
           |-audio_file_00001.wav
           |-audio_file_00002.wav
```

# 実行方法

話者リストのファイルを作成する

```python
python ./src/dataset/create_spk_list.py -i /data/karanovc -o ./results/karanovc_spk_list.txt
```

作成したファイルをコピーして、訓練用話者と評価用話者に分ける

訓練の実行 (適宜ファイルの中の変数を変更してください)

```s
./script/train/vc/00001.sh 
```

以下のディレクトリやファイルが特に変更する部分です。

```s
dataset.data_dir=/data/karanovc \ # データセットのディレクトリ
dataset.dataset_metadata_train_file=/nvc_net/results/train_karanovc_spk_list.txt \ # 訓練用の話者リスト
dataset.dataset_metadata_val_file=/nvc_net/results/val_karanovc_spk_list.txt \ # 評価用の話者リスト
dataset.dataset_metadata_test_file=/nvc_net/results/val_karanovc_spk_list.txt \ # 評価用用の話者リスト
dataset.speaker_list_file=/nvc_net/results/karanovc_spk_list.txt \ # # 話者リスト
```

# gpu関連

もし、`docker-compose-gpu.yml`における以下の設定で上手くいかない場合

```
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            capabilities: [gpu]
```

以下に変更する。

```
runtime: nvidia
```


# vscode extensionの設定

1. view/command palletを開き、shellからcodeをインストール
2. 新しいshellを開く
3. 以下のコマンド実行 (権限は与えておく)

```
./.devcontainer/vscode_extentions_install_batch.sh
```

# 訓練の実行


# Tensorboardの起動

```
tensorboard --logdir=./log/outputs/ --port 18053 --host 0.0.0.0
```