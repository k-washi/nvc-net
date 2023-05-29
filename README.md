# NVC-Net

[NVC-Net](https://github.com/sony/ai-research-code/tree/master/nvcnet ) のpytorch version

BCE lossをmseに変更 (GとDの乖離を減らす)
信号の一致度をとるためsnr lossを追加
spk embeddingの一貫性保証のため、cycle consistency embedding lossを追加
label smoothingを追加

# nvc-net インストール

## docker環境の作成

```
docker-compose up -d
```

## データセットのダウンロード

[VCTKデータセット](http://www.udialogue.org/ja/download-ja/cstr-vctk-corpus.html)

```
mkdir ./data
wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz -O ./data/VCTK-Corpus.tar.gz
cd ./data
tar -xvf VCTK-Corpus.tar.gz
```

# 実行環境作成(エディターモード)

```
poetry install
source .venv/bin/activate
pip install -e .
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


```
# 前処理
# make-testで0.1 evalに使用する (訓練用)
python ./src/dataset/vctk/preprocess.py -i ./data/VCTK-Corpus/wav48/ -o ./data/vctk/train -s ./src/dataset/vctk/spk/list_of_speakers.txt --make-test

# 評価用
python ./src/dataset/vctk/preprocess.py -i ./data/VCTK-Corpus/wav48/ -o ./data/vctk/val -s ./src/dataset/vctk/spk/list_of_subs.txt

# Unseen
python ./src/dataset/vctk/preprocess.py -i ./data/VCTK-Corpus/wav48/ -o ./data/vctk/unseen -s ./src/dataset/vctk/spk/list_of_unseen_speakers.txt 

# 訓練

 python nvcnet/main.py -c cudnn -d 0 --output_path .log/baseline/ --batch_size 2 --speaker_dir=./nvcnet/data/list_of_speakers.txt --save_data_dir=./data/vctk_train/ 
 ```

# Tensorboardの起動

```
tensorboard --logdir=./log/outputs/ --port 18053 --host 0.0.0.0
```