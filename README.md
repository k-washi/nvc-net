# ml-exp-env
機械学習実験環境

# nvc-net インストール

## docker環境の作成

```
docker-compose -f docker-compose-gpu.yml up -d
```

[numbaにおいてllvmliteでpython3.9のpipが対応していないため、3.8.10を使用する](https://github.com/numba/llvmlite/issues/621#issuecomment-727142311)

- [nnbala-cudaのインストール](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html)

```
pip install nnabla
pip install nnabla-ext-cuda110-nccl2-mpi3-1-6
python -c "import nnabla_ext.cuda, nnabla_ext.cudnn"
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

`src.util.~`などモジュールのimportを行うために必要です。

```
pip install -e .
```

# Docker

 CUDAによりpytorchのインストール方法が異なるので、適宜[公式](https://pytorch.org/)を参照し、インストールしてください。

```
docker-compose -f docker-compose-gpu.yml up -d
```

でコンテナを作成し、VS Codeの`ms-vscode-remote.remote-containers`から開発環境に入る


# gpu周り

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

python nvcnet/preprocess.py -i data/VCTK-Corpus/wav48/ -o ./data/vctk_out/ -s ./nvcnet/data/list_of_speakers.txt --make-test

# 訓練

 python nvcnet/main.py -c cudnn -d 0 --output_path .log/baseline/ --batch_size 2 --speaker_dir=./nvcnet/data/list_of_speakers.txt --save_data_dir=./data/vctk_train/ 
 ```