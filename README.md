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
pip install nnabla-ext-cuda110
```

## データセットのダウンロード

[VCTKデータセット](http://www.udialogue.org/ja/download-ja/cstr-vctk-corpus.html)

```
mkdir ./data
wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz -O ./data/VCTK-Corpus.tar.gz
cd ./data
tar 
```

# 実行環境作成(エディターモード)

`src.util.~`などモジュールのimportを行うために必要です。

```
pip install -e .
```

# test

テストの実行方法です。`pytest`か`tox`の使用方法を記載しています。

```
python -m pytest
```

toxで使用されるモジュールは、まず環境の作成を行います。

```
python -m tox
```


もし、toxの環境を作り直す

```
python -m tox -r
```

テストの実行方法です。
```
python -m tox -e py39
```

リンターによるチェックです。
```
python -m tox -e lint
```

# Docker

 CUDAによりpytorchのインストール方法が異なるので、適宜[公式](https://pytorch.org/)を参照し、インストールしてください。

```
docker-compose -f docker-compose-cpu.yml up -d
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

# ローカル環境で仮想環境の作成

```
python -m venv .venv
```

```
source .venv/bin/activate
```

```
deactivate
```

versoinを変更したい場合、最初にpythonのバージョンを変更する。
```
pyenv local 3.8.0
```

# vscode extensionの設定

1. view/command palletを開き、shellからcodeをインストール
2. 新しいshellを開く
3. 以下のコマンド実行 (権限は与えておく)

```
./.devcontainer/vscode_extentions_install_batch.sh
```

