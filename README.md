# TFLite_ObjectDetector
TensorflowのObject Detection APIを用いた物体検出  
Raspberry Pi + Google Coral USB Acceleratorでの使用を目的とした軽量な物体検出モデルを作成します．
</br>  
<div align="center">
<img src=https://github.com/Dansato1203/TFLite_ObjectDetector/assets/53420676/fcb9dc9f-388f-4e41-a249-ec62d3b19935, width=400px>
</div>
</br>
  
## 動作環境
以下の環境で動作を確認しています．  
- OS: Ubuntu 18.04.5 LTS x86_64
- CUDA: 11.0
- cudnn: 8.0.5
- Docker: 20.10.6

## 学習
以下の方法で学習を開始できます．  
1. アノテーション  
画像に対してアノテーションを行います．  
このリポジトリを用いて学習を行う際にはtfrecord形式のファイルが必要になります．  
[microsoft/VoTT](https://github.com/microsoft/VoTT)を使うと比較的かんたんにアノテーションを行いtfrecord形式のファイルを生成できます．  
<div align="center">
<img src=https://user-images.githubusercontent.com/53420676/168482659-011a6fd3-ccd4-4691-a753-e6e4e18d8378.png width=400px>
</div>
</br>
  
2. 教師データのダウンロード元の変更  
先で作成したtfrecordファイルをまとめてGoogle Drive等にアップロードし、共有リンクを取得してください．（リンクを知っている全員が見れる設定で）  
以下のID部分(`1k6Nc2xiwB9d2ZRD4LLCS8ndCmPHWqBko`)を共有リンク内のファイルIDに変更してください．
https://github.com/Dansato1203/TFLite_ObjectDetector/blob/3d4e18a0c22ce4a7db328c2607635c2e67645082/Dockerfile#L39  
</br>


3. dokcer image の作成  
以下のコマンドでdocker image を作成してください．  
```bash
$ docker build -t tf_detection:trainer .
$ docker build -t tf_detection:converter -f Dockerfile.convert .
```
4. 学習モデルの作成  
以下のコマンドでモデルの学習からCoral Edge TPUで使うことのできるモデルへの変換までを行います．  
```bash
$ ./launch_train.sh
```

### 補足
- 作成したモデルを[Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/#download)によって変換することでRaspberry Pi4 model B　+ Google Coral USB Acceleratorによって25fps程度で検出できることを確認しています．
- このディレクトリ内にできるtrain_logs内に訓練でできたモデル等が保存されています．  
  学習を再度行う際にはtrain_logsを削除（あるいは中身だけ削除）してください．
