# nav_cloning
フレームワークをpytorchに移行(開発中)


## Running simulation

### 一括して起動
* nav_cloning (一定経路の模倣学習)
```
roscd nav_cloning/experiments/
./experiment_use_dl_output.sh
```
`nav_cloning/data`フォルダにログと学習済みモデルを保存  
シェルファイルのパラメータを変更することで様々な条件で実験可能

* nav_cloning_with_direction (経路選択を含む模倣学習)
```
roscd nav_cloning/experiments/
./experiment_with_direction_use_dl_output.sh
```
`nav_cloning/data`フォルダにログと学習済みモデルが保存  
シェルファイルのパラメータを変更することで様々な条件で実験可能

[![IMAGE](http://img.youtube.com/vi/6LG06ZbCjto/0.jpg)](https://youtu.be/6LG06ZbCjto)

### 分割して起動
* シミュレータの起動
```
roslaunch nav_cloning nav_cloning_sim.launch
```
* rviz上の2D Pose Estimateで自己位置を合わせる
* 実行
```
rosservice call /start_wp_nav
```
* save data:  /nav_cloning/data/result \
loss \
angle_error : navigationの出力と訓練されたモデルの出力の差 \
distance : 目標経路とロボットの位置の間の距離

## install
* 環境 
  * ubuntu20.04
  * ros noetic
  * Python 3系

* ワークスペースの用意
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
cd ../
catkin_build
```
* nav_cloningの用意
```
cd ~/catkin_ws/src
wget https://raw.githubusercontent.com/open-rdc/nav_cloning/pytorch/nav_cloning.install
wstool init
wstool merge nav_cloning.install
wstool up
```
* 依存パッケージのインストール
```
cd ~/catkin_ws/src
rosdep init
rosdep install --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y
cd ../
catkin_make
```
* その他インストール
GPUを使用するかでインストールするものが変わります．
GPU関連の設定は細心の注意をはらっておこなってください．

<共通>
```
pip３ install scikit-image　　
pip3 install tensorboard
```
<CPU のみ>
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
<GPU使用>
使用しているデバイスを確認し，セットアップします
* nvidia driver
* CUDA 
* cuDNN 

その後インストールしたCUDAのバージョンに対応したPytorchのバージョンを下記からダウンロードします
```
https://pytorch.org/get-started/locally/
```
## Docker
作成次第追加

