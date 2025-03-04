# nav_cloning
フレームワークをpytorchに移行(開発中)


## Running simulation

### シェルスクリプトによる起動
#### nav_cloning (一定経路の模倣学習)
* mode: change_dataset_balance (default)
```
roscd nav_cloning/experiments/
./experiment_change_dataset_balance.sh
```
* mode: use_dl_output
```
roscd nav_cloning/experiments/
./experiment_use_dl_output.sh
```
`nav_cloning/data/result_{select mode}/{起動した時間}`: ログファイルの保存先 
`nav_cloning/data/model_{select mode}/{学習が終了した時間}`: 学習済みモデルの保存先  
シェルファイルのパラメータを変更することで様々な条件で実験可能

#### nav_cloning_with_direction (経路選択を含む模倣学習)
```
roscd nav_cloning/experiments/
./experiment_with_direction_use_dl_output.sh
```
`nav_cloning/data`フォルダにログと学習済みモデルが保存  
シェルファイルのパラメータを変更することで様々な条件で実験可能

[![IMAGE](http://img.youtube.com/vi/6LG06ZbCjto/0.jpg)](https://youtu.be/6LG06ZbCjto)

### launchファイルによる起動
#### nav_cloning (一定経路の模倣学習)
以下のいずれかの方法で起動可能 (`script=nav_cloning_node_pytorch.py`, `mode=change_dataset_balance`の場合)
```
roslaunch nav_cloning nav_cloning_sim.launch
```
```
roslaunch nav_cloning nav_cloning_sim.launch mode:=change_dataset_balance
```
```
roslaunch nav_cloning nav_cloning_sim.launch script:=nav_cloning_node_pytorch.py mode:=change_dataset_balance
```
* 出力される言葉の定義
loss \
angle_error : navigationの出力と訓練されたモデルの出力の差 \
distance : 目標経路とロボットの位置の間の距離

## install
* Environment
  * ubuntu20.04
  * [ros noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
  * Python3

* Install python3-catkin-tools
  ```
  sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
  wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
  sudo apt update
  sudo apt install python3-catkin-tools
  ```

* Install nav_cloning
  ```
  mkdir -p ~/catkin_ws/src
  cd ~/catkin_ws/src
  git clone https://github.com/open-rdc/nav_cloning
  wstool init
  wstool merge nav_cloning/nav_cloning.install
  wstool up
  rosdep install --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y
  cd ~/catkin_ws
  catkin build

  sudo apt install python3-pip
  pip3 install torch torchvision scikit-image tensorboard
  pip3 install --upgrade numpy scikit-image
  echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
  source ~/.bashrc
  ```

* for CPU
  ```
  pip3 install torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
  ```
### mode紹介 (新しい順)
以下の起動例では, すべて`script=nav_cloning_node_pytorch.py`で実行される

* change_dataset_balance (default)
use_dl_outputに比べ、経路から復帰する行動の割合を増やした手法
```
roslaunch nav_cloning nav_cloning_sim.launch mode:=change_dataset_balance
```
* selected_training
use_dl_outputに対して、学習器の出力と目標角速度の差を判断材料として加えた手法
```
roslaunch nav_cloning nav_cloning_sim.launch mode:=selected_training
```
* use_dl_output
zigzagに対して、学習器の出力も用いる手法
```
roslaunch nav_cloning nav_cloning_sim.launch mode:=use_dl_output
```
* zigzag
manualに蛇行を加えた手法
```
roslaunch nav_cloning nav_cloning_sim.launch mode:=zigzag
```
* manual
```
目標経路に近づいたときに、学習器に目標角速度をゼロとして入力する手法
roslaunch nav_cloning nav_cloning_sim.launch mode:=manual
```
* follow_line
ナビゲーションから得られた目標経路に追従する手法
```
roslaunch nav_cloning nav_cloning_sim.launch mode:=follow_line
```
----- old version -----

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

