# nav_cloning
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
* 環境 ubuntu18.04, ros melodic

* ワークスペースの用意
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
cd ../
catkin_make
```
* nav_cloningの用意
```
cd ~/catkin_ws/src
wget https://raw.githubusercontent.com/open-rdc/nav_cloning/master/nav_cloning.install
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
```
sudo apt install python-pip
pip install chainer==6.0
pip install scikit-image
```
## Docker
* Usage
example:
1. 起動
```
cd ~/catkin_ws/src/nav_cloning/docker
docker-compose up
```
or
```
docker pull -p 8080:80 masayaokada/nav_cloning:open-rdc
```
2. アクセス
Access to http://localhost:8080

### Data Analysis
https://github.com/open-rdc/nav_cloning/wiki

