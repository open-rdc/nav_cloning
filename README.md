# nav_cloning
## Running simulation

### 一括して起動
* nav_cloning (一定経路の模倣学習)
```
roscd nav_cloning/experiments/
./experiment_dataset_balance.sh
```
経路からの距離によってデータセットに加える数を変える手法  
シェルファイルのパラメータを変更することで様々な条件で実験可能  
`data/analysis/change_dataset_balance`に解析用のcsvファイルが生成される．

### Data Analysis
２号館３階のシミュレーターで試す場合は  
```
roscd nav_cloning/data/analysis/use_dl_output/
```  
にpath.csvとtraceable_pos.csvがある．（ロボットを配置する場所の計算に用いる）  
新しい環境で試したい場合は以下からpath.csvとtraceable_pos.csvを生成する．  
https://github.com/open-rdc/nav_cloning/wiki

生成された経路追従行動の解析をロボットを動かしながらやる場合(pytorch)
```
roslaunch nav_cloning nav_cloning_2-3.launch scripts:=analysis_with_moving_pytorch.py
```
生成された経路追従行動の解析をロボットを動かしながらやる場合(chainer)
```
roslaunch nav_cloning nav_cloning_2-3.launch scripts:=analysis_with_moving.py
```


