for i in `seq 50`
do
  roslaunch nav_cloning nav_cloning_2-3.launch scripts:=nav_cloning_node_pytorch.py mode:=change_dataset_balance
  sleep 10
done
