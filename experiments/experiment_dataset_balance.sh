for i in `seq 30`
do
  roslaunch nav_cloning nav_cloning_pytorch.launch script:=nav_cloning_node_add_limit.py mode:=change_dataset_balance
  sleep 20
done
