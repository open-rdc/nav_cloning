for i in `seq 10`
do
  roslaunch nav_cloning nav_cloning_sim.launch mode:=change_dataset_balance
  sleep 10
done
