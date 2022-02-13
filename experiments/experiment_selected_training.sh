for i in `seq 10`
do
  roslaunch nav_cloning nav_cloning_sim.launch mode:=selected_training
  sleep 10
done
