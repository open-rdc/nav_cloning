for i in `seq 10`
do
  roslaunch nav_cloning nav_cloning_sim.launch mode:=use_dl_output
  sleep 10
done
