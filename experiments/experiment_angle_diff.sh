for i in `seq 10`
do
  roslaunch nav_cloning nav_cloning_sim.launch script:=nav_cloning_ange_diff.py
  sleep 10
done
