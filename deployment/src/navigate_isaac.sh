#!/bin/bash

# Start TMUX session
SESSION_NAME="GNM_Isaac_Sim_Simulation"
tmux new-session -d -s $SESSION_NAME

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

tmux selectp -t 4    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane


# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "roscore" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate nomad" Enter
tmux send-keys "cd ../../isaac_sim_ros/scripts" Enter
tmux send-keys "python ros_warehouse_turtlebot.py" Enter

# Run the teleop.py script in the third pane
tmux select-pane -t 2
tmux send-keys "cd ../../isaac_sim_ros/rviz" Enter
tmux send-keys "rviz -d warehouse_turtlebot.rviz" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 3
tmux send-keys "conda activate issac_lab" Enter
tmux send-keys "python pd_controller.py" Enter

# Run the navigate.py script with command line args in the fifth pane
tmux select-pane -t 4
tmux send-keys "conda activate issac_lab" Enter
tmux send-keys "sleep 2; python navigate.py --model nomad --dir warehouse_turtlebot" Enter

# Run the pd_controller.py script in the sixth pane
tmux select-pane -t 5
tmux send-keys "rosrun teleop_twist_keyboard teleop_twist_keyboard.py" Enter

# Attach to session
tmux attach-session -t $SESSION_NAME