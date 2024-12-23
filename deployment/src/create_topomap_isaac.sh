#!/bin/bash

# 1st Parameter: topo map name
topo_map_name="${1:-warehouse_turtlebot}"
# 2nd Parameter: bag file name (with .bag) in the topomaps/bags directory
bag_name="${2:-warehouse_turtlebot.bag}"
# 3rd Parameter: delay between messages in the rosbag play command
delay="${3:-1.5}"

# Create a new tmux session
session_name="GNM_Create_Topomap"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run roscore in the first pane
tmux select-pane -t 0
tmux send-keys "roscore" Enter

# Run the create_topoplan.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate issac_lab" Enter
tmux send-keys "python create_topomap.py --dt 1 --dir $topo_map_name" Enter

# Change the directory to ../topomaps/bags and run the rosbag play command in the third pane
tmux select-pane -t 2
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag play -r $delay $bag_name" Enter # feel free to change the playback rate to change the edge length in the graph

tmux selectp -t 2

# Attach to the tmux session
tmux -2 attach-session -t $session_name
