from isaacsim import SimulationApp

# Example ROS bridge sample showing manual control over messages
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False})
import carb
import omni
import omni.graph.core as og
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.extensions import enable_extension

# Global Variebles
control_frequency = 10 # Hz
simulation_step_size = 1.0 / control_frequency

# enable ROS bridge extension
enable_extension("omni.isaac.ros_bridge")

simulation_app.update()

# check if rosmaster node is running
# this is to prevent this sample from waiting indefinetly if roscore is not running
# can be removed in regular usage
import rosgraph

if not rosgraph.is_master_online():
    carb.log_error("Please run roscore before executing this script")
    simulation_app.close()
    exit()

usd_path =  "../assets/ROS_Warehouse_Turtlebot.usd"
omni.usd.get_context().open_stage(usd_path, None)

# Wait two frames so that stage starts loading
simulation_app.update()
simulation_app.update()

print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading

while is_stage_loading():
    simulation_app.update()
print("Loading Complete")

simulation_context = SimulationContext(stage_units_in_meters=1.0)

ros_cameras_graph_path = "/World/Turtlebot3_Camera/turtlebot3_burger/ROS_Cameras"

# Enabling rgb aimage publishers for left camera. Cameras will automatically publish images each frame
og.Controller.set(
    og.Controller.attribute(ros_cameras_graph_path + "/isaac_create_render_product.inputs:enabled"), True
)

simulation_context.play()
simulation_context.step()

# Simulate for one second to warm up sim and let everything settle
for frame in range(60):
    simulation_context.step()

# Dock the second camera window
left_viewport = omni.ui.Workspace.get_window("Viewport")
right_viewport = omni.ui.Workspace.get_window("Viewport 2")
if right_viewport is not None and left_viewport is not None:
    right_viewport.dock_in(left_viewport, omni.ui.DockPosition.RIGHT)
right_viewport = None
left_viewport = None

import rosgraph

if not rosgraph.is_master_online():
    carb.log_error("Please run roscore before executing this script")
    simulation_app.close()
    exit()

import rospy

frame = 0
while simulation_app.is_running():
    # Run with a fixed step size
    simulation_context.step(render=True)
    frame = frame + 1

rospy.signal_shutdown("warehouse turtlebot stage complete")
simulation_context.stop()
simulation_app.close()