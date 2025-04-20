import math
import numpy as np
import carb
import asyncio
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from omni.isaac.kit import SimulationApp

# 启动 Isaac Sim 无 UI 模式（可改为 headless=False）
simulation_app = SimulationApp({"headless": False})

import omni.usd

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import open_stage_async
from omni.isaac.core.utils.types import ArticulationAction


# 加载 dummy.usd 场景
usd_path = "/home/polarbear/Downloads/isaac_sim/exts/load_dummy/data/dummy/dummy.usd"  # 修改为实际路径

async def load_scene():
    await open_stage_async(usd_path)
    print(f"[Scene] Loaded {usd_path}")

simulation_app.update()
asyncio.get_event_loop().run_until_complete(load_scene())

# 创建世界 & 初始化 articulation
world = World(stage_units_in_meters=1.0)
world.reset()

# 等待 Articulation 出现
simulation_app.update()
dummy_arm = Articulation("/dummy")  # 替换为 USD 中 articulation 的 prim 路径
dummy_arm.initialize()

print("[Scene] Dummy Arm initialized")

# 启动仿真
simulation_app.update()
world.play()
print("[Sim] ▶️ Simulation started")

# 创建 ROS2 节点
class JointStateSubscriber(Node):
    def __init__(self, articulation):
        super().__init__('dummy_arm_subscriber')
        self.articulation = articulation
        self.dof_names = articulation.dof_names
        self.subscription = self.create_subscription(
            JointState,
            "/dummy_arm/current/joint_states",
            self.callback,
            10
        )
        # ros_name: ISAAC_name
        self.name_map = {'joint_1': 'Joint1',
                         'joint_2': 'Joint2',
                         'joint_3': 'Joint3',
                         'joint_4': 'Joint4',
                         'joint_5': 'Joint5',
                         'joint_6': 'Joint6'}
        print("[ROS2] Subscribed to /dummy_arm/current/joint_states")

    def callback(self, msg: JointState):

        # joint_map = dict(zip(msg.name, msg.position))
        isaac_names = []
        for ros_name in msg.name:
            if ros_name not in self.name_map:
                print(f"[ROS2] Unknown joint name: {ros_name}")
                return
            isaac_names.append(self.name_map[ros_name])
        joint_map = dict(zip(isaac_names, msg.position))
        
        joint_array = [math.radians(joint_map.get(name, 0.0)) for name in self.dof_names]
        joint_array[2] = joint_array[2] - np.pi / 2
        action = ArticulationAction(joint_positions=np.array(joint_array))
        self.articulation.apply_action(action)
        print(f"[ROS2] Applied joints: {joint_array}")

# 初始化 ROS2
rclpy.init()
ros_node = JointStateSubscriber(dummy_arm)
executor = SingleThreadedExecutor()
executor.add_node(ros_node)

print("[Main] Setup complete, entering main loop")

# 主循环：更新 Isaac Sim + 执行 ROS2 回调
try:
    while simulation_app.is_running():
        world.step(render=True)
        executor.spin_once(timeout_sec=0.01)
except KeyboardInterrupt:
    pass

# 清理
ros_node.destroy_node()
rclpy.shutdown()
simulation_app.close()