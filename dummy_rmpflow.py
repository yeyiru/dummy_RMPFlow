from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import time
import math
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects import cuboid
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import define_prim
from pxr import Gf, UsdGeom, UsdPhysics

import asyncio
from omni.isaac.core.utils.stage import open_stage_async
from omni.isaac.core.utils.rotations import euler_angles_to_quat

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from sensor_msgs.msg import JointState, Joy

from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

rclpy.init()
current_joint_array = None
target_position = np.array([0.0, -0.25, 0.4])
target_yaw = 3.14

class TrajectoryPublisher(Node):
    def __init__(self, joint_names):
        super().__init__('isaac_joint_trajectory_publisher')
        self.publisher = self.create_publisher(JointTrajectory, 
                                               '/dummy/control/joint_trajectory_controller/command', 
                                               10)
        self.joint_names = joint_names

    def publish_action(self, joint_positions):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        joint_positions[0] = -joint_positions[0]
        joint_positions[2] = joint_positions[2] + np.pi / 2
        joint_positions[3] = -joint_positions[3]
        point.positions = np.degrees(joint_positions).tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100_000_000  # 0.1秒到達目標，可調

        msg.points = [point]

        self.publisher.publish(msg)
        print(f"[ROS2] Published JointTrajectory: {point.positions}")

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('dummy_arm_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            "/dummy_arm/current/joint_states",
            self.callback,
            10)
        # ros_name: ISAAC_name
        self.name_map = {'joint_1': 'Joint1',
                         'joint_2': 'Joint2',
                         'joint_3': 'Joint3',
                         'joint_4': 'Joint4',
                         'joint_5': 'Joint5',
                         'joint_6': 'Joint6'}
        self.dof_names = list(self.name_map.values())
        print("[ROS2] Subscribed to /dummy_arm/current/joint_states")

    def callback(self, msg: JointState):
        # 處理接收到的 JointState 消息
        isaac_names = []
        for ros_name in msg.name:
            if ros_name not in self.name_map:
                print(f"[ROS2] Unknown joint name: {ros_name}")
                return
            isaac_names.append(self.name_map[ros_name])
        joint_map = dict(zip(isaac_names, msg.position))
        
        joint_array = [math.radians(joint_map.get(name, 0.0)) for name in self.dof_names]
        joint_array[0] = -joint_array[0]
        joint_array[2] = joint_array[2] - np.pi / 2
        joint_array[3] = -joint_array[3]
        global current_joint_array
        current_joint_array = joint_array

class JoyTargetCubeController(Node):
    def __init__(self, k=0.5):
        super().__init__('joy_target_cube_controller')
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.k = k

    # def joy_callback(self, msg: Joy):
    #     global target_position, target_yaw
    #     # 左搖桿控制位置
    #     dx = msg.axes[0] * 0.01 * self.k
    #     dy = msg.axes[1] * 0.01 * self.k * (-1)

    #     # 右搖桿控制高度與角度
    #     dz = msg.axes[4] * 0.01 * self.k
    #     dyaw = msg.axes[3] * 0.05 * self.k  # 弧度

    #     target_position += np.array([dx, dy, dz])
    #     target_yaw += dyaw

    #     self.get_logger().info(f"[JOY] Pos: {np.round(target_position, 2)}, Yaw: {np.degrees(target_yaw):.1f}°")
    def joy_callback(self, msg: Joy):
        global target_position, target_yaw
        # 1. 原始搖桿輸入（相對於 cube 自己的前/右）
        local_dx = msg.axes[0] * 0.01 * self.k * (-1)   # 前進（Y搖桿）→ 轉成 X 方向分量
        local_dy = msg.axes[1] * 0.01 * self.k   # 右移（X搖桿）→ 轉成 Y 方向分量
        dz = msg.axes[4] * 0.01 * self.k
        dyaw = msg.axes[3] * 0.05 * self.k  # 弧度，右搖桿水平

        # 2. 根據 yaw 旋轉局部向量 → 世界座標下的平移
        cos_yaw = np.cos(target_yaw)
        sin_yaw = np.sin(target_yaw)
        dx = local_dx * cos_yaw - local_dy * sin_yaw
        dy = local_dx * sin_yaw + local_dy * cos_yaw

        # 3. 更新 cube 的位置與朝向
        target_position += np.array([dx, dy, dz])
        target_yaw += dyaw

        self.get_logger().info(f"[JOY] Pos: {np.round(target_position, 2)}, Yaw: {np.degrees(target_yaw):.1f}°")

# 1. 載入場景 + 建立世界
async def load():
    await open_stage_async("./standalone_examples/my_standalone/dummy_standalone/data/dummy_usd/dummy.usd")  # 改成你自己的路徑

asyncio.get_event_loop().run_until_complete(load())
simulation_app.update()
world = World()
world.reset()

# 2. 初始化 articulation（機械臂）
dummy_arm = Articulation("/dummy")  # 根據你的 USD 實際路徑
dummy_arm.initialize()

ros_joint_cmd_pub = TrajectoryPublisher(dummy_arm.dof_names)
ros_joint_state_sub = JointStateSubscriber()
ros_joy_sub = JoyTargetCubeController()
# executor = MultiThreadedExecutor()
# executor.add_node(ros_joint_cmd_pub)
# executor.add_node(ros_joint_state_sub)

# 3. 初始化 RMPFlow 和目標物體
rmpflow = RmpFlow(
    robot_description_path = "./standalone_examples/my_standalone/dummy_standalone/data/robot_description_file.yaml",
    urdf_path = "./standalone_examples/my_standalone/dummy_standalone/data/dummy_urdf/dummy.urdf",
    rmpflow_config_path = "./standalone_examples/my_standalone/dummy_standalone/data/rmpflow_config_file.yaml",
    end_effector_frame_name = "tool0",
    maximum_substep_size = 0.00334
)

physics_dt = 1 / 60
articulation_rmpflow = ArticulationMotionPolicy(dummy_arm, rmpflow, physics_dt)

target_cube = cuboid.VisualCuboid("/World/target", 
                                  position=target_position, 
                                  size=0.05, 
                                  color=np.array([1.0, 0, 0]))
define_prim("/World/target_arrow", "Xform")
define_prim("/World/target_arrow/mesh", "Cube")
arrow = XFormPrim(
    prim_path="/World/target_arrow")
arrow.set_local_scale(np.array([0.1, 0.01, 0.01]))  # 細長箭頭

def update_arrow_pose(arrow, position, yaw):
    yaw = yaw + np.pi / 2
    quat = euler_angles_to_quat(np.array([0.0, 0.0, yaw]))  # 90° 繞 Z 軸
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0]) * 0.06
    arrow_pos = position + forward
    # cube.set_world_pose(position=position, orientation=quat)
    arrow.set_world_pose(position=arrow_pos, orientation=quat)

# obstacle = cuboid.VisualCuboid("/World/obstacle", position=np.array([0.2, 0.2, 0.2]), size=0.05, color=np.array([0, 1.0, 0]))
# rmpflow.add_obstacle(obstacle)

world.play()

# 先获取Dummy Arm的初始位置
initial = True

# 設定旋轉（歐拉角 → 四元數），單位為 **弧度**
quat = euler_angles_to_quat(np.array([0.0, 0.0, target_yaw]))  # 90° 繞 Z 軸

# 設定世界姿態（位置 + 旋轉）
target_cube.set_world_pose(
    position = target_position,
    orientation = quat
)


# 4. 每幀更新 → 目標追踪控制
while simulation_app.is_running():
    world.step(render=True)
    if target_position is not None:
        # 設定旋轉（歐拉角 → 四元數），單位為 **弧度**
        quat = euler_angles_to_quat(np.array([0.0, 0.0, target_yaw]))  # 90° 繞 Z 軸、
        # 設定世界姿態（位置 + 旋轉）
        target_cube.set_world_pose(
            position = target_position,
            orientation = quat
        )
    update_arrow_pose(arrow, target_position, target_yaw)
    if initial:
        # 同步到Isaac
        while current_joint_array is None:
            rclpy.spin_once(ros_joint_state_sub, timeout_sec=0.01)

        action = ArticulationAction(joint_positions=np.array(current_joint_array))
        dummy_arm.apply_action(action)
        # 等待機械臂達到指令位置
        max_iters = 200
        threshold = 0.01  # 米，末端位置誤差容忍度
        reached = False

        for _ in range(max_iters):
            world.step(render=True)

            # 檢查末端位置是否已經達到 apply_action() 的目標
            target_joint_positions = np.array(current_joint_array)
            # 計算與目標關節差距
            current_joint_pos = dummy_arm.get_joint_positions()
            print("[Sync] Sync...ing to Isaac")
            if np.allclose(current_joint_pos, target_joint_positions, atol=threshold):
                reached = True
                break

        # if reached:
        #     tool0_prim = RigidPrim(prim_path="/dummy/link6_1_1/tool0")
        #     ee_pos, ee_quat = tool0_prim.get_world_pose()
        #     target_cube.set_world_pose(position=ee_pos, orientation=ee_quat)
        #     print(f"[Sync] Synced and moved target to EE pos: {np.round(ee_pos, 3)}")
        # else:
        #     print("[Sync] Timed out waiting for joints to reach target.")
        initial = False
    
    rmpflow.update_world()
    rmpflow.set_end_effector_target(
        target_cube.get_world_pose()[0],
        target_cube.get_world_pose()[1]
    )

    action = articulation_rmpflow.get_next_articulation_action()
    dummy_arm.apply_action(action)
    ros_joint_cmd_pub.publish_action(action.joint_positions)
    # executor.spin_once(timeout_sec=0.01)
    rclpy.spin_once(ros_joint_cmd_pub, timeout_sec=0.01)
    rclpy.spin_once(ros_joy_sub, timeout_sec=0.01)
# 5. 關閉 simulation
simulation_app.close()