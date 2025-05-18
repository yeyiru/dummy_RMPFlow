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
import omni.usd
from pxr import Usd, Gf, UsdGeom, UsdPhysics, Sdf, UsdShade

import asyncio
from omni.isaac.core.utils.stage import open_stage_async
from omni.isaac.core.utils.rotations import euler_angles_to_quat

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from sensor_msgs.msg import JointState, Joy

from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from scipy.spatial.transform import Rotation as R

rclpy.init()
current_joint_array = None
target_position = np.array([0.0, -0.3, 0.26])
target_pitch = 0.0
target_roll = 0.0
target_yaw = np.pi


class TrajectoryPublisher(Node):
    def __init__(self, joint_names):
        super().__init__('isaac_joint_trajectory_publisher')
        self.publisher = self.create_publisher(JointTrajectory, 
                                               '/dummy_arm/controller/joints_command', 
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
        global target_position, target_pitch, target_roll, target_yaw
        # 1. 原始搖桿輸入（相對於 cube 自己的前/右）
        local_dx = msg.axes[0] * 0.01 * self.k * (-1)
        local_dy = msg.axes[1] * 0.01 * self.k
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
        
        if msg.buttons[4]:  # LB
            target_pitch += 0.01
        if msg.axes[2] < 0.5:  # LT 扳機扣下時值從 1 → -1
            target_pitch -= 0.01

        if msg.buttons[5]:  # RB
            target_roll += 0.01
        if msg.axes[5] < 0.5:  # RT
            target_roll -= 0.01

        self.get_logger().info(f"[JOY] Pos: {np.round(target_position, 2)}, \
                                       Pitch: {np.degrees(target_pitch)}°,\
                                       Roll: {np.degrees(target_roll)}°, \
                                       Yaw: {np.degrees(target_yaw):.1f}°")

def define_target_arrow(parents_prim_path: str, axis: str, color: tuple):
    #定义箭头
    define_prim(f"{parents_prim_path}/target_arrow_{axis}", "Xform")
    define_prim(f"{parents_prim_path}/target_arrow_{axis}/body", "Cylinder")
    define_prim(f"{parents_prim_path}/target_arrow_{axis}/tip", "Cone")

    target_arrow = XFormPrim(
        prim_path=f"{parents_prim_path}/target_arrow_{axis}"
    )
    target_arrow_body = XFormPrim(
        prim_path=f"{parents_prim_path}/target_arrow_{axis}/body"
    )
    target_arrow_tip = XFormPrim(
        prim_path=f"{parents_prim_path}/target_arrow_{axis}/tip"
    )

    target_arrow_body.set_local_scale(np.array([0.01, 0.01, 0.1]))  # 細長箭頭
    target_arrow_tip.set_local_scale(np.array([0.03, 0.03, 0.03]))  # 箭頭大小

    set_display_color(f"/World/target/target_arrow_{axis}/body", color)
    set_display_color(f"/World/target/target_arrow_{axis}/tip", color)
    return target_arrow, target_arrow_body, target_arrow_tip

def set_display_color(prim_path: str, rgb: tuple):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[Warning] Invalid prim: {prim_path}")
        return
    geom = UsdGeom.Gprim(prim)
    color = Gf.Vec3f(*rgb)
    # display_color_attr = geom.CreateDisplayColorAttr()
    geom.CreateDisplayColorAttr().Set([color])


def update_arrow_pose(arrow_tip, arrow_body, position, pitch, roll, yaw, tip_offset=0.15):
    roll = roll + np.pi / 2
    yaw = yaw + np.pi / 2
    quat = euler_angles_to_quat(np.array([pitch, roll, yaw]))  # 90° 繞 Z 軸
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0]) * 0.06
    arrow_pos = position + forward
    # cube.set_world_pose(position=position, orientation=quat)
    arrow_body.set_world_pose(position=arrow_pos, orientation=quat)
    
    # 計算箭頭在朝向 yaw 方向的 offset 位置
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    tip_pos = position + tip_offset * forward
    arrow_tip.set_world_pose(position=tip_pos, orientation=quat)

def set_arrow_pose(arrow_tip, arrow_body, position, axis, tip_offset=0.15):
    # roll = roll + np.pi / 2
    # yaw = yaw + np.pi / 2
    if axis == 'x':
        yaw = -np.pi / 2
        quat = euler_angles_to_quat(np.array([0.0, np.pi / 2, yaw]))  # 90° 繞 Z 軸
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        
    elif axis == 'y':
        pitch = np.pi / 2
        quat = euler_angles_to_quat(np.array([np.pi / 2, 0.0, pitch]))
        forward = np.array([np.sin(pitch), 0.0, 0.0])

    elif axis == 'z':
        roll = np.pi
        quat = euler_angles_to_quat(np.array([0.0, 0.0, roll]))
        forward = np.array([0.0, 0.0, 1.0])

    # arrow_pos = position + forward  * 0.06
    # # cube.set_world_pose(position=position, orientation=quat)
    # arrow_body.set_world_pose(position=arrow_pos, orientation=quat)
    
    # # 計算箭頭在朝向 yaw 方向的 offset 位置
    # forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    # tip_pos = position + tip_offset * forward
    # arrow_tip.set_world_pose(position=tip_pos, orientation=quat)
    arrow_body_pos = position + 0.06 * forward
    arrow_tip_pos = position + tip_offset * forward

    arrow_body.set_world_pose(position=arrow_body_pos, orientation=quat)
    arrow_tip.set_world_pose(position=arrow_tip_pos, orientation=quat)

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
last_ros_publish_time = time.time()
ros_publish_interval = 0.1

# 定義目標物體父类
define_prim("/World/target", "Xform")
target = XFormPrim(
    prim_path="/World/target",
    position=target_position,
)

# 定义目标方块
target_cube = cuboid.VisualCuboid("/World/target/cube", 
                                  # position=target_position, 
                                  size=0.05, 
                                  color=np.array([1.0, 0, 0]))
#定义箭头
_, target_arrow_x_body, target_arrow_x_tip = define_target_arrow("/World/target", "x", (1.0, 0.0, 0.0))  # 红色x箭头
_, target_arrow_y_body, target_arrow_y_tip = define_target_arrow("/World/target", "y", (0.0, 1.0, 0.0))  # 绿色y箭头
_, target_arrow_z_body, target_arrow_z_tip = define_target_arrow("/World/target", "z", (0.0, 0.0, 1.0))  # 蓝色z箭头


world.play()

# 先获取Dummy Arm的初始位置
initial = True

# 設定旋轉（歐拉角 → 四元數），單位為 **弧度**
quat = euler_angles_to_quat(np.array([target_pitch, target_roll, target_yaw]))  # 90° 繞 Z 軸

# 設定世界姿態（位置 + 旋轉）
target.set_world_pose(
    position = target_position,
    orientation = quat
)
set_arrow_pose(target_arrow_x_tip, target_arrow_x_body, target_position, 'x')
set_arrow_pose(target_arrow_y_tip, target_arrow_y_body, target_position, 'y')
set_arrow_pose(target_arrow_z_tip, target_arrow_z_body, target_position, 'z')


# 4. 每幀更新 → 目標追踪控制

while simulation_app.is_running():
    world.step(render=True)
    if target_position is not None:
        
        # 設定旋轉（歐拉角 → 四元數），單位為 **弧度**
        target_quat = euler_angles_to_quat(np.array([target_pitch, target_roll, target_yaw]))  # 90° 繞 Z 軸、
        # 設定世界姿態（位置 + 旋轉）
        # 目标当前旋转
        r1 = R.from_quat(target_quat)

        # 绕Z轴旋转90度（单位是度）
        r2 = R.from_euler('z', -90, degrees=True)
        r3 = R.from_euler('x', -90, degrees=True)
        r4 = R.from_euler('y', -90, degrees=True)
        r_new = r1 # r4 * r3 * r2 * r1
        
        pitch = r_new.as_euler('xyz', degrees=True)[0]  # 提取 yaw
        r_z = R.from_euler('x', pitch, degrees=True)


        target.set_world_pose(
            position = target_position + np.array([0.05, 0.0, 0.0]),
            orientation = r_z.as_quat() # target_quat
        )

        # # 設定旋轉（歐拉角 → 四元數），單位為 **弧度**
        # quat = euler_angles_to_quat(np.array([target_pitch, target_roll, target_yaw]))  # 90° 繞 Z 軸、
        # # 設定世界姿態（位置 + 旋轉）
        # target.set_world_pose(
        #     position = target_position,
        #     orientation = quat
        # )
        
    # update_arrow_pose(arrow_tip, arrow_body, target_position, target_pitch, target_roll, target_yaw)
    if initial:
        # 同步到Isaac
        while current_joint_array is None:
            rclpy.spin_once(ros_joint_state_sub, timeout_sec=0.01)

        action = ArticulationAction(joint_positions=np.array(current_joint_array))
        dummy_arm.apply_action(action)
        # 等待機械臂達到指令位置
        max_iters = 200
        threshold =  0.01  # 米，末端位置誤差容忍度
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
        initial = False
    
    rmpflow.update_world()
    rmpflow.set_end_effector_target(
        target_cube.get_world_pose()[0],
        target_cube.get_world_pose()[1]
    )

    action = articulation_rmpflow.get_next_articulation_action()
    dummy_arm.apply_action(action)
    now = time.time()
    # if now - last_ros_publish_time >= ros_publish_interval:
    #     ros_joint_cmd_pub.publish_action(action.joint_positions)
    #     last_ros_publish_time = now
    # executor.spin_once(timeout_sec=0.01)
    ros_joint_cmd_pub.publish_action(action.joint_positions)
    rclpy.spin_once(ros_joint_cmd_pub, timeout_sec=0.01)
    rclpy.spin_once(ros_joy_sub, timeout_sec=0.01)
# 5. 關閉 simulation
simulation_app.close()