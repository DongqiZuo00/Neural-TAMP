import sys
import os

# 获取当前脚本的目录: .../Neural-TAMP/src/perception
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录: .../Neural-TAMP
project_root = os.path.abspath(os.path.join(script_dir, "../.."))

# 将项目根目录加入路径，这样 Python 就能找到 'src' 包了
sys.path.append(project_root)
import numpy as np
from src.core.graph_schema import SceneGraph, Node

class SpatialLifter:
    def __init__(self, image_width=640, image_height=480, fov=90):
        self.width = image_width
        self.height = image_height
        self.fov = fov
        
        # 预计算相机内参矩阵 (Assuming perfect pinhole camera)
        # Focal Length = (Width / 2) / tan(FOV / 2)
        self.focal_length = (self.width / 2) / np.tan(np.deg2rad(self.fov / 2))
        self.cx = self.width / 2
        self.cy = self.height / 2
        
    def lift_to_3d(self, scene_graph: SceneGraph, depth_map: np.ndarray, robot_pose: dict) -> SceneGraph:
        """
        核心功能：把 SceneGraph 里所有 Node 的 2D Box 中心点投射到 3D 世界坐标。
        
        Args:
            scene_graph: VLM 解析出的图 (Pos 还是 0)
            depth_map: HxW 的 numpy 数组，单位是米
            robot_pose: {'position': {'x':..., 'y':..., 'z':...}, 'rotation': {'y':...}}
        """
        
        # 1. 获取机器人当前位姿矩阵 (World to Camera 的逆变换)
        # ProcTHOR 坐标系: Y is Up. 
        # 我们需要构建 T_robot_world
        rx, ry, rz = robot_pose['position']['x'], robot_pose['position']['y'], robot_pose['position']['z']
        yaw = np.deg2rad(robot_pose['rotation']['y'])
        
        # 简单的 2D 平面旋转矩阵 (只考虑 Yaw，假设机器人不侧翻)
        # World -> Robot Rotation
        # x_new = x * cos(yaw) + z * sin(yaw)
        # z_new = -x * sin(yaw) + z * cos(yaw)
        
        for node_id, node in scene_graph.nodes.items():
            if node.bbox is None: continue
            
            # 2. 计算 2D 框中心点 (u, v)
            # Box 格式: [ymin, xmin, ymax, xmax] (来自 VLM Prompt)
            # 注意: VLM 输出是 0-1000 的归一化坐标，需要还原到 image_size
            ymin, xmin, ymax, xmax = node.bbox
            
            u = ((xmin + xmax) / 2) / 1000 * self.width
            v = ((ymin + ymax) / 2) / 1000 * self.height
            
            # 3. 获取该点的深度值 Z_cam
            # 加上边界检查防止越界
            u_idx = int(np.clip(u, 0, self.width - 1))
            v_idx = int(np.clip(v, 0, self.height - 1))
            z_cam = depth_map[v_idx, u_idx]
            
            # 如果深度无效 (太远或太近)，跳过
            if z_cam > 5.0 or z_cam < 0.1:
                continue
                
            # 4. 反投影: 像素 -> 相机坐标系 (Camera Coordinate)
            # X_cam = (u - cx) * Z / fx
            # Y_cam = (v - cy) * Z / fy  (注意 ProcTHOR 的 Y 轴方向可能需要反转)
            
            x_cam = (u - self.cx) * z_cam / self.focal_length
            y_cam = -(v - self.cy) * z_cam / self.focal_length # Y-up usually needs invert from image space
            
            # 5. 坐标变换: 相机坐标系 -> 世界坐标系
            # 这一步比较 tricky，取决于 ProcTHOR 的具体定义。
            # 通常：Robot Forward is Z, Right is X, Up is Y.
            
            # 先旋转
            x_rot = x_cam * np.cos(yaw) + z_cam * np.sin(yaw)
            z_rot = -x_cam * np.sin(yaw) + z_cam * np.cos(yaw)
            
            # 再平移
            x_world = rx + x_rot
            y_world = ry + y_cam + 0.9 # 假设相机高度 0.9m (Robot Eye Level)
            z_world = rz + z_rot
            
            # 6. 更新 Node 坐标
            node.pos = (float(x_world), float(y_world), float(z_world))
            
        return scene_graph

# =======================
# 单元测试 (Dummy Data)
# =======================
if __name__ == "__main__":
    print("Testing Spatial Lifter...")
    
    # Mock Data
    lifter = SpatialLifter()
    sg = SceneGraph()
    # 假设有个苹果在图像正中心 (500, 500) -> 归一化中心
    node = Node("apple|1", "apple", (0,0,0), bbox=[450, 450, 550, 550]) 
    sg.add_node(node)
    
    # Mock Depth Map (全白，深度 2.0米)
    depth = np.ones((480, 640)) * 2.0
    
    # Mock Pose (原点，朝北)
    pose = {'position': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {'y': 0}}
    
    sg_new = lifter.lift_to_3d(sg, depth, pose)
    
    print(f"Original Pos: (0, 0, 0)")
    print(f"Lifted Pos:   {sg_new.nodes['apple|1'].pos}")
    # 预期: Z 应该是 2.0 左右, X, Y 接近 0