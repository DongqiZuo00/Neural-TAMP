import prior
import ai2thor.controller
from ai2thor.platform import CloudRendering
from PIL import Image
import random # [新增] 用于随机出生点

class ProcTHOREnv:
    def __init__(self, scene_name="ProcTHOR-Train-1"):
        self.dataset = prior.load_dataset("procthor-10k")
        self.current_scene = self.dataset["train"][0]

        print("[Env] Initializing Controller...")
        self.controller = ai2thor.controller.Controller(
            agentMode="default",
            visibilityDistance=5.0,
            gridSize=0.25,
            renderDepthImage=True,
            renderInstanceSegmentation=False,
            width=640,
            height=480,
            fieldOfView=90,
            platform=CloudRendering
        )

    def reset(self):
        # 1. 重置场景
        self.controller.reset(scene=self.current_scene)
        
        # 2. [关键修改] 获取当前房间所有可达位置 (避免生成在墙里)
        # GetReachablePositions 会返回地板上所有合法的坐标
        event = self.controller.step(action="GetReachablePositions")
        positions = event.metadata["actionReturn"]
        
        if positions and len(positions) > 0:
            # 随机选一个安全点
            start_pos = random.choice(positions)
            
            # 3. [关键修改] 传送机器人 (补全 missing arguments)
            event = self.controller.step(
                action="TeleportFull",
                x=start_pos['x'],
                y=start_pos['y'],
                z=start_pos['z'],
                rotation=dict(x=0, y=90, z=0),
                horizon=0,      # [修复] 必须指定视线水平
                standing=True   # [修复] 必须指定站立状态
            )
        else:
            # 如果获取失败(极少情况)，尝试一个保守的默认点
            print("[Env] Warning: No reachable positions found. Trying default.")
            event = self.controller.step(
                action="TeleportFull",
                x=0, y=0.9, z=0,
                rotation=dict(x=0, y=0, z=0),
                horizon=0,
                standing=True
            )
            
        return self._process_obs(event)

    def change_scene(self, index: int):
        print(f"[Env] Switching to ProcTHOR-10K Scene Index: {index}")
        if index < 0 or index >= len(self.dataset["train"]):
            print(f"❌ Index {index} out of bounds. Using 0.")
            index = 0
        self.current_scene = self.dataset["train"][index]
        return self.reset()

    def get_observation(self):
        return self._process_obs(self.controller.last_event)

    def _process_obs(self, event):
        rgb = Image.fromarray(event.frame)
        depth = event.depth_frame
        pose = event.metadata["agent"]
        return {'rgb': rgb, 'depth': depth, 'pose': pose}

    def stop(self):
        self.controller.stop()

    def save_ground_truth_bev(self, save_path):
        print(f"[Env] Capturing Ground Truth BEV to {save_path}...")
        min_x, max_x = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        
        if "rooms" in self.current_scene:
            for room in self.current_scene["rooms"]:
                for point in room["floorPolygon"]:
                    x, z = point["x"], point["z"]
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_z, max_z = min(min_z, z), max(max_z, z)
        else:
            print("[Env] Warning: No room data found.")
            return

        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        width = max_x - min_x
        depth = max_z - min_z
        max_dim = max(width, depth)
        
        cam_height = max_dim * 1.2 + 5.0

        event = self.controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=center_x, y=cam_height, z=center_z),
            rotation=dict(x=90, y=0, z=0),
            fieldOfView=90,
            skyboxColor="white"
        )

        if len(event.third_party_camera_frames) > 0:
            frame = event.third_party_camera_frames[0]
            Image.fromarray(frame).save(save_path)
            print(f"   -> BEV Captured successfully (Size: {max_dim:.1f}m)")
        else:
            print("   -> ❌ Failed to capture BEV.")