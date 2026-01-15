import numpy as np
from matplotlib.path import Path
from src.core.graph_schema import SceneGraph, Node, Edge

class OracleInterface:
    def __init__(self, env):
        self.env = env
        self.ignore_categories = {
            "Wall", "Floor", "Ceiling", "Room", "Structure", "Lighting", "Window", "DoorFrame"
        }

    def _calculate_polygon_area(self, points):
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def get_hierarchical_graph(self) -> SceneGraph:
        event = self.env.controller.last_event
        metadata = event.metadata["objects"]
        house = self.env.current_scene
        
        graph = SceneGraph()
        graph.robot_pose = event.metadata["agent"]
        
        # --- 1. 处理房间 ---
        room_polygons = [] 
        
        if "rooms" in house:
            for i, room in enumerate(house["rooms"]):
                room_id = f"Room|{i}"
                room_type = room.get("roomType", "GenericRoom")
                
                poly_pts = [(p['x'], p['z']) for p in room['floorPolygon']]
                area = self._calculate_polygon_area(poly_pts)
                xs = [p[0] for p in poly_pts]
                zs = [p[1] for p in poly_pts]
                bounds = (min(xs), min(zs), max(xs), max(zs))
                center_pos = (sum(xs)/len(xs), 0.0, sum(zs)/len(zs))

                graph.add_node(Node(
                    id=room_id, 
                    label=room_type, 
                    pos=center_pos, 
                    state="static",
                    geometry={"polygon": poly_pts, "area": area, "bounds": bounds},
                    room_id=None # 房间自己不属于任何房间
                ))
                
                path = Path(poly_pts)
                room_polygons.append((room_id, path))

        # --- 2. 处理物体 ---
        for obj in metadata:
            if obj["objectType"] in self.ignore_categories: continue
            
            pos = obj["position"]
            # 状态处理
            states = []
            if obj.get("isOpen"): states.append("open")
            if obj.get("isPickedUp"): states.append("held")
            state_str = ", ".join(states) if states else "default"

            # 判定房间归属
            ox, oz = pos["x"], pos["z"]
            assigned_room_id = None
            
            for r_id, r_path in room_polygons:
                # 判定点是否在多边形内
                if r_path.contains_point((ox, oz), radius=0.01):
                    assigned_room_id = r_id
                    break 

            # 创建节点，写入 room_id
            obj_node = Node(
                id=obj["objectId"],
                label=obj["objectType"],
                pos=(pos["x"], pos["y"], pos["z"]),
                bbox=obj["axisAlignedBoundingBox"],
                state=state_str,
                room_id=assigned_room_id # <--- 关键赋值
            )
            graph.add_node(obj_node)
            
            # 添加 Room -> contains -> Object 的 Edge
            if assigned_room_id:
                graph.add_edge(Edge(source_id=assigned_room_id, target_id=obj_node.id, relation="contains"))

        return graph