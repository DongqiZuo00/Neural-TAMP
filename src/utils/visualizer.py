import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
import numpy as np
from src.core.graph_schema import CANONICAL_RELATIONS, PHYSICAL_RELATIONS, Relation

class BEVVisualizer:
    def __init__(self, save_dir="Neural-TAMP/vis_output"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # 为不同房间分配不同颜色 (RGBA)
        self.room_colors = [
            (1.0, 0.8, 0.8, 0.3), # 红
            (0.8, 1.0, 0.8, 0.3), # 绿
            (0.8, 0.8, 1.0, 0.3), # 蓝
            (1.0, 1.0, 0.8, 0.3), # 黄
            (0.8, 1.0, 1.0, 0.3), # 青
            (1.0, 0.8, 1.0, 0.3), # 紫
        ]

    def render(self, scene_graph, filename="bev.png", show_relations="canonical"):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. 先画房间的地板 (多边形)
        room_nodes = [n for n in scene_graph.nodes.values() if "Room" in n.id]
        
        all_x, all_z = [], [] # 用于自动缩放视图

        for i, room in enumerate(room_nodes):
            geo = room.geometry
            if "polygon" in geo:
                # 绘制多边形
                pts = geo["polygon"]
                # 这里的 pts 是 [(x, z)...]
                poly = patches.Polygon(pts, closed=True, 
                                     facecolor=self.room_colors[i % len(self.room_colors)],
                                     edgecolor='black', linestyle='--')
                ax.add_patch(poly)
                
                # 收集坐标用于自动定界
                for p in pts:
                    all_x.append(p[0])
                    all_z.append(p[1])

                # 标注房间名字和面积
                cx, cz = room.pos[0], room.pos[2]
                area_text = f"{geo['area']:.1f}m²"
                ax.text(cx, cz, f"{room.label}\n{area_text}", ha='center', va='center', fontsize=9, fontweight='bold')

        # 2. 画连接线 (Edge)
        if show_relations == "canonical":
            relation_filter = CANONICAL_RELATIONS
        else:
            relation_filter = PHYSICAL_RELATIONS

        for edge in scene_graph.edges:
            if edge.relation not in relation_filter:
                continue
            if edge.relation == Relation.CONTAINS: continue # 不画房间包含关系，太乱了
            
            if edge.source_id in scene_graph.nodes and edge.target_id in scene_graph.nodes:
                n1 = scene_graph.nodes[edge.source_id]
                n2 = scene_graph.nodes[edge.target_id]
                ax.plot([n1.pos[0], n2.pos[0]], [n1.pos[2], n2.pos[2]], color='gray', linestyle=':', alpha=0.5)

        # 3. 画物体
        for node in scene_graph.nodes.values():
            if "Room" in node.id: continue # 房间已经画过了
            
            x, z = node.pos[0], node.pos[2]
            all_x.append(x); all_z.append(z)
            
            # 简单的样式区分
            color = 'blue'
            marker = 'o'
            size = 50
            if "Door" in node.label: color, marker = 'brown', 's'
            elif "Table" in node.label: color, size = 'orange', 100
            
            ax.scatter(x, z, c=color, marker=marker, s=size, edgecolors='k', alpha=0.8)
            # 简化标签，防止重叠
            short_label = node.label.split('|')[0]
            ax.annotate(short_label, (x, z), xytext=(3, 3), textcoords='offset points', fontsize=7)

        # 4. 画机器人
        if scene_graph.robot_pose:
            p = scene_graph.robot_pose
            rx, rz = p['position']['x'], p['position']['z']
            yaw = math.radians(p['rotation']['y'])
            dx, dy = 0.5*math.sin(yaw), 0.5*math.cos(yaw)
            ax.arrow(rx, rz, dx, dy, fc='red', ec='red', width=0.1, head_width=0.3)
            ax.text(rx, rz-0.3, "Robot", color='red', fontsize=8, ha='center')
            all_x.append(rx); all_z.append(rz)

        # 设置视图范围 (自动 padding)
        if all_x:
            margin = 1.0
            ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
            ax.set_ylim(min(all_z)-margin, max(all_z)+margin)

        ax.set_title(f"Semantic Map with Room Geometry ({len(room_nodes)} Rooms)")
        ax.set_xlabel("X (meters)"); ax.set_ylabel("Z (meters)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
        
        plt.savefig(os.path.join(self.save_dir, filename), dpi=100)
        plt.close(fig)
