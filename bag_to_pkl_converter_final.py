import rosbag
import pickle
import rospy
import numpy as np
from tf.transformations import quaternion_matrix
import torch

# ===== 工具函数 =====
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

def quaternion_to_yaw(q):
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

def build_ego_pose_matrix(pos, orientation):
    R = quaternion_to_rotation_matrix(orientation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T.tolist()

# ===== 提取函数 =====
def extract_rosbag_to_pkl(bag_path, output_path):
    ego_pose_seq_dict = {}
    od_seq_dict = {}
    map_line_seq_dict = {}
    map_element_seq_dict = {}
    ego_state_seq_dict = {}
    ego_reference_line_seq_dict = {}
    map_lka_line_seq_dict = {}
    map_lka_old_seq_dict = {}

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[
            '/perception/map_lane_line_list',
            '/perception/obstacle_list',
            '/odom/current_pose',
            '/sensing/ego_car_state',
            '/perception/map_element_array',
            '/planning/trajectory_reference_line',
            '/perception/lka_lane_list',
            '/perception/lka_lane_list_old'
        ]):
            ts = str(msg.header.stamp)

            if topic == '/perception/map_lane_line_list':
                lines = []
                for line in msg.lines:
                    if line.type == 4: continue
                    lines.append({
                        'points': [[p.x, p.y] for p in line.points],
                        'type': line.type,
                        'color': line.color,
                        'confidence': line.confidence,
                        'position': line.position
                    })
                map_line_seq_dict[ts] = lines

            elif topic == '/perception/obstacle_list':
                agents = []
                for obj in msg.tracks:
                    agents.append({
                        'id': obj.id,
                        'x': obj.position.x, 'y': obj.position.y,
                        'vx': obj.velocity.x, 'vy': obj.velocity.y,
                        'accel': [obj.accel.x, obj.accel.y],
                        'yaw': obj.rotation.yaw,
                        'length': obj.bbox3d.size.x,
                        'width': obj.bbox3d.size.y,
                        'type': obj.type,
                        'type_confidence': obj.type_confidence
                    })
                od_seq_dict[ts] = agents

            elif topic == '/odom/current_pose':
                pos = msg.pose.position
                ori = msg.pose.orientation
                pose = build_ego_pose_matrix(
                    [pos.x, pos.y, pos.z],
                    [ori.x, ori.y, ori.z, ori.w]
                )
                ego_pose_seq_dict[ts] = {
                    'position': [pos.x, pos.y, pos.z],
                    'orientation': [ori.x, ori.y, ori.z, ori.w],
                    'ego_pose': pose
                }

            elif topic == '/sensing/ego_car_state':
                ego_state_seq_dict[ts] = {
                    'linear_velocity': msg.linear_velocity,
                    'linear_acceleration': msg.linear_acceleration,
                    'steering_wheel_angle': msg.steering_wheel_angle
                }

            elif topic == '/perception/map_element_array':
                elements = []
                for e in msg.map_element:
                    if e.type != 1: continue  # 只保留 STOP_LINE
                    elements.append({
                        'points': [[p.x, p.y] for p in e.points],
                        'type': e.type,
                        'confidence': e.confidence
                    })
                map_element_seq_dict[ts] = elements

            elif topic == '/planning/trajectory_reference_line':
                ego_reference_line_seq_dict[ts] = [[wp.x, wp.y] for wp in msg.waypoints]

            elif topic == '/perception/lka_lane_list':
                lanes = []
                for lane in msg.lanes:
                    if lane.position_parameter_c0 > 0 or lane.heading_angle_parameter_c1 > 0 or lane.curvature_parameter_c2 > 0:
                        x = np.linspace(lane.view_range_start, lane.view_range_end, 20)
                        y = lane.position_parameter_c0 + lane.heading_angle_parameter_c1 * x + lane.curvature_parameter_c2 * x**2
                        lanes.append([list(p) for p in zip(x, y)])
                map_lka_line_seq_dict[ts] = lanes

            elif topic == '/perception/lka_lane_list_old':
                lanes = []
                for lane in msg.lanes:
                    if lane.position_parameter_c0 > 0 or lane.heading_angle_parameter_c1 > 0 or lane.curvature_parameter_c2 > 0:
                        x = np.linspace(lane.view_range_start, lane.view_range_end, 20)
                        y = lane.position_parameter_c0 + lane.heading_angle_parameter_c1 * x + lane.curvature_parameter_c2 * x**2
                        lanes.append([list(p) for p in zip(x, y)])
                map_lka_old_seq_dict[ts] = lanes

    result = {
        'ego_pose': ego_pose_seq_dict,
        'obstacles': od_seq_dict,
        'map_lane_lines': map_line_seq_dict,
        'map_elements': map_element_seq_dict,
        'ego_state': ego_state_seq_dict,
        'reference_line': ego_reference_line_seq_dict,
        'lka_lane': map_lka_line_seq_dict,
        'lka_lane_old': map_lka_old_seq_dict
    }

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"[✔] 提取完成，保存至: {output_path}")

# ===== 用户只需修改这两个路径 =====
if __name__ == "__main__":
    bag_path = "/path/to/your.bag"  # ← 修改这里
    output_path = "/path/to/save.pkl"  # ← 修改这里
    extract_rosbag_to_pkl(bag_path, output_path)
