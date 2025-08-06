import rosbag
import json
import numpy as np
import os
import tqdm
import argparse
import torch
import math
import copy

import nullmax_junction_utils
import test_nullmax_exits

import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev

'''
TODO:
1. rosbag中哪些type是静态障碍物?

'''


def generate_direct_centerline(lines):
    # import ipdb; ipdb.set_trace()
    sorted_lines = sorted(lines, key=lambda x: x[0][1])
    return None



def calculate_angle(v1, v2):
    """
    计算两个向量之间的夹角（以度为单位）
    """
    # 计算点积
    dot_product = np.dot(v1, v2)
    # 计算向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)
    # 防止浮点数精度问题导致 cos_theta 超出 [-1, 1] 范围
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # 计算夹角（弧度）并转换为角度
    angle = np.arccos(cos_theta) * 180 / np.pi
    return angle

def check_angle(points):
    """
    判断第一个线段和最后一个线段的走向角度是否大于 60 度
    :param points: 点的列表，格式为 [(x1, y1), (x2, y2), ...]
    :return: 如果夹角大于 60 度，返回 True；否则返回 False
    """
    if len(points) < 3:
        raise ValueError("至少需要 3 个点才能计算线段角度")

    # 第一个线段的向量
    v1 = np.array(points[1]) - np.array(points[0])
    # 最后一个线段的向量
    v2 = np.array(points[-1]) - np.array(points[-2])

    # 计算夹角
    angle = calculate_angle(v1, v2)

    # 判断夹角是否大于 60 度
    return angle > 60




def wrap_angle(angle, min_val = -math.pi, max_val = math.pi):
    return min_val + (angle + max_val) % (max_val - min_val)

## ====== agent feature + static feature
def ignore_this_agent(od_data, radius):

    # 超出限定范围的障碍物不做预测
    if math.sqrt(od_data['x'] ** 2 + od_data['y'] ** 2) > radius:
        return True

def is_static_obstacle(od_data):
    # 一些静态障碍物
    static_obstacle_list = ['cone', 'barrier','barrel']   # TODO: rosbag中哪些type是静态障碍物
    # 静态障碍物不检测
    if od_data['type'] in static_obstacle_list:
        return True



def find_closest_value_index(lst, target):
    closest_index = min(range(len(lst)), key=lambda i: abs(int(lst[i]) - int(target)))
    return closest_index

def stupid_list(wired_dict_keys):
    wired_set = set(wired_dict_keys)
    wired_list = []
    for name in wired_set:
        wired_list.append(name)
    return wired_list

def quaternion_to_rotation_matrix(q):
    """
    将四元数 (x, y, z, w) 转换为 3x3 旋转矩阵。
    """
    x, y, z, w = q
    R = torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])
    return R


def quaternion_to_yaw(q):
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return yaw


def build_ego_pose_matrix(pos, orientation):
    """
    构建 ego pose 矩阵 (4x4)。
    :param pos: 平移向量 (x, y, z)
    :param orientation: 四元数 (x, y, z, w)
    :return: 4x4 齐次变换矩阵
    """
    R = quaternion_to_rotation_matrix(orientation)
    T = np.eye(4)  # 初始化为单位矩阵
    T[:3, :3] = R  # 设置旋转部分
    T[:3, 3] = pos  # 设置平移部分
    return T.tolist()



def _extract_raw_data_from_rosbag(bag_file):
    '''
    从 rosbag 中提取 raw data(ego_data, agent_data, map_line_data, map_element_data)

    extract_NM_perception_data_path 的内部函数
    '''    

    topic_list = []
    map_line_seq_dict = {}
    map_element_seq_dict = {}
    od_seq_dict = {}
    ego_pose_seq_dict = {}
    ego_state_seq_dict = {}
    ego_reference_line_seq_dict = {}
    map_lka_line_seq_dict = {}
    map_lka_line_old_seq_dict = {}


    with rosbag.Bag(bag_file, 'r') as bag:
        # print("start extract topic")

        topics = bag.get_type_and_topic_info().topics
        # print("finish extracting topic")
        
        for key in topics.keys():
            topic_list.append(key)

        # for topic, msg, t in bag.read_messages(topics=topic_list):
        for topic, msg, t in bag.read_messages(topics=['/perception/map_lane_line_list', '/perception/obstacle_list', '/odom/current_pose', '/odom/current_pose', '/sensing/ego_car_state', '/perception/map_element_array', '/planning/trajectory_reference_line', '/perception/lka_lane_list', '/perception/lka_lane_list_old', '/planning/high_level_command']):
        
            # if topic == '/fusion/fused_lane_array':
            #     if msg.header.seq > 10:
            #         print(topic)
            #         import pdb; pdb.set_trace()

            # if topic == '/fusion/obstacle_list':
            #     print(topic)
            #     import pdb; pdb.set_trace()    

            # if topic == '/planning/high_level_command':
            #     import ipdb; ipdb.set_trace()
            
            if topic == '/perception/map_lane_line_list':
                # print(topic)
                '''
                enum LANE_TYPE {
                TYPE_NONE = 0;
                TYPE_SOLID = 1;
                TYPE_DASHED = 2;
                TYPE_DOT = 3;
                TYPE_ROAD_EDGE = 4;
                TYPE_WIDE_DASHED = 5;
                TYPE_DASHED_SLOW_DOWN = 6;
                TYPE_SOLID_SLOW_DOWN = 7;
                TYPE_DOUBLE_SOLID = 8;
                TYPE_SOLID_DASHED = 9;  // left solid, right dashed
                TYPE_DASHED_SOLID = 10; // left dashed, right solid
                TYPE_DOUBLE_DASHED = 11;
                TYPE_WIDE_SOLID = 12;
                TYPE_NUM = 13;
                }
                '''

                frame_map_list = []
                for line_item in msg.lines:
                    if line_item.type == 4:
                        continue
                    # import pdb; pdb.set_trace()
                    line_dict = {}
                    line_dict['points'] = [[point.x, point.y] for point in line_item.points]
                    line_dict['type'] = line_item.type
                    line_dict['color'] = line_item.color
                    line_dict['confidence'] = line_item.confidence
                    line_dict['position'] = line_item.position
                    frame_map_list.append(line_dict)
                # map_line_seq_dict[msg.header.seq] = frame_map_list
                map_line_seq_dict[str(msg.header.stamp)] = frame_map_list

            #  感知map信息提取
            if topic == '/perception/map_element_array':
                # for road_sign_item in msg.lines:
                frame_map_element_list = []
                for element in msg.map_element:
                    '''
                    enum MapElementType {
                    UNKNOWN_TYPE = 0;
                    STOP_LINE = 1;
                    ROAD_EDGE = 2;
                    ZEBRA_CROSSING = 3;
                    GROUND_DIRECTION_ARROW = 4;
                    }
                    '''
                    if element.type != 1:
                        continue
                    element_dict = {}
                    element_dict['points'] = [[point.x, point.y] for point in element.points]
                    element_dict['type'] = element.type
                    element_dict['confidence'] = element.confidence
                    frame_map_element_list.append(element_dict)
                map_element_seq_dict[str(msg.header.stamp)] = frame_map_element_list

            # od感知信息
            if topic == '/perception/obstacle_list':
                # print(topic)
                frame_agent_list = []
                for agent_item in msg.tracks:
                    agent_dict = {}
                    agent_dict['id'] = agent_item.id
                    agent_dict['x'] = agent_item.position.x 
                    agent_dict['y'] = agent_item.position.y
                    agent_dict['vx'] = agent_item.velocity.x
                    agent_dict['vy'] = agent_item.velocity.y
                    agent_dict['accel'] = [agent_item.accel.x, agent_item.accel.y]
                    agent_dict['yaw'] = agent_item.rotation.yaw
                    agent_dict['length'] = agent_item.bbox3d.size.x 
                    agent_dict['width'] = agent_item.bbox3d.size.y
                    agent_dict['type'] = agent_item.type
                    agent_dict['type_confidence'] = agent_item.type_confidence
                    
                    frame_agent_list.append(agent_dict)

                # od_seq_dict[msg.header.seq] = frame_agent_list
                od_seq_dict[str(msg.header.stamp)] = frame_agent_list

            # odom坐标系的自车信息
            if topic == '/odom/current_pose':
                frame_ego_pose = {}
                frame_ego_pose['position'] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                frame_ego_pose['orientation'] = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
                ego_pose = build_ego_pose_matrix(frame_ego_pose['position'], frame_ego_pose['orientation'])
                frame_ego_pose['ego_pose'] = ego_pose
                ego_pose_seq_dict[str(msg.header.stamp)] = frame_ego_pose
                # import pdb; pdb.set_trace()

            # 自车状态信息
            if topic == '/sensing/ego_car_state':
                # print(topic)
                frame_ego_state = {}
                frame_ego_state['linear_velocity'] = msg.linear_velocity
                frame_ego_state['linear_acceleration'] = msg.linear_acceleration
                frame_ego_state['steering_wheel_angle'] = msg.steering_wheel_angle

                ego_state_seq_dict[str(msg.header.stamp)] = frame_ego_state

            # 引导线
            if topic == '/planning/trajectory_reference_line':
                frame_reference_line = []
                for point in msg.waypoints:
                    frame_reference_line.append([point.x, point.y])

                ego_reference_line_seq_dict[str(msg.header.stamp)] = frame_reference_line


            # 量产车道线
            if topic == '/perception/lka_lane_list':
                frame_lka_line = []
                for line in msg.lanes:
                    if line.position_parameter_c0 > 0 or line.heading_angle_parameter_c1 > 0 or line.curvature_parameter_c2 > 0:
                        x = np.linspace(line.view_range_start, line.view_range_end, 20)
                        y = line.position_parameter_c0 + line.heading_angle_parameter_c1 * x + line.curvature_parameter_c2 * x * x
                        frame_lka_line.append([point for point in zip(x, y)])
                map_lka_line_seq_dict[str(msg.header.stamp)] = frame_lka_line

            if topic == '/perception/lka_lane_list_old':
                frame_lka_old_line = []
                for line in msg.lanes:
                    if line.position_parameter_c0 > 0 or line.heading_angle_parameter_c1 > 0 or line.curvature_parameter_c2 > 0:
                        x = np.linspace(line.view_range_start, line.view_range_end, 20)
                        y = line.position_parameter_c0 + line.heading_angle_parameter_c1 * x + line.curvature_parameter_c2 * x * x
                        frame_lka_old_line.append([point for point in zip(x, y)])
                map_lka_line_old_seq_dict[str(msg.header.stamp)] = frame_lka_old_line



    
    return ego_pose_seq_dict, od_seq_dict, map_line_seq_dict, map_element_seq_dict, ego_state_seq_dict, ego_reference_line_seq_dict, map_lka_line_seq_dict, map_lka_line_old_seq_dict


def _NM_percetion_data_sample(ego_data, od_data, map_line_data, map_element_data, ego_state_data, ego_reference_line_data, map_lka_line, map_lka_old_line, sample_frequency=2):
    '''
    ros bag 提取出的数据相邻时间不等, 约10-15Hz(时间戳戳相隔约 50ms);
    本函数负责从 raw data中采样出 ≈2Hz数据

    extract_NM_perception_data_path 的内部函数
    '''
    # origin_frequency = 20
    # sample_step_length = origin_frequency // sample_frequency
    # assert origin_frequency % sample_frequency == 0

    # timestamp_step_length = 1e9 // sample_frequency
    # assert 1e9 % sample_frequency == 0

    timestamp_step_length = int(5e8) 

    segment_sample_data_dict = {}
    segment_sample_timestramp_list = []

    ego_timestamp_list = stupid_list(ego_data.keys())
    map_timestamp_list = stupid_list(map_line_data.keys())
    map_element_timestamp_list = stupid_list(map_element_data.keys())
    od_timestamp_list = stupid_list(od_data.keys())
    ego_state_timestamp_list = stupid_list(ego_state_data.keys())
    ego_reference_line_timestamp_list = stupid_list(ego_reference_line_data.keys())
    map_lka_timestamp_list = stupid_list(map_lka_line.keys())
    map_lka_old_timestamp_list = stupid_list(map_lka_old_line.keys())


    od_timestamp_list.sort()

    od_timestamp = od_timestamp_list[0]
    while od_timestamp <= od_timestamp_list[-1]:
        # print('sampling...', od_timestamp, od_timestamp_list[-1])
        od_timestamp_index = find_closest_value_index(od_timestamp_list, od_timestamp)
        od_timestamp = od_timestamp_list[od_timestamp_index]

        assert od_timestamp in map_timestamp_list
        assert od_timestamp in map_element_timestamp_list 
        assert od_timestamp in map_lka_timestamp_list
        assert od_timestamp in map_lka_old_timestamp_list

        map_timestamp = od_timestamp

        ego_timestamp_index = find_closest_value_index(ego_timestamp_list, od_timestamp)
        ego_timestamp = ego_timestamp_list[ego_timestamp_index]

        ego_state_timestamp_index = find_closest_value_index(ego_state_timestamp_list, od_timestamp)
        ego_state_timestamp = ego_state_timestamp_list[ego_state_timestamp_index]

        # assert od_timestamp in ego_reference_line_timestamp_list
        ego_reference_line_timestamp_index = find_closest_value_index(ego_reference_line_timestamp_list, od_timestamp)
        ego_reference_line_timestamp = ego_reference_line_timestamp_list[ego_reference_line_timestamp_index]

        # import ipdb; ipdb.set_trace()
        # map_lka_line, map_lka_old_line

        frame_data = {}
        frame_data['ego_data'] = ego_data[ego_timestamp]
        frame_data['map_line_data'] = map_line_data[map_timestamp]
        frame_data['map_element_data'] = map_element_data[map_timestamp]
        frame_data['od_data'] = od_data[od_timestamp]
        frame_data['ego_state'] = ego_state_data[ego_state_timestamp]
        frame_data['ego_reference_line'] = ego_reference_line_data[ego_reference_line_timestamp]
        frame_data['map_lka_line'] = map_lka_line[od_timestamp]
        frame_data['map_lka_line_old'] = map_lka_old_line[od_timestamp]

        segment_sample_data_dict[od_timestamp] = frame_data
        segment_sample_timestramp_list.append(od_timestamp)
        # import ipdb; ipdb.set_trace()
        od_timestamp = str(int(od_timestamp) + timestamp_step_length)
        
    return segment_sample_data_dict, segment_sample_timestramp_list

def construct_reference_line(segment_sample_data_dict, all_segment_timestramp_list):

    # import ipdb; ipdb.set_trace()

    for current_timestamp in all_segment_timestramp_list:
        # ======== map feature
        map_line_data = []
        for line in segment_sample_data_dict[current_timestamp]['map_line_data']:
            
            if line['position'] != 7:
                map_line_data.append(line)
        # map_line_data = segment_sample_data_dict[current_timestamp]['map_line_data']  # map_line_data 目前放的是所有车道线，包含'points', 'type', 'color', 'confidence'
        # import ipdb; ipdb.set_trace()
        map_element_data = segment_sample_data_dict[current_timestamp]['map_element_data']

        all_line_type = ['solid-lane', 'dash-lane', 'edge', 'wide-dash-lane', 'wide-solid-lane', 'double-solid-lane', 'double-dash-lane', 'dash-solid-lane', 'variable-direction-lane', 'pending-transfer-area-dash-lane', 'drainage-lane', 'slow-down-dash-lane', 'unknown', 'road-edge'] # unknown后面的是 所有road-edge的type
        all_edge_type = ['road-edge', 'traffic-barrier', 'cone-curb', 'water-horse-curb', 'column-curb', 'crash-barrel-curb', 'unknown']
        all_color_type = ['white', 'yellow', 'orange', 'blue', 'green', 'red', 'unknown']
        line_list = []
        line_type_list = []
        line_confident_list = []
        line_id_list = []
        line_color_list = []
        global_line_list = []


        for single_line in map_line_data:
            if len(single_line['points']) > 1:
                line_list.append(single_line['points'])
                line_type_list.append(single_line['type'])
                line_confident_list.append(single_line['confidence'])

        tmp_putup_line_list = copy.deepcopy(line_list)
        tmp_putup_line_list_new, line_type_list = nullmax_junction_utils.put_up_lines(tmp_putup_line_list, line_type_list)

        tmp_putup_line_list = tmp_putup_line_list_new
        # tmp_putup_line_list = []
        # for line in tmp_putup_line_list_new:
        #     if check_angle(line):
        #         tmp_putup_line_list.append(line)


        # ============ 没有 stop line =================
        if len(map_element_data) == 0:
            
            centerline = generate_direct_centerline(segment_sample_data_dict[current_timestamp]['map_lka_line'] )

            print("no map element")
            for lane in segment_sample_data_dict[current_timestamp]['map_line_data']:
                if lane['position'] in [1, 2]:
                    color = 'green'
                    linestyle = '-'
                elif lane['position'] == [3,4]:
                    color = 'blue'
                    linestyle = '-'
                elif lane['position'] == [5,6]:
                    color = 'orange'
                    linestyle = '--'
                elif lane['position'] == 7:
                    color = 'pink'
                    linestyle = '.'
                elif lane['position'] == 8:
                    color = 'brown'
                    # linestyle = '--'
                elif lane['position'] in [9,10]:
                    color = 'black'
                    # linestyle = '--'
                else:
                    color = 'gray'

                x_coords = [point[0] for point in lane['points']]
                y_coords = [-point[1] for point in lane['points']]

                # diff = torch.tensor(lane['points'][0]) - torch.tensor(lane['points'][-1])
                # angle = torch.atan2(diff[0], diff[1])
                # print(angle * 180 / 3.1415926265)
                plt.plot(y_coords, x_coords, color=color)

            # x_coords = [point[0] for point in segment_sample_data_dict[current_timestamp]['ego_reference_line']]
            # y_coords = [-point[1] for point in segment_sample_data_dict[current_timestamp]['ego_reference_line']]
            # plt.plot(y_coords, x_coords, color='purple')


            # for line in segment_sample_data_dict[current_timestamp]['map_lka_line']:
            #     x_coords = [point[0] for point in line]
            #     y_coords = [point[1] for point in line]
            #     plt.plot(y_coords, x_coords, color='green')

            
            


            for line in segment_sample_data_dict[current_timestamp]['map_lka_line_old']:
                x_coords = [point[0] for point in line]
                y_coords = [point[1] for point in line]
                plt.plot(y_coords, x_coords, color='blue')
                # plt.savefig('./test1.png')
                # import ipdb; ipdb.set_trace()
    

            plt.plot(0, 0, 'ro')

            

            lines = []
            for line in segment_sample_data_dict[current_timestamp]['map_lka_line_old']:
                if line[0][0] == 0 and line[1][0] == 0:
                    pass
                else:
                    lines.append(line)
            
            sorted_lines = sorted(lines, key=lambda x: x[0][1])
            centerline_list = []

            # import ipdb; ipdb.set_trace()

            for index in range(len(sorted_lines)-1):
                left_line = np.array(sorted_lines[index])
                right_line = np.array(sorted_lines[index + 1])
                if (left_line[-1][0] - left_line[0][0]) > (right_line[-1][0] - right_line[0][0]):
                    # interp_func = interp1d(right_line[:,0], right_line[:,1])
                    long_lane_x = right_line[:,0]
                else:
                    long_lane_x = left_line[:,0]

                # centerline = (left_line + right_line)/2
                # centerline_list.append(centerline)
                # import ipdb; ipdb.set_trace()

                # 计算中心线点（逐点取平均）
                center_x = (left_line[:, 0] + right_line[:, 0]) / 2
                center_y = (left_line[:, 1] + right_line[:, 1]) / 2

                # 计算累积弧长（用于重新采样）
                distances = np.sqrt(np.diff(center_x)**2 + np.diff(center_y)**2)  # 计算每段长度
                arc_length = np.concatenate(([0], np.cumsum(distances)))  # 计算累积弧长

                # 归一化弧长，使其范围在 0-1 之间
                arc_length_normalized = arc_length / arc_length[-1]

                # 使用 B 样条插值，使中心线点数与长车道线一致
                tck, u = splprep([center_x, center_y], u=arc_length_normalized, s=0)  # B 样条拟合
                resampled_u = np.linspace(0, 1, len(long_lane_x))  # 目标长度
                smooth_center_x, smooth_center_y = splev(resampled_u, tck)  # 重新采样

                
                

                centerline_list.append([point for point in zip(smooth_center_x, smooth_center_y)])

            
            for line in centerline_list:
                x_coords = [point[0] for point in line]
                y_coords = [point[1] for point in line]
                plt.plot(y_coords, x_coords, color='yellow')

            print("no element")
            plt.savefig('./test1.png')
            plt.close()
            import ipdb; ipdb.set_trace()

        else:
            # import ipdb; ipdb.set_trace()
            tmp_stop_lines = [stop_line['points'] for stop_line in map_element_data]

            stop_lines = []
            # 只有距离大于3m的stop_line才进行保留
            for line in tmp_stop_lines:
                # stop_line = loc2global_egopose(line['Coords'], current_ego_pose)
                if nullmax_junction_utils.calculate_distance(line[0], line[-1]) > 3:
                    stop_lines.append(line)


            current_position = torch.tensor([0, 0])
            if not test_nullmax_exits.is_junction_scene(stop_lines, current_position):
                # return None # 非路口区域
                flag_is_junction = False
                print("not junction scene")
                # continue
            else:
                flag_is_junction = True
            
            debug_mode = False
            forward_stop_lines, back_stop_lines, left_stop_lines, right_stop_lines, all_stop_lines = test_nullmax_exits.classify_stoplines(
                stop_lines, current_position, debug_mode, None, None, None)
            

            # 延长筛选出的停止线，目前设置的最大延伸长度为 20m
            left_extend_stop_lines = nullmax_junction_utils.extend_stop_lines(left_stop_lines)
            right_extend_stop_lines = nullmax_junction_utils.extend_stop_lines(right_stop_lines)
            forward_extend_stop_lines = nullmax_junction_utils.extend_stop_lines(forward_stop_lines)
            back_extend_stop_lines = nullmax_junction_utils.extend_stop_lines(back_stop_lines)

            # lane_filter 应该就是 centerline， pair是车道线对
            go_left_center_lane_filters, left_filter_pair, go_right_center_lane_filters, right_filter_pair, \
            go_straight_center_lane_filters, forward_filter_pair, go_back_center_lane_filters, back_filter_pair = test_nullmax_exits.get_centerlines(
                left_stop_lines, right_stop_lines, forward_stop_lines, back_stop_lines, tmp_putup_line_list, line_type_list,
                left_extend_stop_lines, right_extend_stop_lines, forward_extend_stop_lines, back_extend_stop_lines)

            TURN_LEFT = 2
            TURN_RIGHT = 3
            TURN_LEFT_FRONT = 4
            TURN_RIGHT_FRONT = 5
            TURN_LEFT_BACK = 6
            TURN_RIGHT_BACK = 7
            TURN_LEFT_AND_AROUND = 8
            GO_STRAIGHT = 9
            MERGE_LEFT = 65
            MERGE_RIGHT = 66

            selected_lane_filters = None
            # direction = test_sdmap.generate_info(scenes_token, sample, map_ann_path)
            direction = TURN_RIGHT
            sd_map_direction_centerlines = []
            is_straight = False

            if direction == GO_STRAIGHT:
                print("sd map go straight: " + str(direction))
                selected_lane_filters = go_straight_center_lane_filters
                is_straight = True
            elif direction == TURN_LEFT or direction == TURN_LEFT_FRONT or direction == TURN_LEFT_BACK or direction == MERGE_LEFT:
                print("sd map turn left: " + str(direction))
                selected_lane_filters = go_left_center_lane_filters
            elif direction == TURN_RIGHT or direction == TURN_RIGHT_FRONT or direction == TURN_RIGHT_BACK or direction == MERGE_RIGHT:
                print("sd map turn right: " + str(direction))
                selected_lane_filters = go_right_center_lane_filters
            elif direction == TURN_LEFT_AND_AROUND:
                print("sd map turn around: " + str(direction))
                selected_lane_filters = go_back_center_lane_filters
            else:
                return None
                # print("default go straight: " + str(direction))
                # selected_lane_filters = go_straight_center_lane_filters
                # is_straight = True

            if selected_lane_filters is not None:
                for selected_lane_filter in selected_lane_filters:
                    for odom_line in selected_lane_filter:
                        sd_map_direction_centerlines.append(odom_line)
            
            # tmp
            future_trajectory = torch.tensor([[0,0], [0,20], [0,30], [0,40], [0,50], [0,60], [0, 70]])

            bezier_curves_first, start_point, start_and_end_points = test_nullmax_exits.generate_bezier_reference_lines(future_trajectory, 
                stop_lines, sd_map_direction_centerlines, is_straight)
                
            bezier_curves = test_nullmax_exits.get_same_length_curves(bezier_curves_first)



            flag_vis = True
            if flag_vis and flag_is_junction:
                # for agent in data['agent']['position']:
                #     x_coords = []
                #     y_coords = []
                #     for point in agent:
                #         if point[0] != 0 and point[1] != 0:
                #             x_coords.append(point[0])
                #             y_coords.append(-point[1])
                #     plt.plot(y_coords, x_coords, color='green')

                for lane in segment_sample_data_dict[current_timestamp]['map_line_data']:
                    if lane['position'] in [1, 2]:
                        color = 'green'
                        linestyle = '-'
                    elif lane['position'] == [3,4]:
                        color = 'blue'
                        linestyle = '-'
                    elif lane['position'] == [5,6]:
                        color = 'orange'
                        linestyle = '--'
                    elif lane['position'] == 7:
                        color = 'pink'
                        linestyle = '.'
                    elif lane['position'] == 8:
                        color = 'brown'
                        # linestyle = '--'
                    elif lane['position'] in [9,10]:
                        color = 'black'
                        # linestyle = '--'
                    else:
                        color = 'gray'

                    x_coords = [point[0] for point in lane['points']]
                    y_coords = [-point[1] for point in lane['points']]

                    diff = torch.tensor(lane['points'][0]) - torch.tensor(lane['points'][-1])
                    angle = torch.atan2(diff[0], diff[1]) * 180 / 3.1415926265

                    
                    # if 85 < angle < 95 or -7 < angle < 7 or 173 < angle < 180:
                    #     color = 'blue' 
                    # else:
                    #     color = 'gray'

                    # if lane['position'] == 7:
                    #     color = 'pink'
                    
                    plt.plot(y_coords, x_coords, color=color)

                    # import ipdb; ipdb.set_trace()

                # for line in tmp_putup_line_list:
                #     x_coords = [point[0] for point in line]
                #     y_coords = [-point[1] for point in line]
                #     plt.plot(y_coords, x_coords)

                x_coords = [point[0] for point in segment_sample_data_dict[current_timestamp]['ego_reference_line']]
                y_coords = [-point[1] for point in segment_sample_data_dict[current_timestamp]['ego_reference_line']]
                plt.plot(y_coords, x_coords, color='purple')

                # for line in segment_sample_data_dict[current_timestamp]['map_lka_line']:
                #     x_coords = [point[0] for point in line]
                #     y_coords = [point[1] for point in line]
                #     plt.plot(y_coords, x_coords, color='green')

                for line in segment_sample_data_dict[current_timestamp]['map_lka_line_old']:
                    x_coords = [point[0] for point in line]
                    y_coords = [point[1] for point in line]
                    plt.plot(y_coords, x_coords, color='blue')


                for stop_line in tmp_stop_lines:
                    x_coords = [point[0] for point in stop_line]
                    y_coords = [-point[1] for point in stop_line]
                    plt.plot(y_coords, x_coords, color='red')

                for line in bezier_curves:
                    x_coords = [point[0] for point in line]
                    y_coords = [-point[1] for point in line]
                    plt.plot(y_coords, x_coords, color='yellow')


                for lane in go_left_center_lane_filters:
                    for line in lane:
                        x_coords = [point[0] for point in line]
                        y_coords = [-point[1] for point in line]
                        plt.plot(y_coords, x_coords, color='gray', linestyle='-')
                for lane in go_right_center_lane_filters:
                    for line in lane:
                        x_coords = [point[0] for point in line]
                        y_coords = [-point[1] for point in line]
                        plt.plot(y_coords, x_coords, color='black', linestyle='--')
                for lane in go_straight_center_lane_filters:
                    for line in lane:
                        x_coords = [point[0] for point in line]
                        y_coords = [-point[1] for point in line]
                        plt.plot(y_coords, x_coords, color='orange', linestyle='-')
                for lane in go_back_center_lane_filters:
                    for line in lane:
                        x_coords = [point[0] for point in line]
                        y_coords = [-point[1] for point in line]
                        plt.plot(y_coords, x_coords, color='gray', linestyle=':')

                
                # for line in left_filter_pair:
                #     for 
                

                for line in forward_stop_lines:
                    x_coords = [point[0] for point in line]
                    y_coords = [-point[1] for point in line]
                    plt.plot(y_coords, x_coords, color='red')
                for line in back_stop_lines:
                    x_coords = [point[0] for point in line]
                    y_coords = [-point[1] for point in line]
                    plt.plot(y_coords, x_coords, color='red')
                for line in left_stop_lines:
                    x_coords = [point[0] for point in line]
                    y_coords = [-point[1] for point in line]
                    plt.plot(y_coords, x_coords, color='red')
                for line in right_stop_lines:
                    x_coords = [point[0] for point in line]
                    y_coords = [-point[1] for point in line]
                    plt.plot(y_coords, x_coords, color='red')
                
                plt.plot(0, 0, 'ro')

                plt.savefig('./test1.png')

                print('路口')
                import ipdb; ipdb.set_trace()
                plt.close()

        




def extract_NM_perception_clip_path(bag_root_path, num_historical_steps, num_future_steps, clip_step_length=None, sample_frequency=2):
    '''
    给出 rosbag 的存放文件夹，提取 rosbag 的 raw data 并采样成为 约等于2Hz的数据;
    然后根据 历史时间步数 和 未来时间步数 划分 clip 并返回

    args:
        bag_root_path:          rosbag 存放路径，目录级为 bag_root_floder/bag_date_floder/bag_file.bag
        num_historical_steps:   历史时间步
        num_future_steps:       未来时间步
        clip_step_length:       滑动 clip 的步长
                                i.e.: 当设置为 num_historical_steps 时, 则划分的相邻clip之间差距(不重合部分)为 num_historical_steps

    returns:
        all_segment_data_dict_list: 存放 all_segment_data_dict, rosbag 的个数就是这个 list 的长度
        all_segment_timestramp_list: 同理，存放 segment_sample_timestramp_list 的 list
    '''

    num_steps = num_historical_steps + num_future_steps

    src_raw_data_list = []
    _processed_file_names = []
    id_scenario_names = []
    all_segment_timestramp_list = []
    all_segment_data_dict_list = []
    all_segment_index_list = []
    clip_id = 0
    
    

    for bag_date in os.listdir(bag_root_path):
        print(bag_date)
        bag_date_path = os.path.join(bag_root_path, bag_date)
        bag_segment_names = os.listdir(bag_date_path)

        for segment_index, bag_segment_name in enumerate(bag_segment_names):
            print(segment_index)
            bag_segment_path = os.path.join(bag_date_path, bag_segment_name)
            bag_name = bag_segment_path.split('/')[-1].split('.')[0]
            ego_data, agent_data, map_line_data, map_element_data, ego_state_data, ego_reference_line, map_lka_line, map_lka_old_line  = _extract_raw_data_from_rosbag(bag_segment_path)

            # 将 rosbag 的 raw_data 进行固定频率的采样
            segment_sample_data_dict, segment_sample_timestramp_list = _NM_percetion_data_sample(ego_data, agent_data, map_line_data, map_element_data, ego_state_data, ego_reference_line, map_lka_line, map_lka_old_line, sample_frequency=sample_frequency)
            
            # ----- 得把 reference line 放在这里一帧帧提取  ------
            construct_reference_line(segment_sample_data_dict, segment_sample_timestramp_list)

            
            all_segment_data_dict_list.append(segment_sample_data_dict)         # all_segment_data_dict_list 可能保留被丢弃的clip，但这里主要用于记录原始数据
            all_segment_timestramp_list.append(segment_sample_timestramp_list)

            # 如果该clip长度小于设定的num_steps，则剔除
            if len(segment_sample_data_dict) < num_steps:
                continue
            else:
                # # --- Solution3： clip滑动步长由 clip_step_length 决定
                if clip_step_length is None:
                    clip_step_length = num_historical_steps

                num_stramp_segments = len(segment_sample_data_dict) // clip_step_length
                for i in range(num_stramp_segments):
                    # # --- Solution3 section2： clip滑动步长由 clip_step_length 决定
                    start_index = i * clip_step_length
                    end_index = start_index + num_steps - 1
                    if end_index >= len(segment_sample_data_dict):
                        break

                    # TODO: 生成每个存放pt文件的名字
                    scenario_name = bag_name + '_' + str(start_index) + '_' + str(end_index)
                    
                    # 存放所处理的pt文件名
                    _processed_file_names.append(scenario_name + '.pt')
                    # 这里存放的 id 和 name，会在处理时逐个提取出来
                    id_scenario_names.append((clip_id, scenario_name, start_index, end_index, segment_sample_timestramp_list[start_index + num_historical_steps - 1]))      
                    all_segment_index_list.append(segment_index)
                    clip_id += 1
                    # # 存放 get_features中需要读取文件的路径
                    # src_raw_file_path.append(abs_clip_segment_tag_path)       
            
    return all_segment_data_dict_list, all_segment_timestramp_list, all_segment_index_list, id_scenario_names



def process_and_save_NM_perception(segment_sample_data_dict, segment_sample_timestramp_list, id_scenario_name, args, output_path=None, sample_frequency=2):
    # TODO
    radius = 90

    # import pdb; pdb.set_trace
    # agent feature
    case_id = id_scenario_name[0]
    scenario_name = id_scenario_name[1]
    start_index = id_scenario_name[2]
    end_index = id_scenario_name[3]

    scenario_name_int = int(case_id)

    # 当前clip的时间戳list
    clip_timestamps = segment_sample_timestramp_list[start_index: end_index+1]

    # 求当前clip共包含多少agent
    agent_ids = []
    static_obj_ids = []
    for cur_timestamp in clip_timestamps[:args.num_historical_steps]:
        od_datas = segment_sample_data_dict[cur_timestamp]['od_data']
        for od_data in od_datas:
            if ignore_this_agent(od_data, radius):
                continue
            if is_static_obstacle(od_data):
                static_obj_ids.append(od_data['id'])   # rosbag中哪些tyep是 static obstacle
            else:
                agent_ids.append(od_data['id'])

    agent_ids = np.sort(np.unique(agent_ids))
    num_agents = len(agent_ids)
    # num_agents = num_agents
    static_obj_ids = np.sort(np.unique(static_obj_ids))
    num_static_objs = len(static_obj_ids)


    if num_agents == 0:
        there_is_no_target = True
        return

    num_steps = args.num_historical_steps + args.num_future_steps

    # pluto 使用的特征信息
    agent_position = torch.zeros((num_agents, num_steps,2 ),  dtype=torch.float)
    agent_heading = torch.zeros((num_agents, num_steps), dtype=torch.float)
    agent_velocity = torch.zeros((num_agents, num_steps, 2), dtype=torch.float)
    agent_shape = torch.zeros((num_agents, num_steps, 2), dtype=torch.float)
    agent_category = torch.zeros(num_agents, dtype=torch.int)
    agent_valid_mask = torch.zeros((num_agents, num_steps), dtype=torch.bool)
    # agent_target = torch.zeros(num_agents, num_future_steps, 3, dtype=torch.float) # TODO

    if num_static_objs == 0:
        num_static = 1
    else:
        num_static = num_static_objs
    static_obj_position = torch.zeros((num_static, num_steps, 2), dtype=torch.float)
    static_obj_heading = torch.zeros((num_static, num_steps), dtype=torch.float)
    static_obj_shape = torch.zeros((num_static, num_steps, 2), dtype=torch.float)
    static_obj_category = torch.zeros((num_static, num_steps), dtype=torch.int)
    static_obj_valid_mask = torch.zeros((num_static, num_steps), dtype=torch.bool)

    data = {}

    ego_feature = []
    ego_pose_list = []
    static_obstacle = []

    # ------- 遍历时间戳
    for stamp_index, cur_timestamp in enumerate(clip_timestamps[:args.num_historical_steps]):

        od_datas = segment_sample_data_dict[clip_timestamps[stamp_index]]['od_data']  
        
        agent_global_timestep_position_list = []
        agents_baselink_timestep_position = []
        agents_global_timestep_velocity = []

        static_global_timestep_position_list = []
        static_global_timestep_yaw_list = []
        agent_timestep_shape = []

        agent_index_list = []
        static_index_list = []
        agnets_global_timestep_yaw = []

        _agent_type = ['car', 'pedestrian', 'bicycle', 'motorcycle', 'bus', 'truck', 'tricycle', 'cone', 'barrier','barrel']

        ego_pose = torch.tensor(segment_sample_data_dict[cur_timestamp]['ego_data']['ego_pose'])
        ego_pose_list.append(ego_pose)

        # 遍历当前时间戳下所有的障碍物
        for od_data in od_datas:

            if ignore_this_agent(od_data, radius):  # NOTE: 这里的od_data已经是od_data['3d']
                continue
            
            if not is_static_obstacle(od_data):

                agent_idx = np.where(agent_ids == od_data['id'])[0]
                agent_valid_mask[agent_idx, stamp_index] = True

                # 未在历史时间帧中出现过的agent不做预测
                if agent_idx.shape[0] == 0:
                    continue
            
                # agent_id[agent_idx] = od_data['id']
                agent_index_list.append(agent_idx[0])

                # # -------------- 保存信息 ---------------
                # agent的位置信息
                agent_global_loc = ego_pose @ torch.tensor([od_data['x'], od_data['y'], 0, 1], dtype=torch.float)
                agent_global_timestep_position_list.append(agent_global_loc[:2])   
                agents_baselink_timestep_position.append([od_data['x'], od_data['y']])
                agents_global_timestep_velocity.append([od_data['vx'], od_data['vy']])

                yaw_ego = torch.atan2(ego_pose[1, 0], ego_pose[0, 0])
                yaw_agent = wrap_angle(od_data['yaw'] * np.pi / 180 + yaw_ego)
                agnets_global_timestep_yaw.append(yaw_agent)    

                # agent的属性信息
                agent_type_name = od_data['type']
                # agents_global_timestep_velocity.append(odom_v)


                # # length 、width 、type 只需要保存一次
                agent_category[agent_idx] = torch.tensor(agent_type_name, dtype=torch.int)
                agent_shape[agent_idx, stamp_index] = torch.tensor([od_data['length'], od_data['width']])

            else:
                static_idx = np.where(static_obj_ids == od_data['id'])[0]
                # 未在历史时间帧中出现过的agent不做预测
                if static_idx.shape[0] == 0:
                    continue
                static_index_list.append(static_idx[0])
                static_obj_valid_mask[static_idx, stamp_index] = True
                
                # # -------------- 保存信息 ---------------
                # agent的位置信息
                static_global_loc = ego_pose @ torch.tensor([od_data['x'], od_data['y'], od_data['z'], 1], dtype=torch.float)   
                static_global_timestep_position_list.append(static_global_loc[:2])

                yaw_ego = torch.atan2(ego_pose[1, 0], ego_pose[0, 0])
                yaw_static = wrap_angle(od_data['yaw'] * np.pi / 180 + yaw_ego)
                static_global_timestep_yaw_list.append(yaw_agent)    

                # # length 、width 、type 只需要保存一次
                static_type_name = od_data['type']
                static_obj_category[static_idx] = torch.tensor(agent_type_name, dtype=torch.int)
                static_obj_shape[static_idx, stamp_index] = torch.tensor([od_data['length'], od_data['width']])

    # 将 OD的累积坐标系 转到 当前时刻的自车坐标系
    current_timestamp = clip_timestamps[num_historical_steps - 1]
    current_ego_pose = ego_pose_list[num_historical_steps - 1]
    # 求当前时刻的ego_pose逆矩阵
    t = current_ego_pose[:3, 3]
    current_xy = t[:2]
    yaw_ego = torch.atan2(current_ego_pose[1, 0], current_ego_pose[0, 0])
    rotate_mat = torch.tensor(
            [
                [np.cos(yaw_ego), -np.sin(yaw_ego)],
                [np.sin(yaw_ego), np.cos(yaw_ego)],
            ],
            dtype=torch.float,
        )
    
    agent_position = torch.matmul(agent_position - current_xy, rotate_mat)
    agent_velocity = torch.matmul(agent_velocity, rotate_mat)
    agent_heading = agent_heading - yaw_ego

    static_obj_position = torch.matmul(static_obj_position - current_xy, rotate_mat)
    static_obj_heading = static_obj_heading - yaw_ego

    # 将 not valid 的 agnetr 和 static 信息置为 0
    agent_position[~agent_valid_mask.unsqueeze(-1).expand_as(agent_position)] = 0
    agent_velocity[~agent_valid_mask.unsqueeze(-1).expand_as(agent_velocity)] = 0
    agent_heading[~agent_valid_mask] = 0

    static_obj_position[~static_obj_valid_mask.unsqueeze(-1).expand_as(static_obj_position)] = 0
    static_obj_heading[~static_obj_valid_mask] = 0
    
    

    data['origin'] = current_xy
    data['angle'] = yaw_ego


    data['static_objects'] = {
        'static_objects': static_obj_position,
        'heading': static_obj_heading,
        'shape': static_obj_shape,
        'category': static_obj_category,
        'valid_mask': static_obj_valid_mask,
    }


    # # ======= current state
    # current_map_timestramp = map_clip_timestamp_list[num_historical_steps - 1]
    # crew = src_raw_path.split('/')
    # time_hms = crew[-3]
    # if 'changchengvv6' in crew[6]:
    #     time_date = crew[-4]
    # else:
    #     time_date = crew[-5]
    # clip_time_path = time_date + '/' + time_hms
    # raw_data_path = os.path.join(map_data_root, clip_time_path, 'vehicle/vehicle_state_10_key.txt')

    # odom = dict()
    # with open(raw_data_path, 'r') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         line_data = dict()
    #         data_split = line.split(' ')
    #         line_data[data_split[24]] = float(data_split[26])  # theta
    #         line_data[data_split[12]] = float(data_split[14])  # x
    #         line_data[data_split[15]] = float(data_split[17])  # y
    #         line_data[data_split[3]] = float(data_split[5])  # speed
    #         line_data[data_split[6]] = float(data_split[8])  # yaw_rate
    #         line_data[data_split[9]] = float(data_split[11])  # steering_angle
    #         line_data[data_split[18]] = float(data_split[20])  # acc_x
    #         line_data[data_split[21]] = float(data_split[23])  # acc_y
    #         odom[data_split[2]] = line_data

    # odom = dict()
    
    # timestamp_odom_list = list(odom.keys())
    # near_timestamp_index = find_closest_value_index(timestamp_odom_list, current_map_timestramp)
    # near_timestamp = timestamp_odom_list[near_timestamp_index]

    odom_ego_state =  segment_sample_data_dict[current_timestamp]['ego_state']  # linear_velocity、linear_acceleration、steering_wheel_angle
    odom_ego_velocity = odom_ego_state['linear_velocity']
    odom_ego_acceleration = odom_ego_state['linear_acceleration']
    odom_ego_steering = odom_ego_state['steering_wheel_angle']

    odom_ego_xy_yaw = segment_sample_data_dict[current_timestamp]['ego_data']
    current_odom_yaw = quaternion_to_yaw(odom_ego_xy_yaw['orientation'])
    current_odom_pos = torch.tensor(odom_ego_xy_yaw['position'][:2])

    last_timestamp = clip_timestamps[num_historical_steps - 2]
    last_odom_yaw = quaternion_to_yaw(segment_sample_data_dict[last_timestamp]['ego_data']['orientation'])
    yaw_rate = (current_odom_yaw - last_odom_yaw) / (1 / sample_frequency)

    odom2ego_rotation_matrix = torch.tensor([
            [np.cos(current_odom_yaw), -np.sin(current_odom_yaw)],
            [np.sin(current_odom_yaw),  np.cos(current_odom_yaw)]
        ]).float()

    current_odom_velocity = torch.tensor([odom_ego_velocity * np.cos(current_odom_yaw), odom_ego_velocity * np.sin(current_odom_yaw)]).float()
    current_odom_acceleration = torch.tensor([odom_ego_acceleration * np.cos(current_odom_yaw), odom_ego_acceleration * np.sin(current_odom_yaw)]).float()

    current_ego_velocity = current_odom_velocity @ odom2ego_rotation_matrix
    current_ego_acceleration = current_odom_acceleration @ odom2ego_rotation_matrix

    current_state = torch.zeros(7, dtype=torch.float)
    current_state[:2] = torch.tensor([0, 0])                # current_position
    current_state[2] = torch.tensor([0])                    # current_heading
    current_state[3] = current_ego_velocity[0]              # current_velocity
    current_state[4] = current_ego_acceleration[0]          # current_acceleration_x
    current_state[5] = torch.tensor(odom_ego_steering)    # current_steering_angle
    current_state[6] = torch.tensor(yaw_rate)  # current_yaw_rate
    
    data['current_state'] = current_state

    # ======== ego feature
    ego_position = torch.zeros(1, num_steps, 2, dtype=torch.float)
    ego_heading = torch.zeros(1, num_steps, dtype=torch.float)
    ego_velocity = torch.zeros(1, num_steps, 2, dtype=torch.float)
    ego_acceleration = torch.zeros(1, num_steps, 2, dtype=torch.float)
    ego_shape = torch.tensor([[[4.5, 1.8],]*num_steps])  # TODO: 自车的尺寸
    ego_category = torch.zeros(1, dtype=torch.int)
    ego_valid_mask = torch.ones(1, num_steps, dtype=torch.bool)

    for step_index, timestamp in enumerate(clip_timestamps):
        timestramp_ego_state = segment_sample_data_dict[current_timestamp]['ego_state']
        timestramp_ego_xy_yaw = segment_sample_data_dict[current_timestamp]['ego_data']

        timestamp_odom_pos = torch.tensor(timestramp_ego_xy_yaw['position'])[:2]
        timestamp_odom_heading = torch.tensor(quaternion_to_yaw(timestramp_ego_xy_yaw['orientation'])).float()
        current_odom_velocity = torch.tensor([timestramp_ego_state['linear_velocity'] * np.cos(timestamp_odom_heading), timestramp_ego_state['linear_velocity'] * np.sin(timestamp_odom_heading)]).float()
        current_odom_acceleration = torch.tensor([timestramp_ego_state['linear_acceleration'] * np.cos(timestamp_odom_heading), timestramp_ego_state['linear_acceleration'] * np.sin(timestamp_odom_heading)]).float()

        # 将 odom坐标系 转到 当前时刻的自车坐标系
        timestamp_ego_pos = (timestamp_odom_pos - current_odom_pos) @ odom2ego_rotation_matrix
        timestamp_ego_heading = timestamp_odom_heading - current_odom_yaw
        timestamp_ego_velocity = current_odom_velocity @ odom2ego_rotation_matrix
        timestamp_ego_acceleration = current_odom_acceleration @ odom2ego_rotation_matrix

        ego_position[0, step_index] = timestamp_ego_pos
        ego_heading[0, step_index] = timestamp_ego_heading
        ego_velocity[0, step_index] = timestamp_ego_velocity
        ego_acceleration[0, step_index] = timestamp_ego_acceleration

    # data['agent'] 放在这里是因为 agent 和 ego 的信息要concat
    data['agent'] = {
        'position': torch.cat((ego_position, agent_position), dim=0),
        'heading': torch.cat((ego_heading, agent_heading), dim=0),
        'velocity': torch.cat((ego_velocity, agent_velocity), dim=0),
        'shape': torch.cat((ego_shape, agent_shape), dim=0),      # 这里有一个trick： 当该时刻的agent不可见时，shape为0
        'category': torch.cat((ego_category, agent_category), dim=0),
        'valid_mask': torch.cat((ego_valid_mask, agent_valid_mask), dim=0)
    }

    # 参考 pluto的target实现方式： 未来每个时刻的position - 当前时刻的position；未来每个时刻的heading - 当前时刻heading
    target_position = (
        data["agent"]["position"][:, num_historical_steps:]
        - data["agent"]["position"][:, num_historical_steps - 1][:, None]
    )
    target_heading = (
        data["agent"]["heading"][:, num_historical_steps:]
        - data["agent"]["heading"][:, num_historical_steps - 1][:, None]
    )
    target = torch.cat([target_position, target_heading[..., None]], -1)
    target[~data["agent"]["valid_mask"][:, num_historical_steps:]] = 0
    data["agent"]["target"] = target

    # import ipdb; ipdb.set_trace()

    # ======== map feature
    map_line_data = segment_sample_data_dict[current_timestamp]['map_line_data']  # map_line_data 目前放的是所有车道线，包含'points', 'type', 'color', 'confidence'
    map_element_data = segment_sample_data_dict[current_timestamp]['map_element_data']

    all_line_type = ['solid-lane', 'dash-lane', 'edge', 'wide-dash-lane', 'wide-solid-lane', 'double-solid-lane', 'double-dash-lane', 'dash-solid-lane', 'variable-direction-lane', 'pending-transfer-area-dash-lane', 'drainage-lane', 'slow-down-dash-lane', 'unknown', 'road-edge'] # unknown后面的是 所有road-edge的type
    all_edge_type = ['road-edge', 'traffic-barrier', 'cone-curb', 'water-horse-curb', 'column-curb', 'crash-barrel-curb', 'unknown']
    all_color_type = ['white', 'yellow', 'orange', 'blue', 'green', 'red', 'unknown']
    line_list = []
    line_type_list = []
    line_confident_list = []
    line_id_list = []
    line_color_list = []
    global_line_list = []


    for single_line in map_line_data:
        if len(single_line['points']) > 1:
            line_list.append(single_line['points'])
            line_type_list.append(single_line['type'])
            line_confident_list.append(single_line['confidence'])

    tmp_putup_line_list = copy.deepcopy(line_list)
    tmp_putup_line_list_new, line_type_list = nullmax_junction_utils.put_up_lines(tmp_putup_line_list, line_type_list)

    tmp_putup_line_list = tmp_putup_line_list_new
    # tmp_putup_line_list = []
    # for line in tmp_putup_line_list_new:
    #     if check_angle(line):
    #         tmp_putup_line_list.append(line)

    if len(map_element_data) == 0:
        return # 非路口区域
    
    # import ipdb; ipdb.set_trace()
    tmp_stop_lines = [stop_line['points'] for stop_line in map_element_data]

    stop_lines = []
    # 只有距离大于3m的stop_line才进行保留
    for line in tmp_stop_lines:
        # stop_line = loc2global_egopose(line['Coords'], current_ego_pose)
        if nullmax_junction_utils.calculate_distance(line[0], line[-1]) > 3:
            stop_lines.append(line)


    current_position = current_state[:2]
    if not test_nullmax_exits.is_junction_scene(stop_lines, current_state[:2]):
        return None # 非路口区域
    
    debug_mode = False
    forward_stop_lines, back_stop_lines, left_stop_lines, right_stop_lines, all_stop_lines = test_nullmax_exits.classify_stoplines(
        stop_lines, current_position, debug_mode, None, None, None)
    

    # 延长筛选出的停止线，目前设置的最大延伸长度为 20m
    left_extend_stop_lines = nullmax_junction_utils.extend_stop_lines(left_stop_lines)
    right_extend_stop_lines = nullmax_junction_utils.extend_stop_lines(right_stop_lines)
    forward_extend_stop_lines = nullmax_junction_utils.extend_stop_lines(forward_stop_lines)
    back_extend_stop_lines = nullmax_junction_utils.extend_stop_lines(back_stop_lines)

    # lane_filter 应该就是 centerline， pair是车道线对
    go_left_center_lane_filters, left_filter_pair, go_right_center_lane_filters, right_filter_pair, \
    go_straight_center_lane_filters, forward_filter_pair, go_back_center_lane_filters, back_filter_pair = test_nullmax_exits.get_centerlines(
        left_stop_lines, right_stop_lines, forward_stop_lines, back_stop_lines, tmp_putup_line_list, line_type_list,
        left_extend_stop_lines, right_extend_stop_lines, forward_extend_stop_lines, back_extend_stop_lines)

    TURN_LEFT = 2
    TURN_RIGHT = 3
    TURN_LEFT_FRONT = 4
    TURN_RIGHT_FRONT = 5
    TURN_LEFT_BACK = 6
    TURN_RIGHT_BACK = 7
    TURN_LEFT_AND_AROUND = 8
    GO_STRAIGHT = 9
    MERGE_LEFT = 65
    MERGE_RIGHT = 66

    selected_lane_filters = None
    # direction = test_sdmap.generate_info(scenes_token, sample, map_ann_path)
    direction = TURN_LEFT
    sd_map_direction_centerlines = []
    is_straight = False

    if direction == GO_STRAIGHT:
        print("sd map go straight: " + str(direction))
        selected_lane_filters = go_straight_center_lane_filters
        is_straight = True
    elif direction == TURN_LEFT or direction == TURN_LEFT_FRONT or direction == TURN_LEFT_BACK or direction == MERGE_LEFT:
        print("sd map turn left: " + str(direction))
        selected_lane_filters = go_left_center_lane_filters
    elif direction == TURN_RIGHT or direction == TURN_RIGHT_FRONT or direction == TURN_RIGHT_BACK or direction == MERGE_RIGHT:
        print("sd map turn right: " + str(direction))
        selected_lane_filters = go_right_center_lane_filters
    elif direction == TURN_LEFT_AND_AROUND:
        print("sd map turn around: " + str(direction))
        selected_lane_filters = go_back_center_lane_filters
    else:
        return None
        # print("default go straight: " + str(direction))
        # selected_lane_filters = go_straight_center_lane_filters
        # is_straight = True

    if selected_lane_filters is not None:
        for selected_lane_filter in selected_lane_filters:
          for odom_line in selected_lane_filter:
            sd_map_direction_centerlines.append(odom_line)
    
    # tmp
    future_trajectory = torch.tensor([[0,0], [0,20], [0,30], [0,40], [0,50], [0,60], [0, 70]])

    bezier_curves_first, start_point, start_and_end_points = test_nullmax_exits.generate_bezier_reference_lines(future_trajectory, 
        stop_lines, sd_map_direction_centerlines, is_straight)
        
    bezier_curves = test_nullmax_exits.get_same_length_curves(bezier_curves_first)

    data['reference_line'] = bezier_curves

    
    # num_points = 20

    # line_dict = {}
    # for single_line in map_line_data:
    #     line_dict[single_line['ID']] = single_line

    # lane_nums = len(map_line_data)   # pluto中 lane lane_connector crosswalk的总和
    # map_point_position = torch.zeros(lane_nums, 3, num_points, 2, dtype=torch.float)
    # map_point_vector = torch.zeros(lane_nums, 3, num_points, 2, dtype=torch.float)
    # map_point_side = torch.zeros((lane_nums, 3), dtype=torch.int)
    # map_point_orientation = torch.zeros((lane_nums, 3, num_points), dtype=torch.float)
    # map_polygon_center = torch.zeros((lane_nums, 3), dtype=torch.float)
    # map_polygon_position = torch.zeros((lane_nums, 2), dtype=torch.float)
    # map_polygon_orientation = torch.zeros(lane_nums, dtype=torch.float)
    # map_polygon_type = torch.zeros(lane_nums, dtype=torch.int)
    # map_polygon_on_route = torch.zeros(lane_nums, dtype=torch.bool)
    # map_polygon_tl_status = torch.zeros(lane_nums, dtype=torch.int)
    # map_polygon_speed_limit = torch.ones(lane_nums, dtype=torch.float) * 60
    # map_polygon_has_speed_limit = torch.zeros(lane_nums, dtype=torch.bool)
    # map_polygon_road_block_id = torch.zeros(lane_nums, dtype=torch.int)
    
    # for lane_index, single_lane in enumerate(lane_seg_list):

    #     centerline = crop_line_within_radius(single_lane['Coords'], radius)

    #     if len(single_lane['Left_Lane']) > 1:
    #         leftline = []
    #         for single_line in single_lane['Left_Lane']:
    #             leftline = leftline + line_dict[single_line['ID']]['Coords'][1:]
    #     else:
    #         leftline = line_dict[single_line['ID']]['Coords']
    #     leftline = crop_line_within_radius(leftline, radius)

    #     if len(single_lane['Right_Lane']) > 1:
    #         rightline = []
    #         for single_line in single_lane['Right_Lane']:
    #             rightline = rightline + line_dict[single_line['ID']]['Coords'][1:]
    #     else:
    #         rightline = line_dict[single_line['ID']]['Coords']
    #     rightline = crop_line_within_radius(rightline, radius)  
    
    #     sample_points = num_points + 1 # +1 是因为要计算 point_vector，最终保存点的时候最后一个点会丢弃
    #     if len(centerline) == 0 and len(leftline) == 0 and len(rightline) == 0:
    #         continue
    #     if len(centerline) > 0:
    #         centerline_resample = resample_points(torch.tensor(centerline)[:,:2], sample_points)
    #     else:
    #         centerline_resample = torch.zeros(21,2)
    #     if len(leftline) > 0:
    #         leftline_resample = resample_points(torch.tensor(leftline)[:,:2], sample_points)
    #     else:
    #         leftline_resample = torch.zeros(21,2)
    #     if len(rightline) > 0:
    #         rightline_resample = resample_points(torch.tensor(rightline)[:,:2], sample_points)
    #     else:
    #         rightline_resample = torch.zeros(21,2)
    #     lines = torch.stack([centerline_resample, leftline_resample, rightline_resample], axis=0)

    #     # 构造 map feature
    #     map_point_position[lane_index] = lines[:, :-1]
    #     map_point_vector[lane_index] = lines[:, 1:] - lines[:, :-1]
    #     map_point_orientation[lane_index] = torch.atan2(map_point_vector[lane_index, :, :, 1], map_point_vector[lane_index, :, :, 0])
    #     map_point_side[lane_index] = np.arange(3)
    #     map_polygon_center[lane_index] = torch.cat(
    #         [
    #             centerline_resample[int(sample_points / 2)],
    #             [map_point_orientation[lane_index, 0, int(sample_points / 2)]],
    #         ],
    #         axis = -1,
    #     )
    #     map_polygon_position[lane_index] = centerline_resample[0]
    #     map_polygon_orientation[lane_index] = map_point_orientation[lane_index, 0, 0]
    #     # map_polygon_type[lane_index] =  # TODO: self
    #     # map_polygon_on_route[lane_index] # TODO
    #     # map_polygon_tl_status[lane_index] # TODO
    #     # map_polygon_has_speed_limit[lane_index] # TODO
    #     map_polygon_road_block_id[lane_index] = single_lane['ID']

    # # TODO: 检查是否需要翻转点的坐标；检查当有多条线时，是否就是第一个 加 第二个
    # map_features = {
    #     "point_position": map_point_position,
    #     "point_vector": map_point_vector,
    #     "point_orientation": map_point_orientation,
    #     "point_side": map_point_side,
    #     "polygon_center": map_polygon_center,
    #     "polygon_position": map_polygon_position,
    #     "polygon_orientation": map_polygon_orientation,
    #     "polygon_type": map_polygon_type,
    #     "polygon_on_route": map_polygon_on_route,
    #     "polygon_tl_status": map_polygon_tl_status,
    #     "polygon_has_speed_limit": map_polygon_has_speed_limit,
    #     "polygon_speed_limit": map_polygon_speed_limit,
    #     "polygon_road_block_id": map_polygon_road_block_id,
    # }

    # data['map'] = map_features

    # flag_vis = True
    # if flag_vis:
    #     # for agent in data['agent']['position']:
    #     #     x_coords = []
    #     #     y_coords = []
    #     #     for point in agent:
    #     #         if point[0] != 0 and point[1] != 0:
    #     #             x_coords.append(point[0])
    #     #             y_coords.append(-point[1])
    #     #     plt.plot(y_coords, x_coords, color='green')

    #     for lane in segment_sample_data_dict[current_timestamp]['map_line_data']:

    #         x_coords = [point[0] for point in lane['points']]
    #         y_coords = [-point[1] for point in lane['points']]

    #         diff = torch.tensor(lane['points'][0]) - torch.tensor(lane['points'][-1])
    #         angle = torch.atan2(diff[0], diff[1]) * 180 / 3.1415926265
    #         print(angle)

    #         # import ipdb; ipdb.set_trace()
            
    #         # if -10 < angle < 10 or 80 < angle < 100:
    #         #     color = 'gray'
    #         # else:
    #         #     color = 'blue'

    #         # if lane['type'] == 7:
    #         #     color = 'pink'

    #         if lane['position'] in [1, 2]:
    #             color = 'green'
    #             linestyle = '-'
    #         if lane['position'] == [3,4]:
    #             color = 'blue'
    #             linestyle = '-'
    #         if lane['position'] == [5,6]:
    #             color = 'orange'
    #             linestyle = '--'
    #         if lane['position'] == 7:
    #             color = 'pink'
    #             linestyle = '.'
    #         if lane['position'] == 8:
    #             color = 'brown'
    #             # linestyle = '--'
    #         if lane['position'] in [9,10]:
    #             color = 'black'
    #         else:
    #             color = 'gray'
            
    #         plt.plot(y_coords, x_coords, color=color)

    #         # plt.savefig('./test1.png')

    #         # import ipdb; ipdb.set_trace()

    #     # for line in tmp_putup_line_list:
    #     #     x_coords = [point[0] for point in line]
    #     #     y_coords = [-point[1] for point in line]
    #     #     plt.plot(y_coords, x_coords)

    #     for stop_line in tmp_stop_lines:
    #         x_coords = [point[0] for point in stop_line]
    #         y_coords = [-point[1] for point in stop_line]
    #         plt.plot(y_coords, x_coords, color='red')

    #     for line in data['reference_line']:
    #         x_coords = [point[0] for point in line]
    #         y_coords = [-point[1] for point in line]
    #         plt.plot(y_coords, x_coords, color='yellow')


    #     for lane in go_left_center_lane_filters:
    #         for line in lane:
    #             x_coords = [point[0] for point in line]
    #             y_coords = [-point[1] for point in line]
    #             plt.plot(y_coords, x_coords, color='gray', linestyle='-')
    #     for lane in go_right_center_lane_filters:
    #         for line in lane:
    #             x_coords = [point[0] for point in line]
    #             y_coords = [-point[1] for point in line]
    #             plt.plot(y_coords, x_coords, color='black', linestyle='--')
    #     for lane in go_straight_center_lane_filters:
    #         for line in lane:
    #             x_coords = [point[0] for point in line]
    #             y_coords = [-point[1] for point in line]
    #             plt.plot(y_coords, x_coords, color='orange', linestyle='-')
    #     for lane in go_back_center_lane_filters:
    #         for line in lane:
    #             x_coords = [point[0] for point in line]
    #             y_coords = [-point[1] for point in line]
    #             plt.plot(y_coords, x_coords, color='gray', linestyle=':')

        
    #     # for line in left_filter_pair:
    #     #     for 
        

    #     for line in forward_stop_lines:
    #         x_coords = [point[0] for point in line]
    #         y_coords = [-point[1] for point in line]
    #         plt.plot(y_coords, x_coords, color='red')
    #     for line in back_stop_lines:
    #         x_coords = [point[0] for point in line]
    #         y_coords = [-point[1] for point in line]
    #         plt.plot(y_coords, x_coords, color='red')
    #     for line in left_stop_lines:
    #         x_coords = [point[0] for point in line]
    #         y_coords = [-point[1] for point in line]
    #         plt.plot(y_coords, x_coords, color='red')
    #     for line in right_stop_lines:
    #         x_coords = [point[0] for point in line]
    #         y_coords = [-point[1] for point in line]
    #         plt.plot(y_coords, x_coords, color='red')
        
    #     plt.plot(0, 0, 'ro')

    #     plt.savefig('./test2.png')

    #     import ipdb; ipdb.set_trace()
    #     plt.close()





def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag_file_root", type=str, required=False, default='/media/ubuntu/fd8ddf16-e236-4dc0-baa6-e731e9e806c9/rosbag/demo_raw_data', help="Path to raw data")
    parser.add_argument(
        "--dataset", type=str, required=False, help="'train' or 'val' or 'test'")
    parser.add_argument(
        "--output-path", type=str, required=False, help="Path to save data")
    
    parser.add_argument(
        "--num_historical_steps", type=int, default=4, required=False, help=" ")
    parser.add_argument(
        "--num_future_steps", type=int, default=6, required=False, help=" ")
    parser.add_argument(
        "--clip_step_length", type=int, default=4, required=False, help=" ")
    
    parser.add_argument(
        "--n-jobs", type=int, default=1, required=False,
        help="Number of threads")
    parser.add_argument(
        "--n-shards", type=int, default=1, required=False,
        help="Use `1/n_shards` of full dataset")
    parser.add_argument(
        "--shard-id", type=int, default=0, required=False,
        help="Take shard with given id")
    parser.add_argument(
        "--config", type=str, required=False,
        help="Config file path")
    parser.add_argument(
        "--sample_frequency", default=2, type=int, required=False,
        help="sample_frequency")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()

    bag_file_root = args.bag_file_root
    
    num_historical_steps = args.num_historical_steps
    num_future_steps = args.num_future_steps
    clip_step_length = num_historical_steps
    
    all_segment_sample_data_dict, all_segment_sample_timestramp_list, all_segment_index_list, id_scenario_names = \
          extract_NM_perception_clip_path(bag_file_root, num_historical_steps, num_future_steps, clip_step_length=clip_step_length, \
                                          sample_frequency = args.sample_frequency )
    
    
    for segment_index, id_scenario_name in zip(all_segment_index_list, id_scenario_names):
        segment_sample_data_dict = all_segment_sample_data_dict[segment_index]
        segment_sample_timestramp_list = all_segment_sample_timestramp_list[segment_index]

        process_and_save_NM_perception(segment_sample_data_dict=segment_sample_data_dict,
                                    segment_sample_timestramp_list=segment_sample_timestramp_list,
                                    id_scenario_name=id_scenario_name,
                                    args=args, sample_frequency = args.sample_frequency
                                    )
        
    
