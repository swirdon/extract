
import pickle
import matplotlib.pyplot as plt
import numpy as np

def draw_flag_vis_from_pkl(pkl_path, timestamp=None, save_path=None):
    """
    从 pkl 文件中加载数据并使用 flag_vis 控制画图
    """
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)

    if not isinstance(data_dict, dict):
        raise ValueError("pkl 文件内容不是字典格式")

    all_timestamps = list(data_dict.keys())
    if not all_timestamps:
        raise ValueError("没有找到任何时间戳数据")

    # 自动选择中间帧
    if timestamp is None:
        timestamp = all_timestamps[len(all_timestamps) // 2]

    current_data = data_dict[timestamp]
    if not isinstance(current_data, dict):
        raise ValueError("单个时间戳下数据格式异常")

    plt.figure(figsize=(10, 10))
    plt.title(f"Visualization @ timestamp {timestamp}")

    # ===== map_lane_data 车道线可视化 =====
    for lane in current_data.get('map_line_data', []):
        x_coords = [pt[0] for pt in lane['points']]
        y_coords = [-pt[1] for pt in lane['points']]  # y轴取反

        # 颜色 & 样式根据 lane 的 position 属性设置
        position = lane.get('position', None)
        color = 'gray'
        linestyle = '-'

        if position in [1, 2]:
            color = 'green'
        elif position in [3, 4]:
            color = 'blue'
        elif position in [5, 6]:
            color = 'orange'
            linestyle = '--'
        elif position == 7:
            color = 'pink'
        elif position == 8:
            color = 'brown'
        elif position in [9, 10]:
            color = 'black'

        plt.plot(y_coords, x_coords, color=color, linestyle=linestyle)

    # ===== stop_line 可视化（红色） =====
    for line in current_data.get('map_element_data', []):
        x_coords = [pt[0] for pt in line['points']]
        y_coords = [-pt[1] for pt in line['points']]
        plt.plot(y_coords, x_coords, color='red')

    # ===== reference_line 可视化（黄色） =====
    for line in current_data.get('reference_line', []):
        x_coords = [pt[0] for pt in line]
        y_coords = [-pt[1] for pt in line]
        plt.plot(y_coords, x_coords, color='yellow')

    # ===== 画出原点 ego =====
    plt.plot(0, 0, 'ro', label="Ego")

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("Y (m)")
    plt.ylabel("X (m)")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ 图像已保存至: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 修改这里以使用你自己的 PKL 路径
    pkl_file = "/home/ubuntu/Downloads/bag_to_pkl/demo_bag_to_pkl.pkl"
    draw_flag_vis_from_pkl(pkl_file)
