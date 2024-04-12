import json
import os
import shutil
from glob import glob

# 新的指定目录
final_output_path = '/workspace/Octree-GS/data/matrix_city/images/'
os.makedirs(final_output_path, exist_ok=True)  # 确保新目录存在

# 合并后的 frames 列表
merged_frames = []

# 处理每个 block 文件夹
for block_index in range(1, 11):
    block_folder_name = f"block_{block_index}"
    data_directory_path = f'/workspace/Octree-GS/data/train/{block_folder_name}/'
    json_file_path = os.path.join(data_directory_path, "transforms.json")

    # 读取 JSON 文件
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    frame_indices = {frame['frame_index'] for frame in json_data['frames']}

    # 删除和重命名 PNG 文件
    png_files = glob(os.path.join(data_directory_path, '*.png'))
    for png_file in png_files:
        file_index = int(os.path.basename(png_file)[:-4])
        if file_index not in frame_indices:
            os.remove(png_file)
        else:
            new_file_name = f"block_{block_index:02d}_{file_index:04d}.png"
            os.rename(png_file, os.path.join(data_directory_path, new_file_name))

            # 移动 PNG 文件到新目录
            shutil.move(os.path.join(data_directory_path, new_file_name), final_output_path)

    # 更新 JSON 文件中的 frame_index
    for frame in json_data['frames']:
        new_file_name = f"block_{block_index:02d}_{frame['frame_index']:04d}"
        frame['frame_index'] = new_file_name
        merged_frames.append(frame)

    # 保存新的 JSON 文件
    new_json_file_path = os.path.join(data_directory_path, 'transforms_latest.json')
    with open(new_json_file_path, 'w') as new_json_file:
        json.dump(json_data, new_json_file, indent=4)

# 检查 PNG 文件是否移动成功
moved_png_files = glob(os.path.join(final_output_path, '*.png'))
if len(moved_png_files) != len(merged_frames):
    raise ValueError("The number of moved PNG files does not match the number of frames in the JSON data.")

# 保存合并后的 JSON 文件
final_json_data = {
    "camera_angle_x": 0.7853981852531433,  # TODO: 目测内参一致
    "frames": merged_frames
}
final_json_path = os.path.join(os.path.dirname(os.path.dirname(final_output_path)), 'transforms.json')

with open(final_json_path, 'w') as final_json_file:
    json.dump(final_json_data, final_json_file, indent=4)

# 最终验证
if len(glob(os.path.join(final_output_path, '*.png'))) != len(merged_frames):
    raise ValueError("The number of PNG files in the final output path does not match the number of frames in the final JSON.")
