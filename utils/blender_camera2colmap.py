import numpy as np
import json
import os
import argparse
from scipy.spatial.transform import Rotation as R

# Define the argument parser to read input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Convert Blender camera data to COLMAP format.")
    parser.add_argument('--json_path', default='transforms.json', help="Path to the Blender JSON file (default: 'transforms_train.json').")
    parser.add_argument('--cameras_path', default='cameras.txt', help="Output path for the COLMAP cameras.txt file (default: 'cameras.txt').")
    parser.add_argument('--images_path', default='images.txt', help="Output path for the COLMAP images.txt file (default: 'images.txt').")
    parser.add_argument('--points_path', default='points3D.txt', help="Output path for the COLMAP points3D.txt file (default: 'points3D.txt').")
    parser.add_argument('--image_dir', default='images', help="Directory containing images (default: 'images').")
    parser.add_argument('--width', type=int, default=1920, help="Width of the images (default: 1920).")
    parser.add_argument('--height', type=int, default=1080, help="Height of the images (default: 1080).")
    return parser.parse_args()

def rotmat_to_quaternion(rot_mat):
    # Assuming rot_mat is a 4x4 matrix, we extract the 3x3 rotation part
    rotation = R.from_matrix(rot_mat[:3, :3])
    # Convert to quaternion (x, y, z, w)
    return rotation.as_quat()

def main():
    args = parse_args()

    H = args.height
    W = args.width

    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    fnames = list(sorted(os.listdir(args.image_dir)))
    fname2pose = {}

    with open(args.json_path, 'r') as f:
        meta = json.load(f)

    fx = 0.5 * W / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
    fy = fx if 'camera_angle_y' not in meta else 0.5 * H / np.tan(0.5 * meta['camera_angle_y'])
    cx, cy = 0.5 * W, 0.5 * H  # Default center to the middle of the image

    with open(args.cameras_path, 'w') as f:
        f.write(f'1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n')

    with open(args.images_path, 'w') as f:
        idx = 1
        for fname,frame in zip(fnames,meta['frames']):
            # fname = frame['file_path'].split('/')[-1]
            if not (fname.endswith('.png') or fname.endswith('.jpg')):
                fname += '.png'
            pose = np.array(frame['rot_mat']) @ blender2opencv
            fname2pose[fname] = pose

            quat = rotmat_to_quaternion(pose)
            # Transform translation
            T = -np.matmul(pose[:3, :3].T, pose[:3, 3])
            # Write the quaternion (w, x, y, z) and translation to file
            f.write(f"{idx} {quat[3]} {quat[0]} {quat[1]} {quat[2]} {T[0]} {T[1]} {T[2]} 1 {fname}\n")
            f.write('\n')
            idx += 1

    with open(args.points_path, 'w') as f:
        f.write('')

if __name__ == '__main__':
    main()
