import pyvista as pv
import os
import numpy as np
import json


def generate_videoframe(input_dir, output_dir, metadata_file, n_frames=143,
                        start_pose=(1.0, 0.0, 0.0), end_pose=(0.0, 1.0, 0.0),
<<<<<<< HEAD
                        image_size=(640, 480), focal_length=500):
=======
                        image_size=(640, 480)):
>>>>>>> origin/weiling

    # 確保變數初始化
    obj_path, texture_path = None, None

    # 搜尋 .obj, .png/.jpg
    for file_name in os.listdir(input_dir):
        ext = os.path.splitext(file_name)[1].lower()
        if ext == '.obj':
            obj_path = os.path.join(input_dir, file_name)
        elif ext in ['.png', '.jpg']:
            texture_path = os.path.join(input_dir, file_name)

    if obj_path is None or not os.path.exists(obj_path):
        print('ERROR - .obj not found')
        return

    # 讀取 mesh
    mesh = pv.read(obj_path)

    # 讀取貼圖
    texture = None
    if texture_path and os.path.exists(texture_path):
        try:
            texture = pv.read_texture(texture_path)
        except:
            print("WARNING - Failed to load texture")
    else:
        print("WARNING - No texture found")

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 計算相機的移動步長
    unit_move = np.array([(end_pose[i] - start_pose[i]) / n_frames for i in range(3)])

    # 儲存 metadata
    metadata = {}

    for i in range(n_frames):
        camera = pv.Camera()
        camera.position = np.array(start_pose) + i * unit_move
        camera.focal_point = np.array([0, 0, 0])
        sight_dir = np.array(camera.focal_point) - np.array(camera.position)
        camera.up = np.cross(np.cross(sight_dir, [0, 0, 1]), sight_dir)

        plotter = pv.Plotter(off_screen=True)
        plotter.camera = camera

        # 計算 rotation（3x3）
        z_axis = sight_dir / np.linalg.norm(sight_dir)
        x_axis = np.cross(camera.up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3

        # 計算 extrinsics（4x4）
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation_matrix
        extrinsics[:3, 3] = -rotation_matrix @ camera.position  # T = -R * C

        # 設定 intrinsics（3x3 矩陣）
<<<<<<< HEAD
=======
        focal_length = camera.focal_point - camera.position
>>>>>>> origin/weiling
        intrinsics = np.array([
            [focal_length, 0, image_size[0] / 2],  # fx, 0, cx
            [0, focal_length, image_size[1] / 2],  # 0, fy, cy
            [0, 0, 1]  # 0, 0, 1
        ])

        # 儲存相機資訊
        metadata[str(i)] = {
            "position": list(camera.position),
            "rotation": rotation_matrix.tolist(),
            "intrinsics": intrinsics.tolist(),
            "extrinsics": extrinsics.tolist(),
            "near": 0.1,
            "far": 10.0,
            "image_path": f"view_{i:03d}.png"

        }

        # 處理 mesh 旋轉和對齊
        copy_mesh = mesh.copy()
        old_centroid = copy_mesh.center
        copy_mesh.translate(-np.array(old_centroid), inplace=True)  # 移動到原點
        copy_mesh.rotate_vector([1, 0, 0], angle=90, inplace=True)  # 旋轉向上
<<<<<<< HEAD
        copy_mesh.translate(old_centroid, inplace=True)  # 移回原點
=======
        # copy_mesh.translate(old_centroid, inplace=True)  # 移回原點
>>>>>>> origin/weiling

        # 添加 mesh 和貼圖
        if texture:
            plotter.add_mesh(copy_mesh, texture=texture)
        else:
            plotter.add_mesh(copy_mesh)

        # 儲存圖片
        img_path = os.path.join(output_dir, f"view_{i:03d}.png")
        plotter.screenshot(img_path)

    # 儲存 metadata
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"All screenshots are saved in {output_dir}")
    print(f"Camera metadata saved to {metadata_file}")


if __name__ == '__main__':
<<<<<<< HEAD
    output_dir = './data/house3k_test/48/view/'
    input_dir = './data/house3k_test/48/'
=======
    output_dir = './data/house3k/HOUSE48/view/'
    input_dir = './data/house3k/HOUSE48/'
>>>>>>> origin/weiling
    metadata_file = os.path.join(output_dir, "camera_metadata.json")

    generate_videoframe(input_dir, output_dir, metadata_file)
