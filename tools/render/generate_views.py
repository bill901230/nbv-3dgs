import pyvista as pv
# pv.start_xvfb()
# import open3d as o3d
import os
import numpy as np

# def mesh_to_pcd(mesh, output_pcd_path):
#     """
#     Convert a mesh to a point cloud and save it as .pcd file.
    
#     - mesh: PyVista PolyData object
#     - output_pcd_path: Path to save the .pcd file
#     """
#     points = mesh.points  # 提取 mesh 的頂點 (N, 3)
    
#     # 創建 Open3D 點雲物件
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)  # 設定點雲座標
    
#     # 存儲為 .pcd 文件
#     o3d.io.write_point_cloud(output_pcd_path, pcd)
#     print(f"Point cloud saved to {output_pcd_path}")
    
def generate_spherical_points(radius):
    coordinates = []
    for phi in range(0, 121, 30):
        rad_phi = np.radians(phi)
        for index in range(1, 9):
            theta = index * 45
            rad_theta = np.radians(theta)
            sin_phi = np.sin(rad_phi).tolist()
            cos_phi = np.cos(rad_phi).tolist()
            sin_theta = np.sin(rad_theta).tolist()
            cos_theta = np.cos(rad_theta).tolist()
            
            # coordinates.append((radius*sin_phi*cos_theta, radius*sin_phi*sin_theta, radius*cos_phi+radius*0.5))
            coordinates.append((radius*sin_phi*cos_theta, radius*sin_phi*sin_theta, radius*cos_phi))
            if phi == 0:
                break

    return coordinates

def generate_hemisphere_views(input_dir, output_dir, radius = 0.5):
    """
    Generate specified number of screenshot on the object.

    
    - input_dir: A directory contains .obj .mtl .png file 

    - output_dir: views_num of screenshot will be stored there.

    - views_num: the number of view distribute this hemisphere.

    """
    for file_name in os.listdir(input_dir):
        ext = os.path.splitext(file_name)[1].lower()
        if '.obj' == ext:
            obj_path = os.path.join(input_dir, file_name)
        elif '.mtl' == ext:
            mtl_path = os.path.join(input_dir, file_name)
        elif ext in ['.png', '.jpg']:
            texture_path = os.path.join(input_dir, file_name)

    if os.path.exists(obj_path):
        mesh = pv.read(obj_path)
    else:
        print('ERROR - .obj not found')
        return

    if os.path.exists(texture_path):
        texture = pv.read_texture(texture_path)
    else:
        print("WARNING - no texture")

    
    transformed_coordinates = generate_spherical_points(radius)

    for i, pos in enumerate(transformed_coordinates):
        camera = pv.Camera()
        camera.position = (pos[0], pos[1], pos[2])
        camera.focal_point = (0, 0, 0)
        sight_dir = (camera.focal_point[0]-camera.position[0], camera.focal_point[1]-camera.position[1], camera.focal_point[2]-camera.position[2])
        camera.up = np.cross(np.cross(sight_dir, (0, 0, 1)), sight_dir)

        axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=0.5)
        axes.origin = (0.0, 0.0, 0.0)

        plotter = pv.Plotter(off_screen=True)
        plotter.camera = camera
        plotter.add_actor(axes.actor)


        copy_mesh = mesh.copy()
        old_centroid = copy_mesh.center
        copy_mesh.translate([-old_centroid[0], -old_centroid[1], -old_centroid[2]], inplace = True) # translate to origin
        copy_mesh.rotate_vector([1,0,0], angle=90, inplace=True, point=axes.origin)                        # rotate upward vector to z axis
        # copy_mesh.translate([old_centroid[0], old_centroid[2], old_centroid[1]], inplace = True)    # move back to ground
        centroid = copy_mesh.center


        plotter.add_mesh(copy_mesh, texture=texture)
        img_path = os.path.join(output_dir, f"view_{i:02d}.png")  # 儲存圖片
        plotter.screenshot(img_path)

    print(f"All screenshots are saved in {output_dir}")

if __name__ == '__main__':
    
    output_dir = '../../data/house3k/HOUSE48/view/'
    input_dir = '../../data/house3k/HOUSE48/'

    radius = 1
    generate_hemisphere_views(input_dir, output_dir, radius)