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
    
def generate_spherical_viewpoints(radius):
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
1