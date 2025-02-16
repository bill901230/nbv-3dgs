import pyvista as pv
import os
import numpy as np

coordinates_1 = [
    (-4.329780281177467662e-17, 1.000000000000000000e+00, 4.329780281177465196e-17),
(0.000000000000000000e+00, 8.660254037844385966e-01, 5.000000000000001110e-01),
(3.535533905932737864e-01, 8.660254037844385966e-01, 3.535533905932738419e-01),
(5.000000000000001110e-01, 8.660254037844385966e-01, 3.061616997868383634e-17),
(3.535533905932738419e-01, 8.660254037844385966e-01, -3.535533905932737864e-01),
(6.123233995736767268e-17, 8.660254037844385966e-01, -5.000000000000001110e-01),
(-3.535533905932737864e-01, 8.660254037844385966e-01, -3.535533905932738974e-01),
(-5.000000000000001110e-01, 8.660254037844385966e-01, -9.184850993605150903e-17),
(-3.535533905932738974e-01, 8.660254037844385966e-01, 3.535533905932737309e-01),
(0.000000000000000000e+00, 4.999999999999999445e-01, 8.660254037844387076e-01),
(6.123724356957944703e-01, 4.999999999999999445e-01, 6.123724356957945814e-01),
(8.660254037844387076e-01, 4.999999999999999445e-01, 5.302876193624534574e-17),
(6.123724356957945814e-01, 4.999999999999999445e-01, -6.123724356957944703e-01),
(1.060575238724906915e-16, 4.999999999999999445e-01, -8.660254037844387076e-01),
(-6.123724356957944703e-01, 4.999999999999999445e-01, -6.123724356957946924e-01),
(-8.660254037844387076e-01, 4.999999999999999445e-01, -1.590862858087360249e-16),
(-6.123724356957946924e-01, 4.999999999999999445e-01, 6.123724356957944703e-01),
(0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00),
(7.071067811865474617e-01, 0.000000000000000000e+00, 7.071067811865475727e-01),
(1.000000000000000000e+00, 0.000000000000000000e+00, 6.123233995736766036e-17),
(7.071067811865475727e-01, 0.000000000000000000e+00, -7.071067811865474617e-01),
(1.224646799147353207e-16, 0.000000000000000000e+00, -1.000000000000000000e+00),
(-7.071067811865474617e-01, 0.000000000000000000e+00, -7.071067811865476838e-01),
(-1.000000000000000000e+00, 0.000000000000000000e+00, -1.836970198721029688e-16),
(-7.071067811865476838e-01, 0.000000000000000000e+00, 7.071067811865473507e-01),
(0.000000000000000000e+00, -4.999999999999999445e-01, 8.660254037844387076e-01),
(6.123724356957944703e-01, -4.999999999999999445e-01, 6.123724356957945814e-01),
(8.660254037844387076e-01, -4.999999999999999445e-01, 5.302876193624534574e-17),
(6.123724356957945814e-01, -4.999999999999999445e-01, -6.123724356957944703e-01),
(1.060575238724906915e-16, -4.999999999999999445e-01, -8.660254037844387076e-01),
(-6.123724356957944703e-01, -4.999999999999999445e-01, -6.123724356957946924e-01),
(-8.660254037844387076e-01, -4.999999999999999445e-01, -1.590862858087360249e-16),
(-6.123724356957946924e-01, -4.999999999999999445e-01, 6.123724356957944703e-01)
]

def generate_spherical_points(radius):
    coordinates = []
    for phi in range(0, 121, 30):
        rad_phi = np.radians(phi)
        for index in range(1, 9):
            theta = index * 45
            rad_theta = np.radians(theta)
            print(f'{phi}, {theta}')
            sin_phi = np.sin(rad_phi).tolist()
            cos_phi = np.cos(rad_phi).tolist()
            sin_theta = np.sin(rad_theta).tolist()
            cos_theta = np.cos(rad_theta).tolist()
            
            coordinates.append((radius*sin_phi*cos_theta, radius*sin_phi*sin_theta, radius*cos_phi+radius*0.5))
            if phi == 0:
                break

    return coordinates

def generate_hemisphere_views(input_dir, output_dir, transformed_coordinates):
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


    # n_frames = 36  # 旋轉拍攝 36 張圖（每 10° 拍一張）
    # radius = 1  # 攝影機距離模型的半徑

    # angles = np.linspace(0, 360, n_frames, endpoint=False)  # 0 到 360° 均分
    
    # for i, angle in enumerate(angles):
    #     plotter = pv.Plotter(off_screen=True)
    #     plotter.add_mesh(mesh, texture=texture)
    #     centroid = mesh.center  # 計算模型中心點

    #     x = centroid[0] + radius * np.cos(np.radians(angle))
    #     y = centroid[1] + 1
    #     z = centroid[2] + radius * np.sin(np.radians(angle))
    #     plotter.camera_position = [(x, y, z), (centroid[0], centroid[1], centroid[2]), (0, 1, 0)] # [cam pos, target pos, upward direction]
    #     # continue
    #     img_path = os.path.join(output_dir, f"view_{i:02d}.png")  # 儲存圖片
    #     print()
    #     plotter.screenshot(img_path)
    radius = 0.5
    transformed_coordinates = generate_spherical_points(radius)
    # print(transformed_coordinates)
    # return
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
        # plotter.camera_position = [(pos[0], pos[1], pos[2]), (centroid[0], centroid[1], centroid[2]), (0, 0, 1)] # [cam pos, target pos, upward direction]
        # plotter.camera_position = [(pos[0], pos[1], pos[2]), (0, 0, 0), (0, 0, 1)] # [cam pos, target pos, upward direction]
        # continue
        img_path = os.path.join(output_dir, f"view_{i:02d}.png")  # 儲存圖片
        plotter.screenshot(img_path)

    print(f"All screenshots are saved in {output_dir}")

def transform_coordinates(coords, scale_factor=3, y_offset=4.0):
    transformed_coords = coords * scale_factor  
    transformed_coords[:, 1] += y_offset 
    return transformed_coords

# 定義座標數據
# coordinates = np.array([
#     (-4.32978e-17, 1.0, 4.32978e-17),
#     (0.0, 0.8660254, 0.5),
#     (0.353553, 0.8660254, 0.353553),
#     (0.5, 0.8660254, 3.06162e-17),
#     (0.353553, 0.8660254, -0.353553),
#     (6.12323e-17, 0.8660254, -0.5),
#     (-0.353553, 0.8660254, -0.353553),
#     (-0.5, 0.8660254, -9.18485e-17),
#     (-0.353553, 0.8660254, 0.353553),
#     (0.0, 0.5, 0.8660254),
#     (0.612372, 0.5, 0.612372),
#     (0.866025, 0.5, 5.30288e-17),
#     (0.612372, 0.5, -0.612372),
#     (1.06058e-16, 0.5, -0.866025),
#     (-0.612372, 0.5, -0.612372),
#     (-0.866025, 0.5, -1.59086e-16),
#     (-0.612372, 0.5, 0.612372),
#     (0.0, 0.0, 1.0),
#     (0.707107, 0.0, 0.707107),
#     (1.0, 0.0, 6.12323e-17),
#     (0.707107, 0.0, -0.707107),
#     (1.22465e-16, 0.0, -1.0),
#     (-0.707107, 0.0, -0.707107),
#     (-1.0, 0.0, -1.83697e-16),
#     (-0.707107, 0.0, 0.707107),
#     (0.0, -0.5, 0.8660254),
#     (0.612372, -0.5, 0.612372),
#     (0.866025, -0.5, 5.30288e-17),
#     (0.612372, -0.5, -0.612372),
#     (1.06058e-16, -0.5, -0.866025),
#     (-0.612372, -0.5, -0.612372),
#     (-0.866025, -0.5, -1.59086e-16),
#     (-0.612372, -0.5, 0.612372),
# ])


# # 轉換座標
# transformed_coordinates = transform_coordinates(coordinates)

if __name__ == '__main__':
    
    output_dir = '../../data/house3k/HOUSE48/view/'
    input_dir = '../../data/house3k/HOUSE48/'


    # print(generate_spherical_points(33, 2))
    
    generate_hemisphere_views(input_dir, output_dir, coordinates_1)
    # mesh = pv.read('../../data/house3k/HOUSE48/BAT1_SETA_HOUSE48.obj')
    # print(mesh.center)
    # centroid = mesh.center
    # mesh.translate([-centroid[0], -centroid[1], -centroid[2]], inplace = True)
    # camera = pv.Camera()
    # camera.position = (2.0, 2.0, 2.0)
    # camera.focal_point = (0.0, 0.0, 0.0)
    # axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=1)
    # axes.origin = (0.0, 0.0, 0.0)   
    # p = pv.Plotter(off_screen=True)
    # p.add_text("Mesh", font_size=24)
    # p.add_actor(axes.actor)
    # p.camera = camera
    # mesh.rotate_vector([1,0,0], angle=90, inplace=True)
    # mesh.translate([centroid[0], centroid[2], centroid[1]], inplace = True)
    # p.add_mesh(mesh)
    # p.screenshot('output.png')
    # # p.show()
    # exit(0)

    # print(mesh.center)

    # up_vector = mesh.point_data['Normals'].mean(axis=0)
    # print(up_vector)

    # z_axis = np.array([0, 0, 1])
    # up_vector_normalized = up_vector / np.linalg.norm(up_vector)
    # cos_angle = np.dot(up_vector_normalized, z_axis)
    # angle = np.arccos(cos_angle)
    # rotation_axis = np.cross(up_vector_normalized, z_axis)
    # print(rotation_axis)
    
    # rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
    # mesh.rotate_vector(up_vector_normalized, np.degrees(angle), inplace=True)
    # print(mesh.point_data['Normals'].mean(axis=0))