import pyvista as pv
# pv.start_xvfb()
# import open3d as o3d
import os
import numpy as np
    
def generate_spherical_viewpoints(radius, delta_phi, delta_theta):
    coordinates = []
    for phi in range(0, 121, delta_phi):
        rad_phi = np.radians(phi)
        for index in range(1, (int(360/delta_theta) + 1)):
            theta = index * delta_theta
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
def generate_hemisphere_views(input_dir, output_dir, radius = 0.5, delta_phi=30, delta_theta=45):
    """
    Generate specified number of screenshot on the object.

    
    - input_dir: A directory contains .obj .mtl .png file 

    - output_dir: views_num of screenshot will be stored there.

    - views_num: the number of view distribute this hemisphere.

    """

    # make sure output folder is created
    os.makedirs(output_dir, exist_ok=True)

    # load model object
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


    transformed_coordinates = generate_spherical_viewpoints(radius, delta_phi, delta_theta)

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
        # plotter.add_actor(axes.actor)


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
        img_path = os.path.join(output_dir, f"view_{i:02d}.jpg")  # 儲存圖片
        plotter.screenshot(img_path)

    print(f"All screenshots are saved in {output_dir}")

if __name__ == '__main__':

    input_dir = '/home_nfs/weilingchi/nbv-3dgs/data/house3k/HOUSE48/'
    output_dir = '/home_nfs/weilingchi/HOUSE48_4times_views/'

    radius = 1
    delta_phi = 15 # 緯度
    delta_theta = 22.5 # 經度
    generate_hemisphere_views(input_dir, output_dir, radius, delta_phi, delta_theta)