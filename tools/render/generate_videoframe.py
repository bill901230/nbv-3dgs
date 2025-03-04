import pyvista as pv
import os
import numpy as np


def generate_videoframe(input_dir, output_dir, n_frames = 143, start_pose = (1.0, 0.0, 0.0), end_pose = (0.0, 1.0, 0.0)):

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
    
    unit_move = (end_pose[0] / n_frames, end_pose[1] / n_frames, end_pose[2] / n_frames)
    for i in range(n_frames):
        camera = pv.Camera()
        camera.position = (start_pose[0] + i * unit_move[0], start_pose[1] + i * unit_move[1], start_pose[2] + i * unit_move[2])
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
        img_path = os.path.join(output_dir, f"view_{i:02d}.png")  # 儲存圖片
        plotter.screenshot(img_path)

    print(f"All screenshots are saved in {output_dir}")

if __name__ == '__main__':
    
    output_dir = '../../data/house3k/HOUSE48/view/'
    input_dir = '../../data/house3k/HOUSE48/'

    generate_videoframe(input_dir, output_dir)