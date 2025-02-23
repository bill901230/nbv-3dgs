import bpy
import mathutils
import numpy as np
import os
import time

def setup_blender(width, height, focal_length):
    # 設定相機
    if 'Camera' in bpy.data.objects:
        camera = bpy.data.objects['Camera']
    else:
        camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.scene.collection.objects.link(camera)

    camera.data.lens = focal_length  # 設定焦距
    camera.data.clip_end = 10.0
    camera.data.clip_start = 0.1

    # 設定渲染參數
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '16'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    scene.camera = camera
    # 啟用深度通道
    bpy.context.view_layer.use_pass_z = True 

    # 啟用節點
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # 設定渲染圖層與輸出節點
    rl = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.format.file_format = 'OPEN_EXR'
    tree.links.new(rl.outputs['Depth'], output.inputs[0])

    # 刪除預設立方體
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    return scene, camera, output

def import_model(model_obj_path):
    """ 在 Blender 內部導入 OBJ 模型 """
    bpy.ops.wm.obj_import(filepath=model_obj_path)

def rotate_model():
    """ 旋轉模型，以符合 ShapeNet 座標 (Y-up)"""
    bpy.ops.transform.rotate(value=-np.pi / 2, orient_axis='X')
    bpy.context.view_layer.update()  # 確保 Blender 更新視圖

def render_depth(scene, camera, output, model_name, viewspace, output_path):
    """ 渲染 EXR 深度圖並儲存 """
    exr_dir = os.path.join(output_path, model_name, 'exr')
    pose_dir = os.path.join(output_path, model_name, 'pose')
    os.makedirs(exr_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    for i, view in enumerate(viewspace):
        print(f"渲染視角 {i + 1}/{len(viewspace)}")

        # 設定相機位置與旋轉
        cam_pose = mathutils.Vector((view[0], view[1], view[2]))
        center_pose = mathutils.Vector((0, 0, 0))
        direct = center_pose - cam_pose
        rot_quat = direct.to_track_quat('-Z', 'Y')

        camera.rotation_euler = rot_quat.to_euler()
        camera.location = cam_pose
        bpy.context.view_layer.update()

        # 設定輸出檔案名稱
        output.file_slots[0].path = os.path.join(exr_dir, f"{i}.exr")
        bpy.ops.render.render(write_still=True)

        # 儲存相機姿態
        np.savetxt(os.path.join(pose_dir, f'{i}.txt'), camera.matrix_world, '%f')

def main():
    
    bpy.ops.wm.console_toggle()
    # 手動指定 `data_path` 和 `output_path`
    data_path = bpy.path.abspath(r"C:\Users\User\Desktop\house3k_test")  # 內部相對路徑
    output_path = bpy.path.abspath(r"C:\Users\User\Desktop\house3k_test")

    if not os.path.exists(data_path):
        print("[錯誤] 資料夾不存在:", data_path)
        return

    # 相機內參
    width, height, focal = 640, 480, 476
    scene, camera, output = setup_blender(width, height, focal)

    # 讀取視角 (viewspace)
    view_space_path = bpy.path.abspath(r"C:\Users\User\Desktop\house3k_test/viewspace_shapenet_33.txt")
    viewspace = np.loadtxt(view_space_path)

    # 讀取模型列表
    model_list = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    print(f"開始渲染 {len(model_list)} 個模型")

    for model_name in model_list:
        print(f"處理模型: {model_name}")

        model_obj_path = os.path.join(data_path, model_name, "model.obj")
        if not os.path.exists(model_obj_path):
            print(f"[警告] 找不到 OBJ: {model_obj_path}")
            continue

        # 導入模型
        import_model(model_obj_path)

        # 旋轉模型使其符合 ShapeNet
        rotate_model()

        # 渲染深度圖
        render_depth(scene, camera, output, model_name, viewspace, output_path)

        # 清理場景，刪除模型
        bpy.ops.object.delete()
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)
        for mat in bpy.data.materials:
            mat.user_clear()
            bpy.data.materials.remove(mat)

    print("🎉 渲染完成！")

if __name__ == '__main__':
    main()
