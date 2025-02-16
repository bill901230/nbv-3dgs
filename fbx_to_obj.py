import bpy
import os

for i in range(1,11):

    # 設定 FBX 和 OBJ 檔案路徑
    fbx_path = f"data/original_house3k/BAT1_SETA_HOUSE{i}.fbx"
    obj_output_path = f"data/house3k/{i}/model.obj"
    print("轉換{fbx_path}...")

    # 清除場景中的所有物件
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 匯入 FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # 確保匯出目錄存在
    os.makedirs(os.path.dirname(obj_output_path), exist_ok=True)

    # 匯出為 OBJ (使用新的 API)
    bpy.ops.wm.obj_export(
        filepath=obj_output_path,
        export_materials=True,
        path_mode="COPY"
        
    )
    print(f"已成功轉換: {fbx_path} -> {obj_output_path}")
print("所有 FBX 轉換完成")
