import bpy
import os

# 設定 FBX 和 OBJ 檔案路徑
fbx_path = "../../data/house3k/BAT1_SETA_HOUSE48.fbx"
obj_output_path = "../../data/house3k/BAT1_SETA_HOUSE48.obj"
texture_output_dir = '../../data/house3k/'

# 清除場景中的所有物件
bpy.ops.wm.read_factory_settings(use_empty=True)

# 匯入 FBX
bpy.ops.import_scene.fbx(filepath=fbx_path)

# 確保匯出目錄存在
os.makedirs(os.path.dirname(obj_output_path), exist_ok=True)

# 匯出為 OBJ (使用新的 API)
bpy.ops.wm.obj_export(
    filepath=obj_output_path,
    path_mode='COPY')

for image in bpy.data.images:
    if image.packed_file:
        # 解包圖片
        image.unpack(method='USE_ORIGINAL')
        image_extension = ".jpg"  # 預設為 .jpg，若是其他格式需要調整
        
        # 根據圖片的檔案格式來確定副檔名
        if image.file_format == 'JPEG':
            image_extension = ".jpg"
        elif image.file_format == 'PNG':
            image_extension = ".png"
        # 儲存解包的圖片到指定資料夾
        # img_path = os.path.join(texture_output_dir, image.name)
        img_path = os.path.join(texture_output_dir, image.name + image_extension)
        image.save_render(img_path)

print(f"已成功轉換: {fbx_path} -> {obj_output_path}")