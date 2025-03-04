import bpy
import os
import argparse


def transform(fbx_path, output_path):
    # 清除場景中的所有物件
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 匯入 FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # 確保匯出目錄存在
    os.makedirs(output_path, exist_ok=True)

    # 匯出為 OBJ (使用新的 API)
    bpy.ops.wm.obj_export(
        filepath=os.path.join(output_path, os.path.splitext(os.path.basename(fbx_path))[0] + '.obj'),
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
            img_path = os.path.join(output_path, image.name + image_extension)
            image.save_render(img_path)

    print(f"已成功轉換: {fbx_path} -> {output_path}")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fbx_path', type=str, required=True)
    # parser.add_argument('--output_path', type=str, required=True)
    # args = parser.parse_args()

    # # 設定 FBX 和 OBJ 檔案路徑
    fbx_path = "/home_nfs/weilingchi/nbv-3dgs/data/house3k/BAT1_SETA_HOUSE8.fbx"
    output_path = '/home_nfs/weilingchi/nbv-3dgs/data/house3k/BAT1_SETA_HOUSE8'
    transform(fbx_path, output_path)