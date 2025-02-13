import bpy

def import_fbx(file_path):
    # 导入 .fbx 文件
    bpy.ops.import_scene.fbx(filepath=file_path)

def extract_mesh_and_materials():
    # 获取所有选中的对象
    objects = bpy.context.selected_objects
    
    for obj in objects:
        if obj.type == 'MESH':  # 只处理网格类型的对象
            print(f"Object: {obj.name}")
            
            # 提取网格数据
            mesh = obj.data
            print("Vertices:")
            for vertex in mesh.vertices:
                print(f"  Vertex: {vertex.co}")
            
            # 提取材质
            if obj.material_slots:
                print("Materials:")
                for slot in obj.material_slots:
                    print(f"  Material: {slot.name}")
            else:
                print("No materials assigned.")

if __name__ == '__main__':
    # 导入 FBX 文件
    fbx_file_path = "path_to_your_file.fbx"  # 替换成你的 .fbx 文件路径
    import_fbx(fbx_file_path)
    
    # 提取网格和材质
    extract_mesh_and_materials()
