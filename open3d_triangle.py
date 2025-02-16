import open3d as o3d

# 讀取 OBJ 文件
mesh = o3d.io.read_triangle_mesh("data/house3k/5/model.obj")

# 輸出三角形面數量
print(f"Number of triangles: {len(mesh.triangles)}")

# 如果沒有三角形，嘗試顯示頂點和面
if len(mesh.triangles) == 0:
    print("No triangles found. Checking vertices and faces:")
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Sample vertices: {mesh.vertices[:5]}")  # 顯示部分頂點
