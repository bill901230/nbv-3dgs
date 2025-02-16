PARENT_DIR="../../data/house3k"

# 遍歷 PARENT_DIR 中的所有子資料夾
for TARGET_DIR in "$PARENT_DIR"/*/; do
    # 確保 TARGET_DIR 真的存在且是資料夾
    if [ -d "$TARGET_DIR" ]; then
        echo "正在刪除 $TARGET_DIR 內的檔案..."
        
        rm -rf "$TARGET_DIR/exr" "$TARGET_DIR/pose"

        find "$TARGET_DIR" -maxdepth 1 -type f -regex ".*/[0-9]+\.pcd" -delete
        find "$TARGET_DIR" -maxdepth 1 -type f -regex ".*/[0-9]+\.png" -delete
    fi
done

rm -f "$TARGET_DIR/intrinsics.txt"

echo "所有資料夾的刪除作業完成！"
