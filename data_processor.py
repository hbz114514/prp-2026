import os
import glob
import re
import csv
import math
import numpy as np
import pandas as pd
import cv2

# --- 1. 参数与矩阵初始化 ---
FOCAL_LENGTH = 50.0 
PIXEL_SIZE = 5.0e-3
IMG_W, IMG_H = 1920, 1080
F_PX = FOCAL_LENGTH / PIXEL_SIZE
K = np.array([
    [F_PX, 0, IMG_W / 2.0],
    [0, F_PX, IMG_H / 2.0],
    [0, 0, 1]
])
K_INV = np.linalg.inv(K)

# --- 2. 核心算法函数 ---
def build_star_catalog(csv_path, mag_limit=6.0):
    df = pd.read_csv(csv_path)
    df = df[df['phot_g_mean_mag'] <= mag_limit].reset_index(drop=True)
    ra_rad = np.radians(df['ra'].values)
    dec_rad = np.radians(df['dec'].values)
    v_j2k = np.vstack((np.cos(dec_rad)*np.cos(ra_rad), 
                       np.cos(dec_rad)*np.sin(ra_rad), 
                       np.sin(dec_rad))).T
    return df, v_j2k

def extract_star_vectors(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stars = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            stars.append({"u": M["m10"]/M["m00"], "v": M["m01"]/M["m00"], "b": M["m00"]})
            
    stars = sorted(stars, key=lambda s: s["b"], reverse=True)[:3]
    if len(stars) < 3: return None, None

    cam_vectors = []
    for s in stars:
        vec = K_INV @ np.array([s["u"], s["v"], 1.0])
        cam_vectors.append(vec / np.linalg.norm(vec))
    return cam_vectors, stars

def match_triangle_optimized(cam_vectors, v_j2k_db, tolerance_deg=0.05):
    ang12 = math.degrees(math.acos(np.clip(np.dot(cam_vectors[0], cam_vectors[1]), -1, 1)))
    ang13 = math.degrees(math.acos(np.clip(np.dot(cam_vectors[0], cam_vectors[2]), -1, 1)))
    ang23 = math.degrees(math.acos(np.clip(np.dot(cam_vectors[1], cam_vectors[2]), -1, 1)))
    img_chirality = np.dot(cam_vectors[0], np.cross(cam_vectors[1], cam_vectors[2]))
    
    num_stars = len(v_j2k_db)
    best_match, min_error = None, float('inf')

    for i in range(num_stars):
        dots = np.dot(v_j2k_db, v_j2k_db[i])
        angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        j_cands = np.where(np.abs(angles - ang12) < tolerance_deg)[0]
        k_cands = np.where(np.abs(angles - ang13) < tolerance_deg)[0]

        for j in j_cands:
            if j == i: continue
            for k in k_cands:
                if k == i or k == j: continue
                ang_jk = math.degrees(math.acos(np.clip(np.dot(v_j2k_db[j], v_j2k_db[k]), -1.0, 1.0)))
                if abs(ang_jk - ang23) < tolerance_deg:
                    db_chirality = np.dot(v_j2k_db[i], np.cross(v_j2k_db[j], v_j2k_db[k]))
                    if img_chirality * db_chirality > 0: 
                        err = abs(angles[j]-ang12) + abs(angles[k]-ang13) + abs(ang_jk-ang23)
                        if err < min_error:
                            min_error = err
                            best_match = [i, j, k]
    return best_match

# --- 3. 自动化遍历与数据保存 ---
def process_dataset(dataset_dir, catalog_path, output_csv):
    print("正在初始化星表库...")
    df_db, v_j2k_db = build_star_catalog(catalog_path, mag_limit=6.0)
    
    image_files = glob.glob(os.path.join(dataset_dir, "*.jpg"))
    print(f"找到 {len(image_files)} 张测试图片，开始批量处理。大约需要 {len(image_files)} 分钟，请稍候...\n")

    # 定义正则提取文件名中的数据
    pattern = re.compile(r"sim_Az([\d\.]+)_Alt([\d\.]+)_F(\d+)_x([-\d\.]+)_y([-\d\.]+)_z([-\d\.]+)\.jpg")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        header = ['Image_Name', 'Az_enc', 'Alt_enc', 'GT_Cam_x', 'GT_Cam_y', 'GT_Cam_z']
        for i in range(1, 4):
            header.extend([f'Star{i}_u', f'Star{i}_v', 
                           f'Star{i}_Cam_x', f'Star{i}_Cam_y', f'Star{i}_Cam_z',
                           f'Star{i}_J2K_x', f'Star{i}_J2K_y', f'Star{i}_J2K_z'])
        writer.writerow(header)

        for idx, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            match = pattern.search(filename)
            if not match: continue
            
            az, alt, focal, gt_x, gt_y, gt_z = match.groups()
            print(f"[{idx+1}/{len(image_files)}] 正在处理: {filename}")
            
            cam_vecs, star_pixels = extract_star_vectors(img_path)
            if not cam_vecs:
                print("  -> 失败: 未提取到足够亮星")
                continue
                
            matched_ids = match_triangle_optimized(cam_vecs, v_j2k_db, tolerance_deg=0.03)
            
            if matched_ids:
                row = [filename, az, alt, gt_x, gt_y, gt_z]
                for k in range(3):
                    row.extend([
                        f"{star_pixels[k]['u']:.2f}", f"{star_pixels[k]['v']:.2f}",
                        f"{cam_vecs[k][0]:.6f}", f"{cam_vecs[k][1]:.6f}", f"{cam_vecs[k][2]:.6f}",
                        f"{v_j2k_db[matched_ids[k]][0]:.6f}", f"{v_j2k_db[matched_ids[k]][1]:.6f}", f"{v_j2k_db[matched_ids[k]][2]:.6f}"
                    ])
                writer.writerow(row)
                print("  -> 成功匹配并写入数据")
            else:
                print("  -> 失败: 匹配未命中")

    print(f"\n全部处理完成！数据已保存至 {output_csv}")

if __name__ == "__main__":
    process_dataset("star_tracker_dataset", "gaia_northern_12mag.csv", "matched_results.csv")