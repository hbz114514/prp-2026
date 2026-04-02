import math
import random
import numpy as np
import pandas as pd
import cv2
import os
import json
from scipy.spatial import cKDTree
from datetime import datetime, timezone

# =======================================================
# 1. 初始化设置与参数
# =======================================================
random.seed(42)
output_dir = "star_tracker_dataset"
os.makedirs(output_dir, exist_ok=True)

# 机械误差参数
DELTA = random.uniform(-0.01, 0.01)
PHI_X, PHI_Y, PHI_Z = random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02)
THETA_NP = random.uniform(-0.005, 0.005)
EPS_X, EPS_Y, EPS_Z = random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)
LAT, LON = 31.22, 121.48

# 相机参数 (使用 50mm 广角保证星星数量足够)
focal_length = 50.0 
pixel_size = 5.0e-3
img_width, img_height = 1920, 1080 
f_px = focal_length / pixel_size
K = np.array([[f_px, 0, img_width / 2.0], [0, f_px, img_height / 2.0], [0, 0, 1]])

# =======================================================
# 2. 载入星表并进行【硬件极限定制】
# =======================================================
print("正在载入全天区星表...")
df = pd.read_csv('gaia_northern_12mag.csv')

# ！！！核心修改：直接在源头截断，模拟星敏感器 6.5 等感光极限 ！！！
df_filtered = df[df['phot_g_mean_mag'] <= 6.5]
print(f"原数据库恒星数: {len(df)}，过滤后 (<= 6.5等) 剩余极亮恒星数: {len(df_filtered)}")

mags_all = df_filtered['phot_g_mean_mag'].values
ra_rad_all = np.radians(df_filtered['ra'].values)
dec_rad_all = np.radians(df_filtered['dec'].values)
stars_3d_all = np.vstack((np.cos(dec_rad_all)*np.cos(ra_rad_all), np.cos(dec_rad_all)*np.sin(ra_rad_all), np.sin(dec_rad_all))).T 
tree = cKDTree(stars_3d_all)

# 计算 kd-tree 搜索半径
actual_fov = np.degrees(2 * np.arctan((img_width * pixel_size) / (2 * focal_length)))
search_radius = 2.0 * np.sin(np.radians(actual_fov * 1.2) / 2.0)

# =======================================================
# 3. 基础物理与变换矩阵 (修复版)
# =======================================================
def rot_x(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]]) # 已修复
def rot_z(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def euler_zxy(z, x, y): return rot_x(x) @ rot_y(y) @ rot_z(z)

def julian_day(date):
    y, m, d = date.year, date.month, date.day + date.hour/24.0 + date.minute/1440.0 + date.second/86400.0
    if m <= 2: y -= 1; m += 12
    A = y // 100
    return math.floor(365.25*(y + 4716)) + math.floor(30.6001*(m + 1)) + d + (2 - A + A // 4) - 1524.5

def gmst_rad(jd):
    T = (jd - 2451545.0) / 36525.0
    return ((24110.54841 + 8640184.812866 * T + 0.093104 * T*T - 0.0000062 * T*T*T) % 86400) * 2 * math.pi / 86400

def draw_gaussian_spot(img, cx, cy, brightness, sigma=1.5):
    h, w = img.shape
    s = int(sigma * 6) | 1
    x, y = np.arange(0, s, 1, float), np.arange(0, s, 1, float)[:, np.newaxis]
    c = s // 2
    dx, dy = cx - int(cx), cy - int(cy)
    g = brightness * np.exp(-((x - c - dx)**2 + (y - c - dy)**2) / (2 * sigma**2))
    ix, iy = int(cx), int(cy)
    x1, x2 = max(0, ix - c), min(w, ix - c + s)
    y1, y2 = max(0, iy - c), min(h, iy - c + s)
    gx1, gx2 = c - (ix - x1), c + (x2 - ix)
    gy1, gy2 = c - (iy - y1), c + (y2 - iy)
    if x1 < x2 and y1 < y2: img[y1:y2, x1:x2] += g[gy1:gy2, gx1:gx2]

# =======================================================
# =======================================================
# 4. 自动化批量生成流程 (工业标准正交扫描轨迹)
# =======================================================
# 【极其重要】强制锁死仿真宇宙的时间，确保与求解器严格同步！
utc_now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

trajectories = []
# 轨迹 1：方位大圆扫描 (固定仰角 45°，扫描一圈，12 张)
for az in range(0, 360, 30):
    trajectories.append((float(az), 45.0))

# 轨迹 2：子午线扫描 (固定方位 180°，从地平线扫到天顶，8 张)
for alt in range(15, 90, 10):
    trajectories.append((180.0, float(alt)))

num_images = len(trajectories)
success_count = 0

print(f"\n开始离线生成 {num_images} 张按正交轨迹规划的测试星图...")

for i, (az_deg, alt_deg) in enumerate(trajectories):
    az_rad, alt_rad = math.radians(az_deg), math.radians(alt_deg)
    
    # --- 下面的矩阵计算和图片渲染保持原样 ---
    R_J2K_ENU = rot_z(-(gmst_rad(julian_day(utc_now)) + math.radians(LON))) @ rot_y(math.radians(LAT) - math.pi/2)
    R_ENU_MNT = rot_x(EPS_X) @ rot_y(EPS_Y) @ rot_z(EPS_Z)
    R_MNT_GIM = rot_z(-az_rad) @ rot_z(THETA_NP) @ rot_x(-alt_rad) @ rot_z(-THETA_NP)
    R_total_J2K_OBS = R_J2K_ENU @ R_ENU_MNT @ R_MNT_GIM @ euler_zxy(PHI_Z, PHI_X, PHI_Y) @ rot_x(DELTA)
    
    v_j2k = R_total_J2K_OBS @ np.array([0, 0, 1])
    v_j2k /= np.linalg.norm(v_j2k)
    R_OBS_J2K = R_total_J2K_OBS.T 
    
    indices = tree.query_ball_point(v_j2k, search_radius)
    star_map = np.zeros((img_height, img_width), dtype=np.float32)
    
    if indices:
        local_stars_j2k = stars_3d_all[indices]
        local_mags = mags_all[indices]
        stars_obs = (R_OBS_J2K @ local_stars_j2k.T).T
        front_mask = stars_obs[:, 2] > 0
        stars_obs, local_mags = stars_obs[front_mask], local_mags[front_mask]
        coords_homo = (K @ (stars_obs / stars_obs[:, 2:3]).T).T
        u, v = coords_homo[:, 0], coords_homo[:, 1]
        
        for j in range(len(u)):
            mag = local_mags[j]
            brightness = 150.0 * (2.512 ** (6.5 - mag))
            draw_gaussian_spot(star_map, u[j], v[j], brightness, sigma=1.2)
            
    star_map += np.random.normal(3, 2, (img_height, img_width))
    final_img = np.clip(star_map, 0, 255).astype(np.uint8)
    
    filename = f"sim_Az{az_deg:.2f}_Alt{alt_deg:.2f}_F{focal_length:.0f}_x{v_j2k[0]:.4f}_y{v_j2k[1]:.4f}_z{v_j2k[2]:.4f}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, final_img)
    
    print(f"[{i+1}/{num_images}] 生成成功: {filename} (视野亮星数: {len(indices)} 颗)")
    success_count += 1

# === 新增：直接在终端打印这 8 个上帝视角的“标准答案” ===
print("\n" + "="*50)
print("【标准答案】本次仿真注入的 8 项物理真实误差 (真值)：")
print(f"  DELTA (相机视轴偏差):    {DELTA:.6f}")
print(f"  PHI_X (安装偏航角):      {PHI_X:.6f}")
print(f"  PHI_Y (安装俯仰角):      {PHI_Y:.6f}")
print(f"  PHI_Z (安装横滚角):      {PHI_Z:.6f}")
print(f"  THETA_NP (极轴不垂直度): {THETA_NP:.6f}")
print(f"  EPS_X (底座调平误差 X):  {EPS_X:.6f}")
print(f"  EPS_Y (底座调平误差 Y):  {EPS_Y:.6f}")
print(f"  EPS_Z (方位零点误差):    {EPS_Z:.6f}")
print("="*50)

# === 新增：直接在终端打印这 8 个上帝视角的“标准答案” ===
print("\n" + "="*50)
print("【标准答案】本次仿真注入的 8 项物理真实误差 (真值)：")
print(f"  DELTA (相机视轴偏差):    {DELTA:.6f}")
print(f"  PHI_X (安装偏航角):      {PHI_X:.6f}")
print(f"  PHI_Y (安装俯仰角):      {PHI_Y:.6f}")
print(f"  PHI_Z (安装横滚角):      {PHI_Z:.6f}")
print(f"  THETA_NP (极轴不垂直度): {THETA_NP:.6f}")
print(f"  EPS_X (底座调平误差 X):  {EPS_X:.6f}")
print(f"  EPS_Y (底座调平误差 Y):  {EPS_Y:.6f}")
print(f"  EPS_Z (方位零点误差):    {EPS_Z:.6f}")
print("="*50)

print(f"\n数据集生成完毕！请在 {output_dir} 文件夹中查看。")

true_params_dict = {
    "DELTA": DELTA, "PHI_X": PHI_X, "PHI_Y": PHI_Y, "PHI_Z": PHI_Z,
    "THETA_NP": THETA_NP, "EPS_X": EPS_X, "EPS_Y": EPS_Y, "EPS_Z": EPS_Z
}

with open("true_params.json", "w") as f:
    json.dump(true_params_dict, f, indent=4)

print("\n真值已自动保存至 true_params.json，供求解器调用比对。")