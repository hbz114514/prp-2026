import math
import random
import numpy as np
import pandas as pd
import cv2
import base64
from scipy.spatial import cKDTree
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone, timedelta

app = Flask(__name__)
CORS(app)

random.seed(42)

# ==================== 1. 朋友的误差参数 ====================
DELTA = random.uniform(-0.01, 0.01)
PHI_X, PHI_Y, PHI_Z = random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02)
THETA_NP = random.uniform(-0.005, 0.005)
EPS_X, EPS_Y, EPS_Z = random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)
LAT, LON = 31.22, 121.48

# ==================== 2. 星图引擎初始化 ====================
print("正在载入全天区星表并构建 KD-Tree...")
df = pd.read_csv('gaia_northern_12mag.csv')
print("Loaded", len(df), "stars")
mags_all = df['phot_g_mean_mag'].values
ra_rad_all = np.radians(df['ra'].values)
dec_rad_all = np.radians(df['dec'].values)
stars_3d_all = np.vstack((np.cos(dec_rad_all)*np.cos(ra_rad_all), np.cos(dec_rad_all)*np.sin(ra_rad_all), np.sin(dec_rad_all))).T 
tree = cKDTree(stars_3d_all)

# 筛选北半球可见的12等以上亮星
bright_stars_df = df[(df['dec'] >= -30) & (df['phot_g_mean_mag'] <= 13)].sort_values('phot_g_mean_mag').head(20)
print("Bright stars:", len(bright_stars_df))

# 固定相机参数
pixel_size = 5.0e-3
img_width, img_height = 1920, 1080 
print("后端引擎就绪！")

# ==================== 3. 朋友的矩阵计算库 ====================
def rot_x(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
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

def compute_total_rotation(az_deg, alt_deg, utc_time):
    """计算完整的旋转矩阵 (而不是直接求出光轴向量)，星图需要这个大矩阵"""
    az_rad, alt_rad = math.radians(az_deg), math.radians(alt_deg)
    
    # 按照朋友的链条反向相乘得到 R_J2K_OBS (从相机到天球)
    R_J2K_ENU = rot_z(-(gmst_rad(julian_day(utc_time)) + math.radians(LON))) @ rot_y(math.radians(LAT) - math.pi/2)
    R_ENU_MNT = rot_x(EPS_X) @ rot_y(EPS_Y) @ rot_z(EPS_Z)
    R_MNT_GIM = rot_z(-az_rad) @ rot_z(THETA_NP) @ rot_x(-alt_rad) @ rot_z(-THETA_NP)
    
    R_total_J2K_OBS = R_J2K_ENU @ R_ENU_MNT @ R_MNT_GIM @ euler_zxy(PHI_Z, PHI_X, PHI_Y) @ rot_x(DELTA)
    return R_total_J2K_OBS

# ==================== 4. 你的星图渲染库 ====================
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

def draw_grid_and_annotations(img, R_OBS_J2K, K, focal_len, fov_deg):
    h, w = img.shape[:2]
    # 从旋转矩阵反推出相机中心的 RA 和 Dec，用于网格基准
    z_c = R_OBS_J2K[2, :] # R_OBS_J2K 的第三行实际上是 J2000 下的 Z 轴方向
    dec0 = np.degrees(np.arcsin(z_c[2]))
    ra0 = np.degrees(np.arctan2(z_c[1], z_c[0])) % 360
    
    gc, tc = (80, 80, 80), (200, 200, 200)
    for d in range(int(dec0) - 10, int(dec0) + 11):
        if -90 <= d <= 90:
            pts = [[int(p[0]), int(p[1])] for r in np.linspace(ra0-15, ra0+15, 100) 
                   if (v:=(R_OBS_J2K @ np.array([np.cos(np.radians(d))*np.cos(np.radians(r)), np.cos(np.radians(d))*np.sin(np.radians(r)), np.sin(np.radians(d))])))[2] > 0 
                   and 0 <= (p:=(K @ (v/v[2])))[0] < w and 0 <= p[1] < h]
            if pts: cv2.polylines(img, [np.array(pts)], False, gc, 1, cv2.LINE_AA); cv2.putText(img, f"D:{d}", tuple(pts[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, gc, 1)
            
    for r in range(int(ra0) - 15, int(ra0) + 16, 2):
        pts = [[int(p[0]), int(p[1])] for d in np.linspace(dec0-10, dec0+10, 50) 
               if (v:=(R_OBS_J2K @ np.array([np.cos(np.radians(d))*np.cos(np.radians(r)), np.cos(np.radians(d))*np.sin(np.radians(r)), np.sin(np.radians(d))])))[2] > 0 
               and 0 <= (p:=(K @ (v/v[2])))[0] < w and 0 <= p[1] < h]
        if pts: cv2.polylines(img, [np.array(pts)], False, gc, 1, cv2.LINE_AA); cv2.putText(img, f"R:{r}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, gc, 1)

    for i, line in enumerate([f"FOCAL: {focal_len}mm", f"FOV: {fov_deg:.2f} DEG"]):
        cv2.putText(img, line, (20, 40 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 1, cv2.LINE_AA)

# ==================== 5. 融合 API 接口 ====================
@app.route('/stars', methods=['GET'])
def get_stars():
    # 如果筛选失败，使用hardcode示例星星
    if bright_stars_df.empty:
        stars_list = [
            {'name': 'Sirius (101.29°, -16.72°)', 'ra': 101.29, 'dec': -16.72},
            {'name': 'Vega (279.23°, 38.78°)', 'ra': 279.23, 'dec': 38.78},
            {'name': 'Arcturus (213.92°, 19.18°)', 'ra': 213.92, 'dec': 19.18},
            {'name': 'Rigel (78.63°, -8.20°)', 'ra': 78.63, 'dec': -8.20},
            {'name': 'Betelgeuse (88.79°, 7.41°)', 'ra': 88.79, 'dec': 7.41},
        ]
    else:
        stars_list = [{'name': f"Star {int(row['source_id'])} ({row['ra']:.2f}°, {row['dec']:.2f}°)", 'ra': row['ra'], 'dec': row['dec']} for _, row in bright_stars_df.iterrows()]
    print("Stars loaded:", len(stars_list))
    return jsonify(stars_list)

@app.route('/inverse', methods=['GET'])
def inverse():
    ra = float(request.args.get('ra'))
    dec = float(request.args.get('dec'))
    time_offset = float(request.args.get('time_offset', 0))
    utc_time = datetime.now(timezone.utc) + timedelta(hours=time_offset)
    v_j2k = np.array([math.cos(math.radians(dec)) * math.cos(math.radians(ra)),
                      math.cos(math.radians(dec)) * math.sin(math.radians(ra)),
                      math.sin(math.radians(dec))])
    gmst = gmst_rad(julian_day(utc_time))
    R_J2K_ENU = rot_z(-(gmst + math.radians(LON))) @ rot_y(math.radians(LAT) - math.pi/2)
    v_enu = R_J2K_ENU.T @ v_j2k
    az_rad = math.atan2(v_enu[1], v_enu[0])
    alt_rad = math.asin(v_enu[2])
    az_deg = math.degrees(az_rad)
    alt_deg = math.degrees(alt_rad)
    az_slider = 90 - az_deg
    alt_slider = 90 - alt_deg
    return jsonify({"az": az_slider, "alt": alt_slider})

@app.route('/pointing', methods=['GET'])
def pointing():
    try:
        # 获取前端发来的 机械指令角度 和 焦距
        az = float(request.args.get('az', 0))
        alt = float(request.args.get('alt', 0)) - 180.0
        focal_length = float(request.args.get('focal', 150.0))
        time_offset = float(request.args.get('time_offset', 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid params"}), 400

    utc_now = datetime.now(timezone.utc)
    utc_time = utc_now + timedelta(hours=time_offset)
    
    # --- 阶段 1: 朋友的物理位姿计算 ---
    R_J2K_OBS = compute_total_rotation(az, alt, utc_time)  
    v_j2k = R_J2K_OBS @ np.array([0, 0, 1]) # 相机Z轴在天球下的向量
    v_j2k /= np.linalg.norm(v_j2k)
    
    # 计算GMST用于可视化，根据时间偏移
    gmst = gmst_rad(julian_day(utc_time))
    
    # --- 阶段 2: 你的星图渲染引擎 ---
    # 构建外参，注意：星图需要的是 R_OBS_J2K (天球到相机)
    R_OBS_J2K = R_J2K_OBS.T 
    
    f_px = focal_length / pixel_size
    K = np.array([[f_px, 0, img_width / 2.0], [0, f_px, img_height / 2.0], [0, 0, 1]])
    actual_fov = np.degrees(2 * np.arctan((img_width * pixel_size) / (2 * focal_length)))
    search_radius = 2.0 * np.sin(np.radians(actual_fov * 1.2) / 2.0)

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
        
        for i in range(len(u)):
            mag = local_mags[i]
            brightness = 20.0 * (2.512 ** (12.0 - mag))
            draw_gaussian_spot(star_map, u[i], v[i], brightness, sigma=1.0 + max(0, (12.0 - mag) * 0.3))

    star_map += np.random.normal(12, 6, (img_height, img_width))
    final_img_color = cv2.cvtColor(np.clip(star_map, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    draw_grid_and_annotations(final_img_color, R_OBS_J2K, K, focal_length, actual_fov)

    # 压缩成 Base64
    _, buffer = cv2.imencode('.jpg', final_img_color, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 将 向量坐标 和 图片 一起发给前端
    return jsonify({
        "status": "ok",
        "x": v_j2k[0], "y": v_j2k[1], "z": v_j2k[2],
        "gmst": gmst,
        "image": 'data:image/jpeg;base64,' + img_base64,
        "star_count": len(indices)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)