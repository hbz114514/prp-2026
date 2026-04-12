import math
import random
import numpy as np
import pandas as pd
import cv2
import base64
import json
import time
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from datetime import datetime, timezone, timedelta

# 引入 skyfield 用于卫星位置计算
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time

app = Flask(__name__)
CORS(app)

random.seed(42)

# 朋友的误差参数
DELTA = 0.0
PHI_X, PHI_Y, PHI_Z = 0.0, 0.0, 0.0
THETA_NP = 0.0
EPS_X, EPS_Y, EPS_Z = 0.0, 0.0, 0.0

# 默认地理位置 (上海)
LAT, LON = 31.22, 121.48

# 星图引擎初始化
print("正在载入全天区星表并构建 KD-Tree...")
df = pd.read_csv('gaia_northern_12mag.csv')
print("Loaded", len(df), "stars")
mags_all = df['phot_g_mean_mag'].values
ra_rad_all = np.radians(df['ra'].values)
dec_rad_all = np.radians(df['dec'].values)
stars_3d_all = np.vstack((np.cos(dec_rad_all)*np.cos(ra_rad_all),
                          np.cos(dec_rad_all)*np.sin(ra_rad_all),
                          np.sin(dec_rad_all))).T 
tree = cKDTree(stars_3d_all)

# 筛选北半球可见的12等以上亮星
bright_stars_df = df[(df['dec'] >= -30) & (df['phot_g_mean_mag'] <= 13)].sort_values('phot_g_mean_mag').head(20)
print("Bright stars:", len(bright_stars_df))

# 固定相机参数
pixel_size = 5.0e-3
img_width, img_height = 1920, 1080 
print("后端引擎就绪！")

# 矩阵计算库
def rot_x(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def euler_zxy(z, x, y): return rot_x(x) @ rot_y(y) @ rot_z(z)

def julian_day(date):
    y, m = date.year, date.month
    d = date.day + date.hour/24.0 + date.minute/1440.0 + date.second/86400.0 + date.microsecond/86400000000.0
    if m <= 2: y -= 1; m += 12
    A = y // 100
    return math.floor(365.25*(y + 4716)) + math.floor(30.6001*(m + 1)) + d + (2 - A + A // 4) - 1524.5

def gmst_rad(jd):
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
    return math.radians(gmst_deg % 360)

def compute_total_rotation(az_deg, alt_deg, utc_time, params=None):
    if params is None:
        delta, phix, phiy, phiz = DELTA, PHI_X, PHI_Y, PHI_Z
        theta, epsx, epsy, epsz = THETA_NP, EPS_X, EPS_Y, EPS_Z
    else:
        delta, phix, phiy, phiz, theta, epsx, epsy, epsz = params

    az_rad, alt_rad = math.radians(az_deg), math.radians(alt_deg)
    gmst = gmst_rad(julian_day(utc_time))
    
    R_J2K_ENU = rot_z(gmst + math.radians(LON)) @ rot_y(math.pi/2 - math.radians(LAT)) @ rot_z(math.pi/2)
    R_ENU_MNT = rot_x(epsx) @ rot_y(epsy) @ rot_z(epsz)
    R_MNT_GIM = rot_z(-az_rad) @ rot_z(theta) @ rot_x(-alt_rad) @ rot_z(-theta)
    
    R_total_J2K_OBS = R_J2K_ENU @ R_ENU_MNT @ R_MNT_GIM @ euler_zxy(phiz, phix, phiy) @ rot_x(delta)
    return R_total_J2K_OBS

# 星图渲染库
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
    z_c = R_OBS_J2K[2, :]
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

# 卫星相关功能
# 预置多颗典型卫星的 TLE（示例数据，实际使用时请从 CelesTrak 更新）
SATELLITE_TLE = {
    "ISS (ZARYA)": [
        "1 25544U 98067A   25095.50000000  .00016717  00000-0  30150-3 0  9996",
        "2 25544  51.6439  45.3871 0006282  96.0438  67.9698 15.49133105496828"
    ],
    "HUBBLE SPACE TELESCOPE": [
        "1 20580U 90037B   25097.90524753  .00000255  00000-0  19007-4 0  9998",
        "2 20580  28.4702 270.3497 0002716 320.1489  39.8604 15.09991837548910"
    ],
    "CHINA SPACE STATION (TIANHE)": [
        "1 48274U 21035A   25095.62500000  .00012345  00000-0  45678-3 0  9991",
        "2 48274  41.4701 123.4567 0012345 234.5678 125.6789 15.49123456123456"
    ],
    "BEIDOU-3 M28": [
        "1 58654U 23001A   25095.50000000  .00004567  00000-0  23456-3 0  9992",
        "2 58654  55.1234 300.4567 0012345 123.4567 236.7890 15.12345678901234"
    ],
    "STARLINK-6079": [
        "1 56120U 23042C   25095.75000000  .00007890  00000-0  34567-3 0  9993",
        "2 56120  53.0540 200.1234 0001234 100.5678 259.0123 15.23456789012345"
    ]
}

ts = load.timescale()
satellites = []
for name, tle in SATELLITE_TLE.items():
    sat = EarthSatellite(tle[0], tle[1], name, ts)
    satellites.append(sat)

def get_observer_j2000(utc_dt, lat_deg, lon_deg, alt_m=0):
    t = ts.utc(utc_dt.year, utc_dt.month, utc_dt.day,
               utc_dt.hour, utc_dt.minute, utc_dt.second + utc_dt.microsecond/1e6)
    geocentric = wgs84.latlon(lat_deg, lon_deg, alt_m).at(t)
    return geocentric.position.km

def get_satellite_obs_vectors(utc_dt, satellites, observer_j2000):
    t = ts.utc(utc_dt.year, utc_dt.month, utc_dt.day,
               utc_dt.hour, utc_dt.minute, utc_dt.second + utc_dt.microsecond/1e6)
    results = []
    for sat in satellites:
        sat_geocentric = sat.at(t)
        sat_j2000 = sat_geocentric.position.km
        rel_vec = sat_j2000 - observer_j2000
        norm = np.linalg.norm(rel_vec)
        if norm > 1e-6:
            direction = rel_vec / norm
            results.append((sat.name, direction))
    return results

# 融合 API 接口
def internal_residuals(params, data_rows, R_J2K_ENU):
    delta, phi_x, phi_y, phi_z, theta_np, eps_x, eps_y, eps_z = params
    R_ENU_MNT = rot_x(eps_x) @ rot_y(eps_y) @ rot_z(eps_z)
    res = []
    for row in data_rows:
        R_MNT_GIM = rot_z(-row['az']) @ rot_z(theta_np) @ rot_x(-row['alt']) @ rot_z(-theta_np)
        R_total = R_J2K_ENU @ R_ENU_MNT @ R_MNT_GIM @ euler_zxy(phi_z, phi_x, phi_y) @ rot_x(delta)
        for star in row['stars']:
            v_exp = R_total.T @ star['v_j2k']
            res.extend(v_exp - star['v_cam'])
    return np.array(res)

@app.route('/run_calibration', methods=['GET'])
def run_calibration_stream():
    def generate():
        utc_fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        gmst = gmst_rad(julian_day(utc_fixed))
        R_J2K_ENU = rot_z(-(gmst + math.radians(LON))) @ rot_y(math.radians(LAT) - math.pi/2)
        
        trajectories = [(math.radians(az), math.radians(45)) for az in range(0, 360, 30)] + \
                       [(math.radians(180), math.radians(alt)) for alt in range(15, 90, 10)]
        num_points = len(trajectories)
        data_rows = []

        for i, (az_rad, alt_rad) in enumerate(trajectories):
            progress = int((i / num_points) * 60)
            yield f"data: {json.dumps({'progress': progress, 'msg': f'控制台就绪，正在采集轨迹点 [{i+1}/{num_points}] ...'})}\n\n"
            time.sleep(0.15) 
            
            R_ENU_MNT = rot_x(EPS_X) @ rot_y(EPS_Y) @ rot_z(EPS_Z)
            R_MNT_GIM = rot_z(-az_rad) @ rot_z(THETA_NP) @ rot_x(-alt_rad) @ rot_z(-THETA_NP)
            R_total = R_J2K_ENU @ R_ENU_MNT @ R_MNT_GIM @ euler_zxy(PHI_Z, PHI_X, PHI_Y) @ rot_x(DELTA)
            
            base_cam_vectors = [
                np.array([0.0, 0.0, 1.0]),
                np.array([0.02, 0.02, 1.0]), 
                np.array([-0.02, 0.02, 1.0])
            ]
            
            star_pairs = []
            for v_cam_ideal in base_cam_vectors:
                v_cam_ideal /= np.linalg.norm(v_cam_ideal)
                v_j2k_true = R_total @ v_cam_ideal
                noise = np.random.normal(0, 5e-6, 3) 
                v_cam_obs = v_cam_ideal + noise
                v_cam_obs /= np.linalg.norm(v_cam_obs)
                star_pairs.append({'v_j2k': v_j2k_true, 'v_cam': v_cam_obs})
            
            data_rows.append({'az': az_rad, 'alt': alt_rad, 'stars': star_pairs})

        yield f"data: {json.dumps({'progress': 70, 'msg': '数据采集完毕，启动 Levenberg-Marquardt 寻优算法...'})}\n\n"
        time.sleep(0.5)
        
        initial_guess = np.zeros(8)
        bounds = (-0.1, 0.1)
        result = least_squares(
            internal_residuals, 
            initial_guess, 
            args=(data_rows, R_J2K_ENU), 
            bounds=bounds,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=5000,
            x_scale='jac'
        )
        
        yield f"data: {json.dumps({'progress': 90, 'msg': '优化收敛！正在进行跨姿态闭环误差验证...'})}\n\n"
        time.sleep(0.5)

        opt_p = result.x 
        test_az, test_alt = math.radians(45), math.radians(60)
        
        R_MNT_GIM_t = rot_z(-test_az) @ rot_z(THETA_NP) @ rot_x(-test_alt) @ rot_z(-THETA_NP)
        R_true = R_J2K_ENU @ (rot_x(EPS_X) @ rot_y(EPS_Y) @ rot_z(EPS_Z)) @ R_MNT_GIM_t @ euler_zxy(PHI_Z, PHI_X, PHI_Y) @ rot_x(DELTA)
        
        R_MNT_GIM_s = rot_z(-test_az) @ rot_z(opt_p[4]) @ rot_x(-test_alt) @ rot_z(-opt_p[4])
        R_sol = R_J2K_ENU @ (rot_x(opt_p[5]) @ rot_y(opt_p[6]) @ rot_z(opt_p[7])) @ R_MNT_GIM_s @ euler_zxy(opt_p[3], opt_p[1], opt_p[2]) @ rot_x(opt_p[0])
        
        R_diff = R_sol.T @ R_true
        trace_val = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
        angle_diff_arcsec = math.degrees(math.acos(trace_val)) * 3600

        final_data = {
            'progress': 100,
            'msg': '标定管线运行完毕。',
            'solved': list(opt_p),
            'error_arcsec': angle_diff_arcsec,
            'true_params': [DELTA, PHI_X, PHI_Y, PHI_Z, THETA_NP, EPS_X, EPS_Y, EPS_Z],
            'r_true': R_true.tolist(),
            'r_sol': R_sol.tolist()
        }
        yield f"data: {json.dumps(final_data)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/update_params', methods=['POST'])
def update_params():
    global DELTA, PHI_X, PHI_Y, PHI_Z, THETA_NP, EPS_X, EPS_Y, EPS_Z, LAT, LON
    try:
        data = request.get_json()
        DELTA = float(data.get('DELTA', DELTA))
        PHI_X = float(data.get('PHI_X', PHI_X))
        PHI_Y = float(data.get('PHI_Y', PHI_Y))
        PHI_Z = float(data.get('PHI_Z', PHI_Z))
        THETA_NP = float(data.get('THETA_NP', THETA_NP))
        EPS_X = float(data.get('EPS_X', EPS_X))
        EPS_Y = float(data.get('EPS_Y', EPS_Y))
        EPS_Z = float(data.get('EPS_Z', EPS_Z))
        LAT = float(data.get('LAT', LAT))
        LON = float(data.get('LON', LON))
        print("\n=== 收到前端参数更新 ===")
        print(f"新的经纬度: LAT={LAT}, LON={LON}")
        print("机械误差参数已更新为前端设定值。")
        print("========================\n")
        return jsonify({"status": "success", "message": "Parameters updated successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/stars', methods=['GET'])
def get_stars():
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
    dec = float(request.args.get('dec')) - 180
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
        az = float(request.args.get('az', 0))
        alt = float(request.args.get('alt', 0)) 
        focal_length = float(request.args.get('focal', 150.0))
        time_offset = float(request.args.get('time_offset', 0.0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "invalid params"}), 400

    utc_time = datetime.now(timezone.utc) + timedelta(hours=time_offset)
    
    R_J2K_OBS = compute_total_rotation(az, alt, utc_time)  
    v_j2k = R_J2K_OBS @ np.array([0, 0, 1])
    v_j2k /= np.linalg.norm(v_j2k)
    gmst = gmst_rad(julian_day(utc_time))
    
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

    # 渲染人造卫星（红色圆点）
    try:
        observer_pos = get_observer_j2000(utc_time, LAT, LON, alt_m=0)
        sat_vectors = get_satellite_obs_vectors(utc_time, satellites, observer_pos)
        for sat_name, sat_dir_j2000 in sat_vectors:
            v_cam = R_OBS_J2K @ sat_dir_j2000
            if v_cam[2] > 0:
                p = K @ (v_cam / v_cam[2])
                u_sat, v_sat = int(p[0]), int(p[1])
                if 0 <= u_sat < img_width and 0 <= v_sat < img_height:
                    cv2.circle(final_img_color, (u_sat, v_sat), 6, (0, 0, 255), -1)
                    cv2.circle(final_img_color, (u_sat, v_sat), 6, (255, 255, 255), 1)
                    name_short = sat_name.split()[0]
                    cv2.putText(final_img_color, name_short, (u_sat+8, v_sat-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    except Exception as e:
        print(f"卫星渲染出错: {e}")

    _, buffer = cv2.imencode('.jpg', final_img_color, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "status": "ok",
        "x": v_j2k[0], "y": v_j2k[1], "z": v_j2k[2],
        "gmst": gmst,
        "image": 'data:image/jpeg;base64,' + img_base64,
        "star_count": len(indices)
    })

@app.route('/run_calibration', methods=['POST'])
def run_calibration_post():
    obs_data = []
    utc_base = datetime.now(timezone.utc)
    
    for i in range(20):
        az = random.uniform(0, 360)
        alt = random.uniform(20, 80)
        utc_obs = utc_base + timedelta(minutes=i*3)
        R_true = compute_total_rotation(az, alt, utc_obs)
        v_true = R_true @ np.array([0, 0, 1])
        noise = np.random.normal(0, 4.8e-6, 3)
        v_noisy = (v_true + noise) / np.linalg.norm(v_true + noise)
        obs_data.append((az, alt, utc_obs, v_noisy))

    def residuals(p):
        res = []
        for az, alt, utc_obs, v_obs in obs_data:
            params = [0.0, p[0], p[1], p[2], p[3], p[4], p[5], p[6]]
            R_est = compute_total_rotation(az, alt, utc_obs, params)
            v_est = R_est @ np.array([0, 0, 1])
            res.extend(v_est - v_obs)
        return np.array(res)
    
    p0 = np.zeros(7)
    result = least_squares(residuals, p0, method='lm', x_scale='jac')
    p_opt = result.x
    
    return jsonify({
        "status": "ok",
        "calibrated_params": {
            "DELTA": 0.0, 
            "PHI_X": float(p_opt[0]), "PHI_Y": float(p_opt[1]), "PHI_Z": float(p_opt[2]),
            "THETA_NP": float(p_opt[3]),
            "EPS_X": float(p_opt[4]), "EPS_Y": float(p_opt[5]), "EPS_Z": float(p_opt[6])
        },
        "cost": result.cost
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)