import math
import random
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)

random.seed(42)

# 1. 环境与重力干扰修正
DELTA = random.uniform(-0.01, 0.01)

# 2. 安装偏差
PHI_X = random.uniform(-0.02, 0.02)
PHI_Y = random.uniform(-0.02, 0.02)
PHI_Z = random.uniform(-0.02, 0.02)

# 3. 轴系不垂直度
THETA_NP = random.uniform(-0.005, 0.005)

# 4. 底座安装误差
EPS_X = random.uniform(-0.01, 0.01)
EPS_Y = random.uniform(-0.01, 0.01)
EPS_Z = random.uniform(-0.01, 0.01)

# 观测地坐标（上海闵行区）
LAT = 31.22        
LON = 121.48       


# ==================== 基础旋转矩阵 ====================
def rot_x(angle):
    c = math.cos(angle )   
    s = math.sin(angle )
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rot_y(angle):
    c = math.cos(angle ) 
    s = math.sin(angle )
    return np.array([[-c, 0, s],
                     [0, 1, 0],
                     [s, 0, c]])

def rot_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def euler_zxy(phi_z, phi_x, phi_y):
    return rot_x(phi_x) @ rot_y(phi_y) @ rot_z(phi_z)


def julian_day(date):
    year = date.year
    month = date.month
    day = date.day + date.hour/24.0 + date.minute/1440.0 + date.second/86400.0
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    JD = math.floor(365.25*(year + 4716)) + math.floor(30.6001*(month + 1)) + day + B - 1524.5
    return JD

def gmst_rad(jd):
    """输入儒略日，返回格林尼治平恒星时（弧度）"""
    T = (jd - 2451545.0) / 36525.0
    gmst_sec = 24110.54841 + 8640184.812866 * T + 0.093104 * T*T - 0.0000062 * T*T*T
    gmst_sec %= 86400
    return gmst_sec * 2 * math.pi / 86400


def R_J2K_ENU(utc_time):
    jd = julian_day(utc_time)
    gmst = gmst_rad(jd)
    lon_rad = LON * math.pi / 180.0
    lat_rad = LAT * math.pi / 180.0
    theta = gmst + lon_rad                      
    angle_y = lat_rad - math.pi/2              
    Rz = rot_z(-theta)
    Ry = rot_y(angle_y)
    return Rz @ Ry

def R_ENU_MNT():
    Rz = rot_z(EPS_Z)
    Ry = rot_y(EPS_Y)
    Rx = rot_x(EPS_X)
    return Rx @ Ry @ Rz

def R_MNT_GIM(az_rad, alt_rad):
    R_azi = rot_z(az_rad)
    R_alt = rot_x(alt_rad)
    R_NP = rot_z(THETA_NP)
    R_azi_T = rot_z(-az_rad)
    R_alt_T = rot_x(-alt_rad)
    R_NP_T = rot_z(-THETA_NP)
    return R_azi_T @ R_NP @ R_alt_T @ R_NP_T

def R_GIM_ST():
    return euler_zxy(PHI_Z, PHI_X, PHI_Y)

def R_ST_OBS():
    return rot_x(DELTA)


# ==================== 主计算函数 ====================
def compute_pointing_vector(az_deg, alt_deg, utc_time):
    az_rad = az_deg * math.pi / 180.0
    alt_rad = alt_deg * math.pi / 180.0
    R_total = R_J2K_ENU(utc_time)
    R_total = R_total @ R_ENU_MNT()
    R_total = R_total @ R_MNT_GIM(az_rad, alt_rad)
    R_total = R_total @ R_GIM_ST()
    R_total = R_total @ R_ST_OBS()

    v_obs = np.array([0, 0, 1])
    v_j2k = R_total @ v_obs
    v_j2k /= np.linalg.norm(v_j2k)      
    return v_j2k.tolist()


# ==================== Flask API ====================
@app.route('/pointing', methods=['GET'])
def pointing():
    """接收参数 az(方位角), alt(仰角)，返回 J2000 方向向量"""
    try:
        az = float(request.args.get('az', 0))
        alt = float(request.args.get('alt', 0))  - 180.0  
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "az and alt must be numbers"}), 400

    utc_now = datetime.now(timezone.utc)
    x, y, z = compute_pointing_vector(az, alt, utc_now)

    return jsonify({
        "status": "ok",
        "x": x,
        "y": y,
        "z": z
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)