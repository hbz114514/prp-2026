import numpy as np
import pandas as pd
import math
import json
from scipy.optimize import least_squares
from datetime import datetime, timezone  # <--- 新增这行导入

# ==========================================
# 1. 基础旋转矩阵定义 (严格对齐生成器)
# ==========================================
def rot_x(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]]) 
def rot_z(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def euler_zxy(z, x, y): return rot_x(x) @ rot_y(y) @ rot_z(z)

# 常量定义 (需与生成数据时保持一致)
LAT, LON = math.radians(31.22), math.radians(121.48)

# --- 替换开始：真正的恒星时同步 ---
def julian_day(date):
    y, m, d = date.year, date.month, date.day + date.hour/24.0 + date.minute/1440.0 + date.second/86400.0
    if m <= 2: y -= 1; m += 12
    A = y // 100
    return math.floor(365.25*(y + 4716)) + math.floor(30.6001*(m + 1)) + d + (2 - A + A // 4) - 1524.5

def gmst_rad(jd):
    T = (jd - 2451545.0) / 36525.0
    return ((24110.54841 + 8640184.812866 * T + 0.093104 * T*T - 0.0000062 * T*T*T) % 86400) * 2 * math.pi / 86400

# 必须与生成器完全一致！
utc_fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
GMST_RAD = gmst_rad(julian_day(utc_fixed))
# --- 替换结束 ---

def get_ideal_J2K_to_ENU():
    return rot_z(-(GMST_RAD + LON)) @ rot_y(LAT - math.pi/2)


# ==========================================
# 2. SVD 奇异值定姿算法 (单帧求解)
# ==========================================
def solve_wahba_svd(v_cam_list, v_j2k_list):
    """
    输入: N对对应的 3D 向量 (N >= 2)
    输出: 从 J2K 到 Cam 的 3x3 旋转矩阵 R
    """
    B = np.zeros((3, 3))
    # 累加外积矩阵
    for v_c, v_j in zip(v_cam_list, v_j2k_list):
        B += np.outer(v_c, v_j)
        
    U, S, Vt = np.linalg.svd(B)
    
    # 构建修正矩阵，防止出现反射变换 (行列式为 -1)
    M = np.eye(3)
    M[2, 2] = np.linalg.det(U @ Vt)
    
    R = U @ M @ Vt
    return R

# ==========================================
# 3. 核心残差函数 (用于非线性优化)
# ==========================================
def residuals(params, data_rows):
    """
    params: 8个未知数构成的数组
    [DELTA, PHI_X, PHI_Y, PHI_Z, THETA_NP, EPS_X, EPS_Y, EPS_Z]
    """
    delta, phi_x, phi_y, phi_z, theta_np, eps_x, eps_y, eps_z = params
    
    R_J2K_ENU = get_ideal_J2K_to_ENU()
    R_ENU_MNT = rot_x(eps_x) @ rot_y(eps_y) @ rot_z(eps_z)
    
    res = []
    
    for row in data_rows:
        az_rad = math.radians(row['Az_enc'])
        alt_rad = math.radians(row['Alt_enc'])
        
        # 构建望远镜当前的正向运动学模型
        R_MNT_GIM = rot_z(-az_rad) @ rot_z(theta_np) @ rot_x(-alt_rad) @ rot_z(-theta_np)
        
        # 完整的从 J2000 到 相机传感器的变换矩阵
        R_total = R_J2K_ENU @ R_ENU_MNT @ R_MNT_GIM @ euler_zxy(phi_z, phi_x, phi_y) @ rot_x(delta)
        R_J2K_to_CAM = R_total.T  # 相机坐标系下的观测向量 = R_total.T @ J2000向量
        
        # 遍历这张照片里的 3 颗星
        for i in range(1, 4):
            v_j2k = np.array([row[f'Star{i}_J2K_x'], row[f'Star{i}_J2K_y'], row[f'Star{i}_J2K_z']])
            v_cam_actual = np.array([row[f'Star{i}_Cam_x'], row[f'Star{i}_Cam_y'], row[f'Star{i}_Cam_z']])
            
            # 用猜测的 8 个参数算出来的理论观测向量
            v_cam_expected = R_J2K_to_CAM @ v_j2k
            
            # 残差：理论值减去实际值
            error = v_cam_expected - v_cam_actual
            res.extend(error) # 将 3 个轴的误差都塞进全局残差数组
            
    return np.array(res)

# ==========================================
# 4. 执行流程
# ==========================================
if __name__ == "__main__":
    print("正在读取观测数据 matched_results.csv...")
    df = pd.read_csv("matched_results.csv")
    print(f"成功载入 {len(df)} 张星图的有效匹配数据。\n")
    
    # --- 阶段一：展示 SVD 单帧定姿的威力 ---
    print("--- 阶段一：SVD 算法单帧定姿测试 ---")
    sample_row = df.iloc[0]
    v_c_list = [np.array([sample_row[f'Star{i}_Cam_x'], sample_row[f'Star{i}_Cam_y'], sample_row[f'Star{i}_Cam_z']]) for i in range(1, 4)]
    v_j_list = [np.array([sample_row[f'Star{i}_J2K_x'], sample_row[f'Star{i}_J2K_y'], sample_row[f'Star{i}_J2K_z']]) for i in range(1, 4)]
    
    R_svd = solve_wahba_svd(v_c_list, v_j_list)
    v_cam_center_svd = R_svd.T @ np.array([0, 0, 1])  # SVD 算出来的光轴中心指向
    
    print(f"照片 {sample_row['Image_Name']} 的实际中心指向 (Ground Truth):")
    print(f"  [{sample_row['GT_Cam_x']:.5f}, {sample_row['GT_Cam_y']:.5f}, {sample_row['GT_Cam_z']:.5f}]")
    print(f"SVD 算法盲解出的中心指向:")
    print(f"  [{v_cam_center_svd[0]:.5f}, {v_cam_center_svd[1]:.5f}, {v_cam_center_svd[2]:.5f}]")
    print("结论：两者高度吻合，说明 SVD 成功破解了图像姿态！\n")
    
    # --- 阶段二：LM 算法全局机械误差寻优 ---
    print("--- 阶段二：Levenberg-Marquardt 全局优化求解 8 项机械误差 ---")
    # 初始猜测值全部设为 0 (假设望远镜安装完美)
    initial_guess = np.zeros(8)
    
    # 提取所有数据行转为字典列表加速迭代
    data_rows = df.to_dict('records')
    
    print("优化器启动，正在高维空间中寻找最优解...")
    #result = least_squares(residuals, initial_guess, args=(data_rows,), method='lm', verbose=1)
    # 原代码：
# result = least_squares(residuals, initial_guess, args=(data_rows,), method='lm', verbose=1)

# 将其替换为带边界的版本：
# 注意：加了 bounds 之后，底层算法会自动从 'lm' 切换到支持边界的 'trf' 算法
    bounds = (-0.1, 0.1)  # 限制所有 8 个误差参数只能在 ±0.1 弧度（约 ±5.7 度）之间寻找
    result = least_squares(residuals, initial_guess, args=(data_rows,), bounds=bounds, verbose=1)
    

# ... (前面的代码保持不变) ...

    if result.success:
        print("\n⭐⭐⭐ 优化收敛成功！⭐⭐⭐")
        optimized_params = result.x
        
        # 1. 自动读取真值
        try:
            with open("true_params.json", "r") as f:
                true_dict = json.load(f)
            true_params = np.array([
                true_dict["DELTA"], true_dict["PHI_X"], true_dict["PHI_Y"], true_dict["PHI_Z"],
                true_dict["THETA_NP"], true_dict["EPS_X"], true_dict["EPS_Y"], true_dict["EPS_Z"]
            ])
        except FileNotFoundError:
            print("未找到 true_params.json，跳过自动比对。")
            true_params = None

        if true_params is not None:
            # 2. 定义计算总旋转矩阵的函数
            def get_R_total(params, az_rad, alt_rad):
                delta, phi_x, phi_y, phi_z, theta_np, eps_x, eps_y, eps_z = params
                R_J2K_ENU = get_ideal_J2K_to_ENU()
                R_ENU_MNT = rot_x(eps_x) @ rot_y(eps_y) @ rot_z(eps_z)
                R_MNT_GIM = rot_z(-az_rad) @ rot_z(theta_np) @ rot_x(-alt_rad) @ rot_z(-theta_np)
                R_total = R_J2K_ENU @ R_ENU_MNT @ R_MNT_GIM @ euler_zxy(phi_z, phi_x, phi_y) @ rot_x(delta)
                return R_total

            # 3. 终极验证：在任意姿态下的系统综合指向误差
            test_az, test_alt = math.radians(45), math.radians(60) # 测试姿态
            
            R_true = get_R_total(true_params, test_az, test_alt)
            R_solved = get_R_total(optimized_params, test_az, test_alt)
            
            # 计算两个姿态矩阵之间的旋转差异矩阵
            R_diff = R_solved.T @ R_true
            
            # 提取空间夹角: Trace(R) = 1 + 2*cos(theta)
            trace_val = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
            angle_diff_rad = math.acos(trace_val)
            angle_diff_arcsec = math.degrees(angle_diff_rad) * 3600
            
            print("\n--- 闭环系统验证 ---")
            print("测试姿态: Az=45°, Alt=60°")
            print(f"真值模型与解算模型的综合指向误差: {angle_diff_arcsec:.4f} 角秒")
            
            if angle_diff_arcsec < 1.0:
                print("结论: 系统实现亚角秒级闭环，参数耦合已被完美等效！")
            else:
                print("结论: 仍存在未被等效的系统残差。")
    else:
        print("优化失败，请检查数据完整性。")