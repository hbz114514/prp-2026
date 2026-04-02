import pandas as pd
import numpy as np
import cv2
from scipy.spatial import cKDTree
from datetime import datetime

# ==========================================
# 1. 核心渲染与标注函数
# ==========================================
def draw_gaussian_spot(img, cx, cy, brightness, sigma=1.5):
    """亚像素高斯光斑渲染"""
    height, width = img.shape
    size = int(sigma * 6) | 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    dx, dy = cx - int(cx), cy - int(cy)
    g = brightness * np.exp(-((x - x0 - dx)**2 + (y - y0 - dy)**2) / (2 * sigma**2))
    
    ix, iy = int(cx), int(cy)
    r = size // 2
    x_min, x_max = max(0, ix - r), min(width, ix - r + size)
    y_min, y_max = max(0, iy - r), min(height, iy - r + size)
    g_x_min, g_x_max = r - (ix - x_min), r + (x_max - ix)
    g_y_min, g_y_max = r - (iy - y_min), r + (y_max - iy)
    
    if x_min < x_max and y_min < y_max:
        img[y_min:y_max, x_min:x_max] += g[g_y_min:g_y_max, g_x_min:g_x_max]

def draw_grid_and_annotations(img, R_OBS_J2K, K, ra0, dec0, focal_len, fov_deg):
    """在星图上绘制坐标网格、FOV信息、时间和地点"""
    h, w = img.shape[:2]
    grid_color = (80, 80, 80)     # 暗灰色网格线
    text_color = (200, 200, 200)  # 浅灰色文字

    # --- 1. 绘制赤纬线 (Dec) ---
    dec_grid = np.arange(int(dec0) - 10, int(dec0) + 11, 1)
    for d_deg in dec_grid:
        if d_deg < -90 or d_deg > 90: continue
        ra_samples = np.linspace(ra0 - 15, ra0 + 15, 150)
        pts = []
        for r_deg in ra_samples:
            r, d = np.radians(r_deg), np.radians(d_deg)
            v_j2k = np.array([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])
            v_obs = R_OBS_J2K @ v_j2k
            if v_obs[2] > 0: 
                p_homo = K @ (v_obs / v_obs[2])
                if 0 <= p_homo[0] < w and 0 <= p_homo[1] < h:
                    pts.append([int(p_homo[0]), int(p_homo[1])])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, grid_color, 1, cv2.LINE_AA)
            cv2.putText(img, f"Dec: {d_deg}", tuple(pts[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)

    # --- 2. 绘制赤经线 (RA) ---
    ra_grid = np.arange(int(ra0) - 15, int(ra0) + 16, 2)
    for r_deg in ra_grid:
        dec_samples = np.linspace(dec0 - 10, dec0 + 10, 100)
        pts = []
        for d_deg in dec_samples:
            r, d = np.radians(r_deg), np.radians(d_deg)
            v_j2k = np.array([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])
            v_obs = R_OBS_J2K @ v_j2k
            if v_obs[2] > 0:
                p_homo = K @ (v_obs / v_obs[2])
                if 0 <= p_homo[0] < w and 0 <= p_homo[1] < h:
                    pts.append([int(p_homo[0]), int(p_homo[1])])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, grid_color, 1, cv2.LINE_AA)
            cv2.putText(img, f"RA: {r_deg}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)

    # --- 3. 叠加关键文字信息 ---
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_lines = [
        f"CENTER RA: {ra0:.2f} DEG",
        f"CENTER DEC: {dec0:.2f} DEG",
        f"FOCAL LEN: {focal_len:.1f} mm",
        f"FOV (H): {fov_deg:.2f} DEG",
        f"RES: {w}x{h}",
        f"TIME: {current_time}",
        f"LOC: 31.027 N, 121.432 E"
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(img, line, (20, 40 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

# ==========================================
# 2. 初始化：载入数据与构建空间索引
# ==========================================
print("正在载入全天区星表并构建 KD-Tree 空间索引...")
df = pd.read_csv('gaia_northern_12mag.csv')
mags_all = df['phot_g_mean_mag'].values
ra_rad_all = np.radians(df['ra'].values)
dec_rad_all = np.radians(df['dec'].values)

stars_3d_all = np.vstack((
    np.cos(dec_rad_all)*np.cos(ra_rad_all),
    np.cos(dec_rad_all)*np.sin(ra_rad_all),
    np.sin(dec_rad_all)
)).T 

tree = cKDTree(stars_3d_all)
print("系统准备就绪。")

# 固定的传感器硬件参数
pixel_size = 5.0e-3
img_width, img_height = 1920, 1080 

# ==========================================
# 3. 交互式查询循环
# ==========================================
while True:
    print("\n" + "-"*50)
    user_input = input("请输入: 赤经RA(度) 赤纬Dec(度) 焦距(mm) (空格隔开)。输入 q 退出：\n> ")
    if user_input.lower() == 'q':
        break
    try:
        # 解析三个输入参数
        ra_in, dec_in, focal_length = map(float, user_input.split())
    except ValueError:
        print("输入格式错误，请确保输入了三个数字（例如：100 50 150）。")
        continue

    # --- 核心修改：在此处动态计算相机光学参数 ---
    f_px = focal_length / pixel_size
    K = np.array([[f_px, 0, img_width / 2.0], [0, f_px, img_height / 2.0], [0, 0, 1]])
    
    # 计算当前焦距下的实际水平视场角
    actual_fov = np.degrees(2 * np.arctan((img_width * pixel_size) / (2 * focal_length)))
    
    # 动态设定 KD-Tree 的搜索半径 (乘以1.2作为安全余量，防止边缘星星遗漏)
    search_radius = 2.0 * np.sin(np.radians(actual_fov * 1.2) / 2.0)

    target_ra_rad, target_dec_rad = np.radians(ra_in), np.radians(dec_in)
    
    # 建立当前观测光轴 (Z轴)
    z_c = np.array([
        np.cos(target_dec_rad)*np.cos(target_ra_rad),
        np.cos(target_dec_rad)*np.sin(target_ra_rad),
        np.sin(target_dec_rad)
    ])
    
    # KD-Tree 极速检索
    indices = tree.query_ball_point(z_c, search_radius)
    
    if not indices:
        print(f"当前焦距 ({focal_length}mm, FOV={actual_fov:.2f}°) 下，该天区视野内没有星等 <= 12 的恒星。")
        continue
        
    print(f"检索到视野内有 {len(indices)} 颗恒星，正在渲染...")
    
    # 构建外参矩阵 R_OBS_J2K
    np_vec = np.array([0, 0, 1])
    x_c = np.cross(np_vec, z_c)
    if np.linalg.norm(x_c) < 1e-6:
        x_c = np.array([1, 0, 0])
    x_c = x_c / np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    R_OBS_J2K = np.vstack((x_c, y_c, z_c))
    
    # 提取视野内恒星并投影
    local_stars_j2k = stars_3d_all[indices]
    local_mags = mags_all[indices]
    
    stars_obs = (R_OBS_J2K @ local_stars_j2k.T).T
    front_mask = stars_obs[:, 2] > 0
    stars_obs, local_mags = stars_obs[front_mask], local_mags[front_mask]
    
    coords_homo = (K @ (stars_obs / stars_obs[:, 2:3]).T).T
    u, v = coords_homo[:, 0], coords_homo[:, 1]
    
    # 初始化黑色画布
    star_map = np.zeros((img_height, img_width), dtype=np.float32)
    
    for i in range(len(u)):
        mag = local_mags[i]
        sigma = 1.0 + max(0, (12.0 - mag) * 0.3)
        brightness = 20.0 * (2.512 ** (12.0 - mag))
        draw_gaussian_spot(star_map, u[i], v[i], brightness, sigma)
        
    # 添加传感器白噪声
    noise = np.random.normal(12, 6, (img_height, img_width))
    star_map += noise
    
    # 转换为 BGR 彩色图
    final_img_gray = np.clip(star_map, 0, 255).astype(np.uint8)
    final_img_color = cv2.cvtColor(final_img_gray, cv2.COLOR_GRAY2BGR)
    
    # 叠加经纬网格和文字信息 (传入当前的 focal_length)
    draw_grid_and_annotations(final_img_color, R_OBS_J2K, K, ra_in, dec_in, focal_length, actual_fov)
    
    # 保存结果
    filename = f'star_{ra_in}_{dec_in}_f{int(focal_length)}.png'
    cv2.imwrite(filename, final_img_color)
    print(f"渲染成功！星图已保存为 {filename}")