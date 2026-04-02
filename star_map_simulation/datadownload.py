from astroquery.gaia import Gaia
import pandas as pd

print("正在向 ESA 提交全天区（北半球可见）星表异步查询任务...")
print("这需要提取约一百多万颗恒星，请耐心等待几分钟...")

# 筛选视星等 <= 12，且赤纬 > -30度（覆盖北半球全部及南半球部分可见天区）
adql_query = """
SELECT source_id, ra, dec, phot_g_mean_mag 
FROM gaiadr3.gaia_source 
WHERE phot_g_mean_mag <= 12.0 
  AND dec > -30.0
"""

job = Gaia.launch_job_async(adql_query)
results = job.get_results()

df = results.to_pandas()
print(f"下载完成！共获取到 {len(df)} 颗恒星数据。")

# 为了提高后续加载速度，我们不仅存为 CSV，还可以存为 Numpy 的压缩格式 .npz
df.to_csv('gaia_northern_12mag.csv', index=False)
print("数据已保存至 gaia_northern_12mag.csv")