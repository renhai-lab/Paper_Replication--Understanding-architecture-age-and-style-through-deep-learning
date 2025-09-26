"""
通过街景API获取包含URL的CSV文件

该脚本用于从建筑物坐标和角度信息生成Google街景地图的URL，
支持多线程处理以提高效率。
"""

import concurrent.futures
import glob
import os
import threading
from datetime import datetime

import pandas as pd
from streetview import search_panoramas
from tqdm import tqdm

# 常量定义
MAX_THREADS = 15
TEST_ROWS = 50  # 用于测试的行数
ZOOM_LEVEL = 80  # 街景缩放比例，范围0-90，越大距离越远
PUSHKEY_PLACEHOLDER = "PDUXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# 创建一个锁对象，以保护文件写入操作
write_lock = threading.Lock()

# 加载 tqdm.pandas() 以便使用 progress_apply
tqdm.pandas()


def construct_streetview_url(row):
    """根据给定的DataFrame行构建Google街景地图的URL。

    Args:
        row (pandas.Series): DataFrame的一行，应包含lat, lng和angle字段
        
    Returns:
        tuple: 包含URL和日期的元组，如果失败则返回(None, None)
    """
    try:
        # 从行数据中提取经纬度和朝向
        lat = row['lat']
        lng = row['lng']
        heading = row['angle']

        # 使用streetview模块搜索给定经纬度的全景图
        panos = search_panoramas(lat, lng)

        # 如果没有找到全景图，返回None
        if not panos:
            return None, None

        # 获取当前日期和时间
        current_date = datetime.now()

        # 使用列表推导式和min函数找到日期最接近current_date的全景图
        # 如果某个全景图没有日期，我们将其设置为无穷大，这样它就不会被选择为最近的全景图
        # 在这里，我们确保 .total_seconds() 只应用于 timedelta 对象
        closest_pano = min(panos, key=lambda p: (
                    current_date - datetime.strptime(p.date, '%Y-%m')).total_seconds() 
                    if p.date else float('inf'))

        # 提取最接近的全景图的经纬度和ID
        lat = closest_pano.lat
        lng = closest_pano.lon
        date = closest_pano.date
        pano_id = closest_pano.pano_id

        # 使用提取的数据构建Google街景地图的URL
        # ZOOM_LEVEL为缩放比例，范围0-90，越大距离越远，多次调整后80比较合适
        url = (f"https://www.google.com/maps/@{lat},{lng},3a,{ZOOM_LEVEL}y,{heading}h,96t/"
               f"data=!3m6!1e1!3m4!1s{pano_id}!2e0!7i16384!8i8192")

        result = (url, date)
        print("返回结果:", result)

        # 返回一个包含url和date的元组
        return result
    except Exception as e:
        # 处理异常的逻辑，例如打印错误信息
        print(f"构建URL时出错: {e}")
        # 返回空值或其他默认值
        return None, None

def process_csv_file(path, output_path):
    """处理单个CSV文件，为每一行生成街景URL。
    
    Args:
        path (str): 输入CSV文件路径
        output_path (str): 输出CSV文件路径
    """
    # 读取CSV文件，只读取前指定行数进行测试
    df = pd.read_csv(path, encoding='utf-8', header=0).head(TEST_ROWS)

    # 使用 progress_apply 替代 apply，以显示进度条
    df[['url', 'date']] = df.progress_apply(
        lambda row: pd.Series(construct_streetview_url(row)), axis=1
    )

    # 删除空值并移除不需要的列：identificatie, bouwjaar, lat, lng, angle
    df_clean = df.dropna().drop(['lat', 'lng', 'angle'], axis=1)

    # 使用锁保护文件写入操作
    with write_lock:
        # 只在文件不存在时写入标题
        df_clean.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        print(f"{output_path} 文件保存成功")

if __name__ == "__main__":
    """主函数：处理CSV文件并生成街景URL。"""
    # 配置输入和输出路径
    # pathlists = glob.glob("street_view_point_Amsterdam_buildings_subset_*.csv")
    pathlists = ["../data/output/Points_Amsterdam_use_gpd_all.csv"]
    output_path = "../../data/output/street_view_url_all.csv"

    # 检查输出目录是否可写
    assert os.access(os.path.dirname(output_path), os.W_OK), "输出目录不可写"

    # 创建线程池
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS)

    # 使用线程池进行文件处理
    futures = []
    for path in pathlists:
        future = executor.submit(process_csv_file, path, output_path)
        futures.append(future)

    # 等待所有任务完成
    concurrent.futures.wait(futures)

    # 发送完成通知
    try:
        from pypushdeer import PushDeer
        pushdeer = PushDeer(pushkey=PUSHKEY_PLACEHOLDER)
        pushdeer.send_text(f"URL构建完成", desp=f"输出路径：{output_path}")
    except ImportError:
        print("PushDeer模块未安装，跳过通知")
    except Exception as e:
        print(f"发送通知时出错: {e}")
        
    print("街景URL生成任务完成！")

