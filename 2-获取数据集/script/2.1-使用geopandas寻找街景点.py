"""
使用GeoPandas寻找街景点

该脚本使用GeoPandas库处理建筑物几何数据，找到最近的道路点并计算角度。
支持并行处理以提高性能。
"""

import time
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import nearest_points


# 常量定义
BUFFER_DISTANCE = 30  # 30米
MAX_WORKERS = 10
PUSHKEY_PLACEHOLDER = "PDUxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


def calculate_midpoints(geometry):
    """根据输入的几何形状（Polygon或MultiPolygon），计算所有边的中点。
    
    Args:
        geometry: Polygon或MultiPolygon几何对象
        
    Returns:
        MultiPoint: 包含所有边中点的MultiPoint对象
    """
    midpoints = []

    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)

    for polygon in polygons:
        coords = list(polygon.exterior.coords)
        midpoints.extend(
            Point((coords[i][0] + coords[i + 1][0]) / 2, (coords[i][1] + coords[i + 1][1]) / 2) 
            for i in range(len(coords) - 1)
        )

    return MultiPoint(midpoints)


def calculate_angle(xs, ys, xc, yc):
    """计算两点之间的角度，相对于正北方向。
    
    Args:
        xs (float): 起始点X坐标
        ys (float): 起始点Y坐标  
        xc (float): 终点X坐标
        yc (float): 终点Y坐标
        
    Returns:
        float: 相对于正北方向的角度（度）
    """
    vn = np.array([0, 1])  # 北向量
    vsc = np.array([xc - xs, yc - ys])
    cos_theta = np.dot(vn, vsc) / (np.linalg.norm(vn) * np.linalg.norm(vsc))
    angle = np.degrees(np.arccos(cos_theta))

    # 调整角度为顺时针方向
    return angle if (xc - xs) >= 0 else 360 - angle


def read_road_data(road_gdf, building_polygon, buffer):
    """读取道路数据，只保留距离指定建筑物一定距离范围内的部分。
    
    Args:
        road_gdf: 道路GeoDataFrame
        building_polygon: 建筑物多边形
        buffer (float): 缓冲区距离
        
    Returns:
        GeometryCollection or None: 附近的道路几何集合，如果为空则返回None
    """
    buffered_poly = building_polygon.buffer(buffer)
    road_clip = road_gdf.geometry.intersection(buffered_poly)

    # 过滤掉空的几何形状
    road_nearby = GeometryCollection([geom for geom in road_clip if not geom.is_empty])

    return road_nearby if not road_nearby.is_empty else None


def process_geometry_and_calculate_angle(row, road_gdf, buffer_distance):
    """处理几何形状并计算角度。
    
    Args:
        row: DataFrame行数据，包含几何形状和中点信息
        road_gdf: 道路GeoDataFrame
        buffer_distance (float): 缓冲区距离
        
    Returns:
        dict: 包含最近点和角度的字典
    """
    building_polygon = row['geometry']
    midpoints = row['midpoints']  # 假设已经有了一个 'midpoints' 列

    road_nearby = read_road_data(road_gdf, building_polygon, buffer_distance)

    if road_nearby is None:
        return {'nearest_point': None, 'angle': None}

    nearest_center_point, nearest_road_point, shortest_distance = None, None, float('inf')

    for point in midpoints.geoms:
        current_nearest_road_point = nearest_points(point, road_nearby)[1]
        distance = point.distance(current_nearest_road_point)

        if distance < shortest_distance:
            shortest_distance, nearest_center_point, nearest_road_point = distance, point, current_nearest_road_point

    # 计算角度
    angle = calculate_angle(nearest_road_point.x, nearest_road_point.y, nearest_center_point.x, nearest_center_point.y)
    return {'nearest_point': nearest_center_point, 'angle': angle}


def parallel_processing(gdf, road_gdf, buffer_distance, max_workers):
    """使用多进程并行处理数据。
    
    Args:
        gdf: 建筑物GeoDataFrame
        road_gdf: 道路GeoDataFrame
        buffer_distance (float): 缓冲区距离
        max_workers (int): 最大工作进程数
        
    Returns:
        pd.DataFrame: 包含处理结果的DataFrame
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到进程池，每行数据使用一个进程
        futures = [
            executor.submit(process_geometry_and_calculate_angle, row, road_gdf, buffer_distance) 
            for index, row in gdf.iterrows()
        ]

        results = []
        for future in futures:
            # 从Future对象中获取结果
            result = future.result()
            results.append(result)

    # 构建结果DataFrame
    print("results", results)
    results_df = pd.DataFrame(results, columns=['nearest_point', 'angle'])
    return results_df


def save_results(gdf, output_path, crs):
    """将结果保存到文件中。
    
    Args:
        gdf: 包含结果数据的GeoDataFrame
        output_path (str): 输出文件路径
        crs: 坐标参考系统
    """
    gdf = gdf.dropna(subset=['nearest_point'])
    gdf = gdf[['identificatie', 'bouwjaar', 'angle', 'nearest_point']].rename(
        columns={'nearest_point': 'geometry'}).set_geometry('geometry')
    gdf.crs = crs
    gdf = gdf.to_crs(epsg=4326)
    gdf['lat'] = gdf.apply(lambda row: row['geometry'].y if pd.notnull(row['geometry']) else None, axis=1)
    gdf['lng'] = gdf.apply(lambda row: row['geometry'].x if pd.notnull(row['geometry']) else None, axis=1)

    gdf[['identificatie', 'bouwjaar', 'lat', 'lng', 'angle']].to_csv(output_path, index=False)


if __name__ == "__main__":
    """主函数：处理建筑物数据并找到最近的街景点。"""
    start_time = time.time()
    
    # 配置路径
    road_path = r"../../5-ArcgisPro工程/Amsterdam_road.gpkg"
    building_gdb = "..\\5-ArcgisPro工程\\建筑风格和年代深度学习.gdb"
    layer_name = 'Amsterdam_buildings_Project'
    output_path = "../data/output/Points_Amsterdam_use_gpd_all.csv"

    # 读取建筑数据
    gdf = gpd.read_file(building_gdb, layer=layer_name)
    gdf['midpoints'] = gdf.geometry.apply(calculate_midpoints)
    crs = gdf.crs
    print(f"坐标参考系统: {crs}")
    
    # 读取道路数据
    road_gdf = gpd.read_file(road_path, layer="edges").to_crs(crs)[["geometry"]].dropna()

    # 修复无效的几何形状
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    road_gdf['geometry'] = road_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))

    # 使用并行处理
    results_df = parallel_processing(gdf, road_gdf, BUFFER_DISTANCE, MAX_WORKERS)

    # 合并结果
    gdf = pd.concat([gdf, results_df], axis=1)

    # 保存结果
    save_results(gdf, output_path, crs)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'总共耗时：{elapsed_time:.2f}秒')

    # 发送通知
    try:
        from pypushdeer import PushDeer
        pushdeer = PushDeer(pushkey=PUSHKEY_PLACEHOLDER)
        pushdeer.send_text(
            f"寻找POINTS完成", 
            desp=f"总共耗时：{elapsed_time:.2f}s，数据集：{layer_name}，输出路径：{output_path}"
        )
    except Exception as e:
        print(f"发送通知失败: {e}")
