import geopandas as gpd
import time
import numpy as np
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.ops import nearest_points
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


# 定义计算多边形每条边中点的函数
def calculate_midpoints(geometry):
    """根据输入的几何形状（Polygon或MultiPolygon），计算所有边的中点。"""
    midpoints = []

    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)

    for polygon in polygons:
        coords = list(polygon.exterior.coords)
        midpoints.extend(Point((coords[i][0] + coords[i + 1][0]) / 2, (coords[i][1] + coords[i + 1][1]) / 2) for i in
                         range(len(coords) - 1))

    return MultiPoint(midpoints)


def calculate_angle(xs, ys, xc, yc):
    """计算两点之间的角度，相对于正北方向。"""
    Vn = np.array([0, 1])  # 北向量
    Vsc = np.array([xc - xs, yc - ys])
    cos_theta = np.dot(Vn, Vsc) / (np.linalg.norm(Vn) * np.linalg.norm(Vsc))
    angle = np.degrees(np.arccos(cos_theta))

    # 调整角度为顺时针方向
    return angle if (xc - xs) >= 0 else 360 - angle


# 从文件中读取和处理道路数据
def read_road(road_gdf, building_polygon, buffer):
    """读取道路数据，只保留距离指定建筑物一定距离范围内的部分。"""
    buffered_poly = building_polygon.buffer(buffer)
    road_clip = road_gdf.geometry.intersection(buffered_poly)

    # 过滤掉空的几何形状
    road_nearby = GeometryCollection([geom for geom in road_clip if not geom.is_empty])

    return road_nearby if not road_nearby.is_empty else None


# 查找最近点和计算角度
def process_geometry_and_calculate_angle(row, road_gdf, buffer_distance):
    building_polygon = row['geometry']
    midpoints = row['midpoints']  # 这里假设你已经有了一个 'midpoints' 列

    road_nearby = read_road(road_gdf, building_polygon, buffer_distance)

    if road_nearby is None:
        return {'nearest_point': None, 'angle': None}  # 或者其他适当的处理

    nearest_center_point, nearest_road_point, shortest_distance = None, None, float('inf')

    for point in midpoints.geoms:
        current_nearest_road_point = nearest_points(point, road_nearby)[1]
        distance = point.distance(current_nearest_road_point)

        if distance < shortest_distance:
            shortest_distance, nearest_center_point, nearest_road_point = distance, point, current_nearest_road_point

    # if shortest_distance > 20:
    #     # print("距离太远，不予考虑")
    #     return None  # 或者其他适当的处理

    # 计算角度
    angle = calculate_angle(nearest_road_point.x, nearest_road_point.y, nearest_center_point.x, nearest_center_point.y)
    return {'nearest_point': nearest_center_point, 'angle': angle}  # 根据需要返回相关信息


def parallel_processing(gdf, road_gdf, buffer_distance, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:  # 你可以根据你的系统资源调整 max_workers
        # 提交所有任务到进程池，每行数据使用一个进程
        futures = [executor.submit(process_geometry_and_calculate_angle, row, road_gdf, buffer_distance) for index, row
                   in gdf.iterrows()]

        results = []
        for future in futures:
            # 从Future对象中获取结果
            result = future.result()
            results.append(result)

    # 你可能需要根据返回的结果调整这部分，以构建你的结果数据帧
    print("results", results)
    results_df = pd.DataFrame(results, columns=['nearest_point', 'angle'])  # 或者使用适当的方式转换结果列表
    return results_df


def save_results(gdf, output_path, crs):
    """将结果保存到文件中。"""
    gdf = gdf.dropna(subset=['nearest_point'])
    gdf = gdf[['identificatie', 'bouwjaar', 'angle', 'nearest_point']].rename(
        columns={'nearest_point': 'geometry'}).set_geometry('geometry')
    gdf.crs = crs
    gdf = gdf.to_crs(epsg=4326)
    gdf['lat'] = gdf.apply(lambda row: row['geometry'].y if pd.notnull(row['geometry']) else None, axis=1)
    gdf['lng'] = gdf.apply(lambda row: row['geometry'].x if pd.notnull(row['geometry']) else None, axis=1)

    gdf[['identificatie', 'bouwjaar', 'lat', 'lng', 'angle']].to_csv(output_path, index=False)


if __name__ == "__main__":  # 建议使用这个检查，因为新进程会尝试重新运行脚本
    time1 = time.time()
    # ... [数据加载和预处理] ...
    path = r"../../5-ArcgisPro工程/Amsterdam_road.gpkg"

    buffer_distance = 30  # 30米

    # 读取建筑
    gdb = "..\\5-ArcgisPro工程\\建筑风格和年代深度学习.gdb"
    lr_name = 'Amsterdam_buildings_Project'
    gdf = gpd.read_file(gdb, layer=lr_name)  # rows=1000
    gdf['midpoints'] = gdf.geometry.apply(calculate_midpoints)
    crs = gdf.crs
    print(crs)
    # 读取道路
    road_gdf = gpd.read_file(path, layer="edges").to_crs(crs)[["geometry"]].dropna()

    # 修复无效的几何形状
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    road_gdf['geometry'] = road_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))

    # 使用并行处理
    results_df = parallel_processing(gdf, road_gdf, buffer_distance, max_workers=10)

    gdf = pd.concat([gdf, results_df], axis=1)

    # print(gdf)
    # print(gdf.columns)
    # 保存结果
    save_results(gdf, output_path:="../data/output/Points_Amsterdam_use_gpd_all.csv", crs)

    time2 = time.time()
    print('总共耗时：' + str(time2 - time1) + 's')

    # 通知
    from pypushdeer import PushDeer

    pushdeer = PushDeer(pushkey="PDUxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    pushdeer.send_text(f"寻找POINTS完成", desp=f"总共耗时：{time2 - time1}s，数据集：{lr_name}，输出路径：{output_path}")
