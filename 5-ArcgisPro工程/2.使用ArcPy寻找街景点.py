import time
import uuid

import arcpy
import pandas as pd


def transform_angle(original_angle):
    """
    将角度从一个坐标系转换为另一个，并更改方向表示。
    :param original_angle: 初始的角度（基于东为0°的系统）
    :return: 转换后的角度（基于北为0°的系统）
    """
    # 从建筑物到街道的角度需要将角度旋转180度以“反转”方向
    print("original_angle:", original_angle)
    reversed_angle = original_angle + 180

    # 规范化角度在0到360之间
    if reversed_angle >= 360:
        reversed_angle -= 360
    elif reversed_angle < 0:
        reversed_angle += 360

    # 现在，我们需要将“东为0度”转变为“北为0度”，这需要一个90度的逆时针旋转
    north_based_angle = reversed_angle + 90

    if north_based_angle >= 360:
        north_based_angle -= 360
    print("north_based_angle：", north_based_angle)
    return north_based_angle

def field_exists(feature_class, field_name):
    return field_name in [f.name for f in arcpy.ListFields(feature_class)]


def get_midpoint(start_point, end_point):
    if not start_point or not end_point:
        return None
    mid_x = (start_point.X + end_point.X) / 2
    mid_y = (start_point.Y + end_point.Y) / 2
    return arcpy.Point(mid_x, mid_y)


def main(feature_class, road_feature_class, csv_path):
    """主函数
    输入参数：
    feature_class: 输入的多边形数据集
    road_feature_class: 道路数据集
    csv_path: 输出的 csv 文件路径
    """
    # 开始计时
    start_time = time.time()

    sr = arcpy.SpatialReference(32631)
    field_name_list = ["NEAR_DIST", "NEAR_X", "NEAR_Y", "NEAR_ANGLE"]

    # 获取多边形总数
    total_polygons = int(arcpy.GetCount_management(feature_class).getOutput(0))
    print(f"{feature_class}的所有多边形数量为{total_polygons}个")

    data = feature_class, road_feature_class, csv_path, sr, total_polygons, field_name_list

    # 处理数据的主要函数
    process_feature(data)

    # 计算并输出执行时间
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f"执行完成：{feature_class}数据集的执行实现: {minutes} minutes {seconds:.2f} seconds")


def process_feature(data):
    """处理并保存，
    输入参数：
    data: input_polygon_fc, road_feature_class, csv_path, sr, total_polygons, field_name_list 一个元组，包含了所有需要的参数
    """
    input_polygon_fc, road_feature_class, csv_path, sr, total_polygons, field_name_list = data

    results = []
    # 计数器初始化
    processed_count = 0

    with arcpy.da.SearchCursor(input_polygon_fc, ["identificatie", "bouwjaar", "SHAPE@"]) as cursor:
        for polygon_id, building_age, polygon in cursor:
            midpoints = [get_midpoint(part[i], part[i + 1]) for part in polygon for i in range(len(part) - 1)]
            point_array = arcpy.Array([p for p in midpoints if p is not None])
            multipoint = arcpy.Multipoint(point_array, sr)

            # 生成一个基于 UUID 的唯一名称，以便在 in_memory 中创建临时要素类
            unique_name = f"in_memory\\multipoint_{uuid.uuid4().hex}"

            arcpy.CopyFeatures_management(multipoint, unique_name)

            arcpy.analysis.Near(unique_name,
                                road_feature_class,
                                "25 Meters", "LOCATION", "ANGLE", "PLANAR", "NEAR_FID NEAR_FID;NEAR_DIST NEAR_DIST;NEAR_X NEAR_X;NEAR_Y NEAR_Y;NEAR_ANGLE NEAR_ANGLE")

            if field_exists(unique_name, field_name_list[0]):
                with arcpy.da.SearchCursor(unique_name, field_name_list) as cursor2:
                    for row in cursor2:
                        NEAR_DIST, lng, lat, angle = row

                        if NEAR_DIST != -1:
                            results.append({
                                "polygon_id": polygon_id,
                                "building_age": building_age,
                                "NEAR_DIST": NEAR_DIST,
                                "lat": lat,
                                "lng": lng,
                                "heading": transform_angle(angle)
                            })
            # 操作完成后，删除 in-memory feature 释放内存
            arcpy.Delete_management(unique_name)

            # 更新计数器
            processed_count += 1
            if processed_count % 200 == 0:  # 每处理200个多边形，更新进度
                print(f"{feature_class}处理进度：{processed_count} of {total_polygons} polygons...")

    # 存储所有结果到CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode='w', header=False, index=False)


if __name__ == '__main__':
    arcpy.env.workspace = "建筑风格和年代深度学习.gdb"
    arcpy.env.overwriteOutput = True

    # 列出简化和拆分后的数据集中的所有要素类
    polygon_feature_classes = arcpy.ListFeatureClasses("Amsterdam_buildings_subset_*", feature_type="Polygon")
    print(polygon_feature_classes)

    road_feature_class = "./Amsterdam_road.gpkg/main.edges"  # 路网数据

    # 不使用多进程，而是逐个处理 feature classes，多线程会出现文件锁定的问题
    for feature_class in polygon_feature_classes[:1]:
        csv_path = f"..\\data\\output\\{feature_class}_point.csv"  # 数据量太大，分数据集保存的，可以提前进行后续的处理了
        main(feature_class, road_feature_class, csv_path)
