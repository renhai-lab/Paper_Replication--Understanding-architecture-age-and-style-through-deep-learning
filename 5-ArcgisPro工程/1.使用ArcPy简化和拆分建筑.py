import arcpy


def split_dataset(input_polygon_fc, group_size):
    """拆分数据集,
    :param input_polygon_fc: 输入要素类
    :param group_size: 每个要素类的要素数量
    :return: 无返回值"""
    # 获取输入要素类的所有 OBJECTID
    objectids = [row[0] for row in arcpy.da.SearchCursor(input_polygon_fc, "OBJECTID")]

    for i in range(0, len(objectids), group_size):
        # 获取当前组的 OBJECTID 范围
        current_ids = objectids[i:i + group_size]

        # 构建 SQL 查询以选择当前组的 OBJECTID
        where_clause = f"OBJECTID IN ({','.join(map(str, current_ids))})"

        # 定义输出要素类的名称
        output_feature_class = f"Amsterdam_buildings_subset_{i // group_size + 1}"  #

        # 使用 Select_analysis 复制当前组到新的要素类
        arcpy.Select_analysis(input_polygon_fc, output_feature_class, where_clause)

        print(f"成功拆分数据集： {output_feature_class}")

    print("完成拆分数据集任务！")

def simplify_feature(input_polygon_fc, out_polygon_feature_class, tolerance):
    arcpy.cartography.SimplifyBuilding(input_polygon_fc,
                                       out_polygon_feature_class,
                                       tolerance, None, "NO_CHECK", None, "NO_KEEP")

if __name__ == '__main__':
    arcpy.env.workspace = "建筑风格和年代深度学习.gdb"
    arcpy.env.overwriteOutput = True

    input_polygon_fc = "Amsterdam_buildings_Project"  # 只包含98要素的测试集
    simplify_fc = input_polygon_fc + "simplify"

    # 简化
    tolerance = "3 Meters"
    simplify_feature(input_polygon_fc, simplify_fc, tolerance)

    # 拆分
    group_size = 5000
    split_dataset(simplify_fc, group_size)
