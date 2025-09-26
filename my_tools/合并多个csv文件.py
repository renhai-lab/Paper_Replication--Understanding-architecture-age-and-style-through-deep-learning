"""
CSV文件合并工具

用于将多个CSV文件合并为一个单独的文件。
"""

import pandas as pd
import glob

def merge_csv_files(dir_path, pattern, output_filename):
    """
    合并符合特定模式的多个CSV文件。
    
    参数：
        dir_path (str): 目录路径
        pattern (str): 文件匹配模式  
        output_filename (str): 输出文件名
        
    返回：
        None: 将合并结果保存到文件
    """
    # 搜索符合条件的文件并创建文件列表
    csv_list = glob.glob(f'{dir_path}/{pattern}')
    
    # 打印找到的文件列表和数量
    print("找到的文件:", csv_list)
    print("文件数量:", len(csv_list))
    
    # 创建一个空的DataFrame来存储所有合并的数据
    combined_df = pd.DataFrame()
    
    # 通过循环读取每个文件并将其内容合并到一个DataFrame中
    for file_path in csv_list:
        # 读取CSV文件（如果您的CSV文件有标题行（列名），请删除header=None）
        df = pd.read_csv(file_path, encoding='utf-8', header=None)
        print(f"文件 {file_path} 的形状:", df.shape)
        
        # 合并数据
        combined_df = pd.concat([combined_df, df], axis=0)
        
        # 打印当前合并后的DataFrame的大小
        print("当前合并后的数据形状:", combined_df.shape)
    
    # 打印最终合并后的DataFrame的大小
    print("最终合并后的数据形状:", combined_df.shape)
    
    # 将合并后的数据保存到一个新的CSV文件中
    output_path = f'{dir_path}/{output_filename}'
    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"合并完成，已保存到: {output_path}")

if __name__ == "__main__":
    # 设定目录路径
    dir_path = '../data/output'
    
    # 合并街景URL文件
    merge_csv_files(dir_path, 'street_view_url_part*.csv', 'street_view_url_all.csv')
