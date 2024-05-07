import pandas as pd
import glob

# 设定目录路径
dir_path = '../data/output'

# 搜索符合条件的文件并创建文件列表
csv_list = glob.glob(f'{dir_path}/street_view_url_part*.csv')

# 打印找到的文件列表和数量
print(csv_list)
print(len(csv_list))

# 创建一个空的DataFrame来存储所有合并的数据
combined_df = pd.DataFrame()

# 通过循环读取每个文件并将其内容合并到一个DataFrame中
for file_path in csv_list:
    # 读取CSV文件
    df = pd.read_csv(file_path, encoding='utf-8', header=None)  # 如果您的CSV文件有标题行（列名），请删除header=None
    print(df.shape)
    # 合并数据
    combined_df = pd.concat([combined_df, df], axis=0)

    # （可选）打印当前合并后的DataFrame的大小
    print(combined_df.shape)

# 打印最终合并后的DataFrame的大小
print(combined_df.shape)

# 将合并后的数据保存到一个新的CSV文件中
combined_df.to_csv(f'{dir_path}/street_view_url_all.csv', index=False, encoding='utf-8')
