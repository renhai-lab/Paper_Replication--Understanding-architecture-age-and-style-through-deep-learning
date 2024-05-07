'''
说明文章：[Part3.获取高质量的阿姆斯特丹建筑立面图像（下）——《通过深度学习了解建筑年代和风格》](https://cdn.renhai-lab.tech/archives/Understanding_architecture_age_and_style_through_deep_learning_part3-2)
'''

import concurrent.futures
import platform
import threading
import time
import os
import glob
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from tqdm import tqdm
from pypushdeer import PushDeer
from fake_useragent import UserAgent
from PIL import Image

# 全局安全锁
file_lock = threading.Lock()

def get_webdriver(cpu_arch, headless):
    """根据cpu架构初始化浏览器"""
    options = Options()
    service = None
    if cpu_arch == 'amd64':
        service = Service(ChromeDriverManager().install())
    elif cpu_arch == 'arm64':
        chromedriver_path = "/usr/bin/chromedriver"
        service = Service(chromedriver_path)
    else:
        raise ValueError(f'Unsupported CPU architecture: {cpu_arch}')

    # Common browser options
    options.add_argument('--headless') if headless else None
    options.add_argument('lang=zh_CN.UTF-8')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')  # To resolve the DevToolsActivePort file error
    options.add_argument(f'user-agent={UserAgent().chrome}')
    options.add_argument("--start-maximized")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--color-depth=24")
    # options.add_argument('--proxy-server=http://192.168.0.118:7890')

    # 启动webdriver
    driver = webdriver.Chrome(service=service, options=options)

    return driver

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    print(f"创建文件夹{path}成功")


def clip_image(image_path):
    """利用pillow对图片进行裁剪"""
    image = Image.open(image_path)
    return image.crop((470, 50, 1450, 1030)).resize((512, 512))


def save_image(origin_path, output_folder, image, building_age):
    """分类保存图片"""
    with file_lock:
        label = judge_label(building_age)
        new_path = os.path.join(output_folder, label, os.path.basename(origin_path))
        image.save(new_path)
        print(f"保存图片{new_path}成功")


def judge_label(year):
    """根据输入的建筑年代，判断其所属的标签（共九类）"""
    try:
        year = int(year)
        if year <= 1652:
            return "pre-1652"
        elif 1653 <= year <= 1705:
            return "1653–1705"
        elif 1706 <= year <= 1764:
            return "1706–1764"
        elif 1765 <= year <= 1845:
            return "1765–1845"
        elif 1846 <= year <= 1910:
            return "1846–1910"
        elif 1911 <= year <= 1943:
            return "1911–1943"
        elif 1944 <= year <= 1977:
            return "1944–1977"
        elif 1978 <= year <= 1994:
            return "1978–1994"
        elif 1995 <= year <= 2023:  # 作者是2020年写的，我们采用今年的
            return "1995–2023"
    except:
        print(f"!!!!!输入的年代{year}不是整数!!")
        return "未知年代"


def get_cpu_arch():
    machine = platform.machine()
    print(f"当前机器的cpu架构是{machine}")
    if machine == 'x86_64' or machine == 'AMD64':
        return 'amd64'
    elif machine == 'aarch64':
        return 'arm64'
    else:
        print(f"不支持的cpu架构{machine}")
        return None

# 定义获取截图的主程序：
def get_screenshot(index, url, df, subset_name, pushdeer, cpu_arch, driver_implicity_wait_time, delay_time):

    """获取截图"""
    if index >= 0:  # 用来跳过已经获取的街景
        try:
            # 初始化浏览器
            print(f"Initializing browser for {cpu_arch}")
            # ！！！！ 单独为每一个线程创建一个driver，避免多线程共用一个driver导致的问题
            driver = get_webdriver(cpu_arch, True)
            driver.get(url)
            driver.implicitly_wait(driver_implicity_wait_time)
            time.sleep(delay_time)

        except Exception as e:
            print(f"出现异常{e}")
            driver.quit()


        id = df.loc[index, 'identificatie']  # polygon_id
        building_age = df.loc[index, 'bouwjaar']  # building_age
        date = df.loc[index, 'date']  # date

        # 截图
        driver.save_screenshot(
            origin_path := f"../data/GSV/origin/subset_{subset_name}--{index}--{id}--{date}.png")  # := 在表达式中同时进行变量赋值和返回赋值的结果。

        # 处理图片
        image = clip_image(origin_path)

        # 保存照片
        output_folder = "../data/GSV/clip"
        save_image(origin_path, output_folder, image, building_age)

        # 删除原来的照片
        os.remove(origin_path)

        if (index + 1) % 200 == 0:
            pushdeer.send_text(f"repo selenium windows Got {index + 1} GSVs",
                               desp=f"now processing subset_{subset_name} now index is {index}")

        driver.quit()
        


    else:
        print(f"跳过已经获取的街景{url}")
        pass


if __name__ == "__main__":
    # 要创建的文件夹列表
    folders = [
        "../data/GSV/origin",
        "../data/GSV/clip"
    ]
    # 建筑年代：分为9个文件夹
    facade_photos = [
        "pre-1652",
        "1653–1705",
        "1706–1764",
        "1765–1845",
        "1846–1910",
        "1911–1943",
        "1944–1977",
        "1978–1994",
        "1995–2023",
        "未知年代"
    ]

    # 将facade_photos中的每个标签添加到输出文件夹中
    folders.extend(
        [os.path.join("../../data/GSV/clip", label) for label in facade_photos])  # extend 是列表的一个方法，它允许你添加另一个列表的所有元素到当前列表中。

    # 创建所有文件夹
    for folder in folders:
        mkdir(folder)

    # 获取cpu架构
    cpu_arch = get_cpu_arch()

    # 主程序
    # 读取url
    #    pathlists = glob.glob("street_view_url_*.csv")
    pathlists = [
        #    'street_view_url_23.csv',
        'street_view_url_part2.csv',
        #    'street_view_url_27.csv', 'street_view_url_29.csv', 'street_view_url_21.csv'
    ]
    print(pathlists)

    for path in pathlists:
        subset_name = os.path.basename(path).split(".")[0].split("_")[-1]
        print(subset_name)

        df = pd.read_csv(path, header=0, encoding='utf-8')
        print(df.head())

        urls = df['url'].tolist()
        print(urls[:5])

        # 计数和通知
        pushdeer = PushDeer(pushkey="PDU22018TBKAygHi6CfrjI99HYdp6H2U4JVRVkOXQ")

        # 创建线程池

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:

            # 循环获取街景
            futures = []
            for index, url in tqdm(enumerate(urls), desc="正在获取街景", ncols=100):
                # print(f"正在获取街景{url},index={index}")
                # 提交任务到线程池
                future = executor.submit(get_screenshot, index, url, df, subset_name, pushdeer,
                                         cpu_arch=cpu_arch,
                                         driver_implicity_wait_time=5,
                                         delay_time=random.randint(6, 8))  # 随机延迟

                futures.append(future)

            # 等待所有任务完成
            concurrent.futures.wait(futures)

            # 处理完成通知
            pushdeer.send_text(f"selenium subset_{subset_name} Got all GSVs",
                               desp=f" ")
