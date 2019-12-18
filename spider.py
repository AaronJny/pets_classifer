# -*- coding: utf-8 -*-
# @File    : spider.py
# @Author  : AaronJny
# @Time    : 2019/12/16
# @Desc    : 从谷歌下载指定图片
from gevent import monkey

monkey.patch_all()
import functools
import logging
import os
from bs4 import BeautifulSoup
from gevent.pool import Pool
import requests
import settings

# 设置日志输出格式
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

# 搜索关键词字典
keywords_map = settings.IMAGE_CLASS_KEYWORD_MAP

# 图片保存根目录
images_root = settings.IMAGES_ROOT
# 每个类别下载多少页图片
download_pages = settings.SPIDER_DOWNLOAD_PAGES
# 图片编号字典，每种图片都从0开始编号，然后递增
images_index_map = dict(zip(keywords_map.keys(), [0 for _ in keywords_map]))
# 图片去重器
duplication_filter = set()

# 请求头
headers = {
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'zh-CN,zh;q=0.9',
    'user-agent': 'Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.76 Mobile Safari/537.36',
    'accept': '*/*',
    'referer': 'https://www.google.com/',
    'authority': 'www.google.com',
}


# 重试装饰器
def try_again_while_except(max_times=3):
    """
    当出现异常时，自动重试。
    连续失败max_times次后放弃。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_cnt = 0
            error_msg = ''
            while error_cnt < max_times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    error_cnt += 1
            if error_msg:
                logging.error(error_msg)

        return wrapper

    return decorator


@try_again_while_except()
def download_image(session, image_url, image_class):
    """
    从给定的url中下载图片，并保存到指定路径
    """
    # 下载图片
    resp = session.get(image_url, timeout=20)
    # 检查图片是否下载成功
    if resp.status_code != 200:
        raise Exception('Response Status Code {}！'.format(resp.status_code))
    # 分配一个图片编号
    image_index = images_index_map.get(image_class, 0)
    # 更新待分配编号
    images_index_map[image_class] = image_index + 1
    # 拼接图片路径
    image_path = os.path.join(images_root, image_class, '{}.jpg'.format(image_index))
    # 保存图片
    with open(image_path, 'wb') as f:
        f.write(resp.content)
    # 成功写入了一张图片
    return True


@try_again_while_except()
def get_and_analysis_google_search_page(session, page, image_class, keyword):
    """
    使用google进行搜索，下载搜索结果页面，解析其中的图片地址，并对有效图片进一步发起请求
    """
    logging.info('Class:{} Page:{} Processing...'.format(image_class, page + 1))
    # 记录从本页成功下载的图片数量
    downloaded_cnt = 0
    # 构建请求参数
    params = (
        ('q', keyword),
        ('tbm', 'isch'),
        ('async', '_id:islrg_c,_fmt:html'),
        ('asearch', 'ichunklite'),
        ('start', str(page * 100)),
        ('ijn', str(page)),
    )
    # 进行搜索
    resp = requests.get('https://www.google.com/search', params=params, timeout=20)
    # 解析搜索结果
    bsobj = BeautifulSoup(resp.content, 'lxml')
    divs = bsobj.find_all('div', {'class': 'islrtb isv-r'})
    for div in divs:
        image_url = div.get('data-ou')
        # 只有当图片以'.jpg','.jpeg','.png'结尾时才下载图片
        if image_url.endswith('.jpg') or image_url.endswith('.jpeg') or image_url.endswith('.png'):
            # 过滤掉相同图片
            if image_url not in duplication_filter:
                # 使用去重器记录
                duplication_filter.add(image_url)
                # 下载图片
                flag = download_image(session, image_url, image_class)
                if flag:
                    downloaded_cnt += 1
    logging.info('Class:{} Page:{} Done. {} images downloaded.'.format(image_class, page + 1, downloaded_cnt))


def search_with_google(image_class, keyword):
    """
    通过google下载数据集
    """
    # 创建session对象
    session = requests.session()
    session.headers.update(headers)
    # 每个类别下载10页数据
    for page in range(download_pages):
        get_and_analysis_google_search_page(session, page, image_class, keyword)


def run():
    # 首先，创建数据文件夹
    if not os.path.exists(images_root):
        os.mkdir(images_root)
    for sub_images_dir in keywords_map.keys():
        # 对于每个图片类别都创建一个单独的文件夹保存
        sub_path = os.path.join(images_root, sub_images_dir)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
    # 开始下载，这里使用gevent的协程池进行并发
    pool = Pool(len(keywords_map))
    for image_class, keyword in keywords_map.items():
        pool.spawn(search_with_google, image_class, keyword)
    pool.join()


if __name__ == '__main__':
    run()
