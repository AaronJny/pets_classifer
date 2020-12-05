# 介绍

这是一个从零开始构建的深度学习小项目，提供猫、狗、鼠、兔四种宠物的识别服务。

共包含如下几部分：

- 爬虫
    - 从网络上下载宠物图片，构建训练用的数据集
    - gevent + requests + beautifulsoup4
- 深度学习模型
    - 鉴于我们的数据比较少，这部分需要做迁移学习
    - TensorFlow 2.0 + DenseNet121
- Web服务
    - 使用Web提供宠物图片识别服务
    - 后端 Flask
    - 前端 Vue.js + Element UI

# 快速开始

1.如何启动web服务？

- 切换到项目根目录下;
- 安装requirements.txt中的依赖;
- 通过`python3 app.py`启动web服务;
- 浏览器打开 http://localhost:5000/ 访问index页面。

> 项目现在支持使用docker启动啦~你可以选择拉取我构建好的docker镜像简单体验一下。
> 
> 安装好docker，然后在终端输入：
> 
> `docker run --rm -p 5000:5000 --name pet aaronjny/pets_classifier`
> 
> 等镜像拉取完成、并成功运行了容器后，可以在浏览器输入`http://localhost:5000/`进行体验。

2.如何训练自己的模型？

- 切换到项目根目录下;
- 安装requirements.txt中的依赖;
- 通过`python3 spider.py`启动爬虫程序;
- 人工筛选，删除不正确的图片;
- 修改`settings.py`里的相关参数;
- 通过`python3 train.py`启动训练脚本。


# 更多信息

请访问博文:

[从零开始编写一个宠物识别系统（爬虫、模型训练和调优、模型部署、Web服务）](https://www.aaronjny.com/articles/2019/12/17/1576592367309.html) (https://www.aaronjny.com/articles/2019/12/17/1576592367309.html)

或

[从零开始编写一个宠物识别系统（爬虫、模型训练和调优、模型部署、Web服务）](https://blog.csdn.net/aaronjny/article/details/103605988) (https://blog.csdn.net/aaronjny/article/details/103605988)