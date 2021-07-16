2021-07-15

地址： [microsoft](https://github.com/microsoft)/**[AI-System](https://github.com/microsoft/AI-System)**



Lecture7 :[Scheduling and resource management system](https://github.com/microsoft/AI-System/blob/main/docs/SystemforAI-7-Platform.pdf)

Lecture8: [Inference systems](https://github.com/microsoft/AI-System/blob/main/docs/SystemforAI-8-Inference.pdf)
这两个ppt都很好，把最近看的文章都串起来了，知道每个部分在系统的什么位置。


Lab 5 [Configure containers for customized training and inference](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab5/README.md)



##### Docker测试

Dockerfile
```
#继承自哪一个基础镜像
FROM nvidia/cuda:10.1-cudnn7-devel

#创建镜像中的文件夹，用于存储代码或文件
RUN mkdir -p /src/app

# WORKDIR指令设置Dockerfile中的任何RUN，CMD，ENTRPOINT，COPY和ADD指令的工作目录
WORKDIR /src/app

# 拷贝本地文件到Docker镜像中相应目录
COPY pytorch_mnist_basic.py /src/app

# 需要安装的依赖
RUN apt-get update && apt-get install wget -y \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \ 
    && bash miniconda.sh -b -p /opt/conda 
ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# 容器启动命令
CMD [ "python", "pytorch_mnist_basic.py" ]



 #### 1. 熟悉环境

参考   [2\. 运行你的第一个容器 - 内容，步骤，作业](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab5/alpine.md)

 #### 2.Docker部署PyTorch训练程序
![打包完成](https://upload-images.jianshu.io/upload_images/1016401-f22f59e946b12e2d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

构建镜像
 ``sudo docker build -f Dockerfile -t train_dl . ``

![构建成功](https://upload-images.jianshu.io/upload_images/1016401-9d8b7267dec0e332.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

启动镜像
``sudo docker run --name training train_dl ``

![正在运行](https://upload-images.jianshu.io/upload_images/1016401-aa74f2ea03dd499d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

![成功训练](https://upload-images.jianshu.io/upload_images/1016401-0a8e238d5d15a84a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

![epoch 4](https://upload-images.jianshu.io/upload_images/1016401-3e0be32fca204496.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

训练结果
```
Test set: Average loss: 0.0262, Accuracy: 9918/10000 (99%)


/opt/conda/lib/python3.8/site-packages/torchvision/datasets/mnist.py:502: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /opt/conda/conda-bld/pytorch_1616554786078/work/torch/csrc/utils/tensor_numpy.cpp:143.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

```

#### 3. Docker部署PyTorch推理程序

写dockerfile.infer.gpu,代码都有
写  dockerd-entrypoint.sh ，代码都有
写 config.properties，代码都有
执行命令``sudo  docker build --file Dockerfile.infer.gpu -t torchserve:0.1-gpu .``
![成功](https://upload-images.jianshu.io/upload_images/1016401-47c81ec0af6e3d02.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

![挺占地方](https://upload-images.jianshu.io/upload_images/1016401-fe1db71c1d620d3f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

启动镜像
``sudo docker run --rm -it  -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest``

这里加了``--gpu``的没有成功，故而删掉了这个标签

ps：服务器给我搞没空间了，花了2小时要权限去删东西

![成功](https://upload-images.jianshu.io/upload_images/1016401-0c5a5dcc86ecfc97.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

![image.png](https://upload-images.jianshu.io/upload_images/1016401-924dc9fbf96ae455.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1040)
![image.png](https://upload-images.jianshu.io/upload_images/1016401-fdab92a2d348c60a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/320)

部署模型，进行推理
卡在了权限这一步，为什么在docker镜像中，还需要权限呢？
没有解决这个问题

解决方案：[解决apt-get /var/lib/dpkg/lock-frontend 问题 - 北麓牧羊人的文章 - 知乎](https://zhuanlan.zhihu.com/p/126538251) 不好使

解决方案：  ``sudo docker exec -u root -it id /bin/bash``
[docker 容器内权限问题](https://www.jianshu.com/p/36beca7076a8)

![模型下载成功](https://upload-images.jianshu.io/upload_images/1016401-18adfb2dba8c6d7f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

使用model archiver进行模型归档
代码没有这个工具

解决方案
现在镜像中装``apt-get install pip``
然后 ``git clone https://github.com/pytorch/serve.git``

``torch-model-archiver --model-name densenet161 --version 1.0 --model-file ./serve/examples/image_classifier/densenet_161/model.py --serialized-file /home/model-server/model-store/densenet161-8d451a50.pth --export-path /home/model-server/model-store --extra-files ./serve/examples/image_classifier/index_to_name.json --handler image_classifier``

![生成推理模型](https://upload-images.jianshu.io/upload_images/1016401-e4b0fb2decd482d7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![启动torchserve](https://upload-images.jianshu.io/upload_images/1016401-4db6f696801edfdf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![成功](https://upload-images.jianshu.io/upload_images/1016401-5f512c20007c3d24.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

代码：[github](https://github.com/microsoft/AI-System/tree/main/Labs/BasicLabs/Lab5)
笔记：[[Microsoft/AI-System]微软AI系统Lab5+Lecture7+Lecture8](https://www.jianshu.com/p/97ace8b501df)



















