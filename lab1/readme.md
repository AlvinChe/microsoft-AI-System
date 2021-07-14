2021-07-12

地址： [microsoft](https://github.com/microsoft)/**[AI-System](https://github.com/microsoft/AI-System)**

课程内容，讲座+实验



Lecture1：[Introduction](https://github.com/microsoft/AI-System/blob/main/docs/SystemforAI-1-2-Introduction%20and%20System%20Perspective.pdf)

Lab1：[A simple end-to-end AI example,from a system perspective](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab1/README.md)

实验内容
>1.安装依赖包。PyTorch==1.5, TensorFlow>=1.15.0
2.下载并运行PyTorch仓库中提供的MNIST样例程序。
3.修改样例代码，保存网络信息，并使用TensorBoard画出神经网络数据流图。
4.继续修改样例代码，记录并保存训练时正确率和损失值，使用TensorBoard画出损失和正确率趋势图。
5.添加神经网络分析功能（profiler），并截取使用率前十名的操作。
6.更改批次大小为1，16，64，再执行分析程序，并比较结果。
【可选实验】改变硬件配置（e.g.: 使用/ 不使用GPU），重新执行分析程序，并比较结果。

准备工作：实验环境配置|  [Setup Environment](https://github.com/microsoft/AI-System/blob/main/Labs/Prerequisites.md) |

实验步骤
配环境没什么坑，都是之前配好的环境，装了anaconda，tensorflow，这个实验的代码是基于torch的，还安装了torch，为了可视化，还需要按转tensorboardX，安装参考：[Pytorch的模型结构可视化（tensorboard）](https://zhuanlan.zhihu.com/p/58961505)


第一步：跑通示例代码：[PyTorch-MNIST Code](https://github.com/pytorch/examples/blob/master/mnist/main.py)
朴素做法，写了一个mnist.py，把代码贴上去，然后直接
python mnist.py
注意：示例代码的迭代次数太多，我这把epoch从14改成了4
```
'--epochs', type=int, default=4
```

第二步：修改样例代码，保存网络信息，并使用TensorBoard画出神经网络数据流图
参考：[MSRA AI-System课程Lab](https://zhuanlan.zhihu.com/p/387253917)

2.1 在开头倒入对应的库(参考：[Pytorch的模型结构可视化（tensorboard）](https://zhuanlan.zhihu.com/p/58961505))
```
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

#writer就相当于一个日志，保存你要做图的所有信息。第二句就是在你的项目目录下建立一个文件夹log，存放画图用的文件。刚开始的时候是空的
from tensorboardX import SummaryWriter
writer = SummaryWriter('log') #建立一个保存数据用的东西
```

2.2 保存网络信息
在def main():函数里添加
```
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    #----newly added----- 
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    model = Net().to(device)
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    # writer.add_graph(model, images)
    writer.add_graph(model.to(device), images.to(device))
    # -----end---------------
    # model = Net().to(device) 
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
```

>4.继续修改样例代码，记录并保存训练时正确率和损失值，使用TensorBoard画出损失和正确率趋势图。

``writer.add_graph(model.to(device), images.to(device))``
注意，存模型这里的代码，这里不改成这样子无法运行
修改train和test函数，注意test函数加入了epoch参数

>5.添加神经网络分析功能（profiler），并截取使用率前十名的操作。

参考：[MSRA AI-System课程Lab](https://zhuanlan.zhihu.com/p/387253917)

以上都是在修改完mnist.py代码，在终端直接```python mnist.py```后得到的结果

如何利用tensorboard展示结果
```
tensorboard --logdir /home/xxx/xx/log --host=xxx.xx.xx.xxx 
```
因为我的代码跑在服务器上，所以需要配置ip地址
注意的是``--logfix``后面只要跟日志的地址就好，不要具体到内容，
``--event_file			指定一个特定的事件日志文件``
才是指定对应的event文件


## 实验报告

### 实验环境

|  |  |  |
| --- | --- | --- |
| 硬件环境 | CPU（vCPU数目） |                 2                        |
|  | GPU(型号，数目) | 1089Ti，4 |
| 软件环境 | OS版本 | Ubuntu 16.04.5  |
|  | 深度学习框架| PyTorch|
||python包名称及版本 | Python 3.7.4 | 
|  | CUDA版本 |  Cuda release 10.1, V10.1.105
| |  |  |

---

### 实验结果

1.  模型可视化结果截图


神经网络数据流图

 ![graph长这样，怀疑哪里不对](https://upload-images.jianshu.io/upload_images/1016401-7d6d6c152a41134a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/124)

损失和正确率趋势图

 ![准确度记录下来了](https://upload-images.jianshu.io/upload_images/1016401-67003a3060c72e2d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

网络分析，使用率前十名的操作

 
在main函数中添加profile函数的调用，结果如下
```
 ---------------  
Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls  
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
cudnn_convolution            32.46%           572.903us        32.46%           572.903us        286.452us        2                
addmm                        24.13%           425.919us        24.13%           425.919us        212.960us        2                
relu                         8.97%            158.258us        8.97%            158.258us        52.753us         3                
add                          8.33%            146.979us        8.33%            146.979us        73.490us         2                
max_pool2d_with_indices      5.69%            100.502us        5.69%            100.502us        100.502us        1                
_convolution                 4.58%            80.766us         48.32%           852.742us        426.371us        2                
_log_softmax                 4.01%            70.825us         4.01%            70.825us         70.825us         1                
view                         4.01%            70.701us         4.01%            70.701us         17.675us         4                
unsigned short               2.66%            46.970us         2.66%            46.970us         23.485us         2                
select                       1.48%            26.040us         1.48%            26.040us         26.040us         1                
reshape                      0.87%            15.383us         4.88%            86.084us         21.521us         4                
conv2d                       0.54%            9.510us          49.33%           870.710us        435.355us        2                
convolution                  0.48%            8.458us          48.80%           861.200us        430.600us        2                
dropout                      0.46%            8.094us          0.46%            8.094us          4.047us          2                
log_softmax                  0.39%            6.795us          4.40%            77.620us         77.620us         1                
flatten                      0.37%            6.453us          1.25%            22.075us         22.075us         1                
max_pool2d                   0.33%            5.825us          6.02%            106.327us        106.327us        1                
---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
Self CPU time total: 1.765ms

```

2.  网络分析，不同批大小结果比较
![python mnist.py --batch-size=1](https://upload-images.jianshu.io/upload_images/1016401-69ede9bf6302150a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

![ python mnist.py --batch-size=16](https://upload-images.jianshu.io/upload_images/1016401-1417a28874b0641a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)

![python mnist.py --batch-size=32](https://upload-images.jianshu.io/upload_images/1016401-b3cb152b351fb2b2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/620)



实验报告：https://www.jianshu.com/p/20e512047459
完整代码：https://github.com/AlvinChe/microsoft-AI-System/tree/main/lab1
