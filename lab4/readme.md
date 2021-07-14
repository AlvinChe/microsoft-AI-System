2021-07-13
地址： [microsoft](https://github.com/microsoft)/**[AI-System](https://github.com/microsoft/AI-System)**

---
Lecture 4 
[Computer architecture for Matrix computation](https://github.com/microsoft/AI-System/blob/main/docs/SystemforAI-4-Computer%20architecture%20for%20Matrix%20computation.pdf)

Lab 4 
[AllReduce implementation](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab4/README.md)
---

实验准备
1. 安装openmpi 
参考1:[# [Linux中openmpi配置](https://www.cnblogs.com/sdxk/p/4029850.html)
](https://www.cnblogs.com/sdxk/p/4029850.html)
我安装的是最新的4.1.1的版本
注意``make install``的时候需要加sudo
遇到``mpicc: error while loading shared libraries: libopen-pal.so.40: cannot open shared object file: No such file or directory``的问题
解决方案：``sudo ldconfig``
参考：[here](https://patrickandgarry.wordpress.com/2019/09/14/error-loading-shared-libraries-libopen-pal-so-40/)
上文的解决方案没有用，   ``apt-file search libopen-pal.so.40``找不到结果
`` mpirun -n 4 ./openmpi-4.1.1/examples/hello_c``

![安装成功，测试结果](https://upload-images.jianshu.io/upload_images/1016401-69ec62c9630382a0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2. 安装NCCL
参考：[Ubuntu NCCL安装 - benjiachong的文章 ](https://zhuanlan.zhihu.com/p/174710896)
`` ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 4``

![安装成功？](https://upload-images.jianshu.io/upload_images/1016401-7fa32d8ce6acec70.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

安装到这里，我的环境崩掉了
一直出现error，

出现``Could not load dynamic library 'libcudart.so.10.0``问题
出现问题,pip安装包时出现`` WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None))``
就不知道哪里出了问题
搞了一个晚上了，搞不定，明天再来看看

---
2021-07-14
今天试了掘金这篇文章，[Horovod安装](https://juejin.cn/post/6844904158508630023)
>horovod编译的时候需要cpu版本和GPU版本的tensorflow，要确保环境中两者都安装了，不然会触发下载最新版本的tensorflow的操作（这个不确定什么原因，但是我自己安装的时候如果没有CPU版本就自动触发下载tensorflow-2.0版本，所以我都安装了再编译horovod，如果没有这种情况则直接编译即可）
作者：lshua
链接：https://juejin.cn/post/6844904158508630023
来源：掘金
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


今天可以pip了，重新安装了horovod，现在是这个样子，算是成功了？
![没报找不到cuda的错](https://upload-images.jianshu.io/upload_images/1016401-b1f44cdfc6721332.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

示例代码：[tensorflow_mnist_estimator.py](https://github.com/horovod/horovod/tree/master/examples/tensorflow)
运行指令：``horovodrun -np 4 -H localhost:4 python tensorflow_mnist_estimator.py``
参考：[如何安装Horovod？ - sookienlane](https://zhuanlan.zhihu.com/p/63158504)
![示例代码运行ing](https://upload-images.jianshu.io/upload_images/1016401-8b9f1e17dadb4d85.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

看了一下设备情况，没用上GPU
![不知道哪里不对](https://upload-images.jianshu.io/upload_images/1016401-c59b0a9c79e56ec5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

试试这篇文章[Horovod的安装和使用 - Mario](https://zhuanlan.zhihu.com/p/78303865)里的，单机多卡

``mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
-x PATH -mca pml ob1 -mca btl ^openib python tensorflow_mnist_estimator.py ``

![好像没什么变化？](https://upload-images.jianshu.io/upload_images/1016401-c600577462d08780.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![好像还是CPU？](https://upload-images.jianshu.io/upload_images/1016401-9cc86c5d068efc8d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


换成了文章的示例代码[Horovod的安装和使用 - Mario](https://zhuanlan.zhihu.com/p/78303865)
运行命令``mpirun -np 4 -H localhost:4 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
-x PATH -mca pml ob1 -mca btl ^openib python horovod_test.py ``

![感觉差不多？](https://upload-images.jianshu.io/upload_images/1016401-3b9a0972330c9d6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![top](https://upload-images.jianshu.io/upload_images/1016401-867f27ab9f64afb3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

查了一下，看你是tensorflow的版本不对，
我的设备是cuda10.1，cudnn是7.1
参考：[这里](https://tensorflow.google.cn/install/source#linux)
应该选择tensorflow-2.1.0版本
![版本](https://upload-images.jianshu.io/upload_images/1016401-96d79e1257f6181d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

``uninstall tensrflow``
``pip install tensorflow==2.1.0 tensorflow-gpu==2.1.0``
horovod也要重新安装
``HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod[tensorflow,keras,pytorch]`` 
参考文章：[掘金-lshua](https://juejin.cn/post/6844904158508630023)

这下测试代码跑不起来了

再改成1.15的版本吧
是不是nccl版本不对？

换成1.15的版本可以跑起来了，就是好像没有用起来GPU，不知道哪里不对

![务必安装CPU版本？](https://upload-images.jianshu.io/upload_images/1016401-f09b59fb39372566.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---
开始跑实验
准备好代码,[pytorch_mnist_horovod.py](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab4/pytorch_mnist_horovod.py "pytorch_mnist_horovod.py")
`` horovodrun -np 2 --verbose python pytorch_mnist_horovod.py ``
会变成下面的命令
  ``mpirun --allow-run-as-root --tag-output -np 2 -H localhost:2 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib  -mca btl_tcp_if_include lo -x NCCL_SOCKET_IFNAME=lo  -x CONDA_DEFAULT_ENV -x CONDA_EXE -x CONDA_MKL_INTERFACE_LAYER_BACKUP -x CONDA_PREFIX -x CONDA_PROMPT_MODIFIER -x CONDA_PYTHON_EXE -x CONDA_SHLVL -x CUDA_HOME -x DERBY_HOME -x GSETTINGS_SCHEMA_DIR -x GSETTINGS_SCHEMA_DIR_CONDA_BACKUP -x HOME -x J2REDIR -x J2SDKDIR -x JAVA_HOME -x LANG -x LANGUAGE -x LD_LIBRARY_PATH -x LESSCLOSE -x LESSOPEN -x LOGNAME -x LS_COLORS -x MAIL -x MKL_INTERFACE_LAYER -x PATH -x PWD -x SHELL -x SHLVL -x SSH_CLIENT -x SSH_CONNECTION -x SSH_TTY -x TERM -x USER -x VIRTUALENVWRAPPER_SCRIPT -x XDG_DATA_DIRS -x XDG_RUNTIME_DIR -x XDG_SESSION_ID -x _ -x _CE_CONDA -x _CE_M -x _VIRTUALENVWRAPPER_API  python pytorch_mnist_horovod.py``

感动，多卡跑起来了，速度真的唰的一下就上来了
![多卡，开心](https://upload-images.jianshu.io/upload_images/1016401-80bc9e8e7c5bf3e0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
















