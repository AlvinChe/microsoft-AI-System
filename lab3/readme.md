2021-07-13
地址： [microsoft](https://github.com/microsoft)/**[AI-System](https://github.com/microsoft/AI-System)**
---

Lecture 3：[Computation frameworks for DNN](https://github.com/microsoft/AI-System/blob/main/docs/SystemforAI-3-Framework.pdf)
主要讲了
1. Tensor概念
2.DAG图
3.反向传播和自动求导
4. 图执行和调度
5.静态图vs动态图
6. 硬件支持

两个概念，我一直没弄清楚的
基本数据结构：Tensor
基本运算单元：Operator
计算内核（kernel）是什么

![基本概念](https://upload-images.jianshu.io/upload_images/1016401-fb41d8916ca2fd80.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720)

![基本概念2](https://upload-images.jianshu.io/upload_images/1016401-65e571517be983f4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

4. 图执行和调度
![GEMM自动融合](https://upload-images.jianshu.io/upload_images/1016401-e16f013a73bea187.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![并发流程](https://upload-images.jianshu.io/upload_images/1016401-453a86392ae563bf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![kernel是什么](https://upload-images.jianshu.io/upload_images/1016401-bb66a0080ecafaa8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

总结
![总结](https://upload-images.jianshu.io/upload_images/1016401-b9c223ae09e7d068.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![总结2](https://upload-images.jianshu.io/upload_images/1016401-26e7ea38a0cf16fd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


---

Lab 3 CUDA实现和优化
地址：[CUDA implementation](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab3/README.md)

实验目标
>1.理解PyTorch中Linear张量运算的计算过程，推导计算公式(同[Lab2](https://www.jianshu.com/p/4268f9e1c55b))
2.了解GPU端加速的原理，CUDA内核编程和实现一个kernel的原理
3.实现CUDA版本的定制化张量运算
	>>3.1编写.cu文件，实现矩阵相乘的kernel
	3.2在上述.cu文件中，编写使用cuda进行前向计算和反向传播的函数
	3.3基于C++ API，编写.cpp文件，调用上述函数，实现Linear张量运算的前向计算和反向传播。
	3.4将代码生成python的C++扩展
	3.5使用基于C++的函数扩展，实现自定义Linear类模块的前向计算和反向传播函数
	3.6运行程序，验证网络正确性


2.了解GPU端加速的原理，CUDA内核编程和实现一个kernel的原理
难点：cuda编程
概念扫盲：[CUDA编程入门极简教程 - 小小将的文章 - 知乎](https://zhuanlan.zhihu.com/p/34587739)
cude编程：[Nvdia示例](https://developer.nvidia.com/zh-cn/blog/easy-introduction-cuda-c-and-c/)

cuda 测试代码from [Nvdia示例](https://developer.nvidia.com/zh-cn/blog/easy-introduction-cuda-c-and-c/)
 cuda 性能数据 nvprof 
遇到的问题``The user does not have permission to profile on the target device``
解决方案：[靠谱](https://blog.csdn.net/Whisper321/article/details/103050888)

本次实验的内容
先把[Lab3](https://github.com/microsoft/AI-System/tree/main/Labs/BasicLabs/Lab3)的代码copy到实验环境中
进入extend文件夹
输入``python setup.py install --user``
注意readme.md的部分写错了，漏了setup的后缀

但是这的setup代码出错了，
``error: ‘TORCH_CHECK’ was not declared in this scope``
还没找到解决方案

所以本次实验暂告失败
虽然代码没跑起来，但是学习了cuda编程的基础内容。

代码：没有
笔记：https://www.jianshu.com/p/2a4c0ab9a864




