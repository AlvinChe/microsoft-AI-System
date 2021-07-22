2021-07-20
 
Lecture11: System for AI-11-AutoML systems [ [PDF](https://github.com/microsoft/AI-System/blob/main/docs/SystemforAI-11-AutoML.pdf) ]


![](https://upload-images.jianshu.io/upload_images/1016401-3d072807941f6696.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/320)


Lab 8 : [AutoML](https://github.com/microsoft/AI-System/blob/main/Labs/AdvancedLabs/Lab8/README.md) 




---

其他环境和之前的实验差不多
特别软件包：安装和熟悉环境

1.  熟悉 NNI 的基本使用。阅读教程：[https://nni.readthedocs.io/en/latest/Tutorial/QuickStart.html](https://nni.readthedocs.io/en/latest/Tutorial/QuickStart.html)

测试代码结果：
![命令行环境](https://upload-images.jianshu.io/upload_images/1016401-60920bb3ea031c4e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![实验结果](https://upload-images.jianshu.io/upload_images/1016401-e45604e326ed28b7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



实验代码：
需要配置的文件
config.ym
``command: python main.py``
这里配置代码的命令
 ```
authorName: default
experimentName: cifar10
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
trainingServicePlatform: local
searchSpacePath: search_space.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
trial:
  command: python main.py
  codeDir: .
  gpuNum: 0
```

``util.py ``代码没有动
``main.py``代码增加了一行，
``import nni```
``nni.utils.merge_parameter(args, nni.get_next_parameter())``






遇到问题：TypeError: ufunc 'log' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
解决方案：更新scipy，不好使[Dispatcher stream error, tuner may have crashed](https://github.com/microsoft/nni/issues/3405)

现在的版本是2.3，卸载了nni，环境又蹦了，明天看吧。
``python3 -m pip install --upgrade nni==1.8.0  -i http://pypi.douban.com/simple --trusted-host pypi.douban.com``

