2021-07-20

 Lecture12: System for AI-12-Reinforcement learning systems [ [PDF](https://github.com/microsoft/AI-System/blob/main/docs/SystemforAI-12-System%20for%20Reinforcement%20Learning.pdf) ]


Lab 9 [RL Systems](https://github.com/microsoft/AI-System/blob/main/Labs/AdvancedLabs/Lab9/README.md) 

---
需要两台服务器，还要保证两边python的版本和ray的版本一致

* 安装环境依赖包 ray 和 rllib ，并测试是否安装成功。

pip install -U ray
pip install ray[rllib] 

新建一个3.7.6的环境，更新旧的不方便。
``conda create -n py376. python=3.7.6``

现在两边的服务器ray服务启动了，
>3. 配置不同的脚本，测试不同算法对应不同并行条件/不同环境下的收敛速度。至少挑选一种分布式算法，并测试其worker并行数目为4，8，16的情况下在至少两个Atari环境下的收敛情况，提交配置文件和对应的启动脚本文件

>检测依赖包是否安装成功
测试ray
``
git clone https://github.com/ray-project/ray.git 
cd ray 
python -m pytest -v python/ray/tests/test_mini.py 
``

![测试成功](https://upload-images.jianshu.io/upload_images/1016401-6e2a6eb2cb40770c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


>测试rllib
``rllib train --run=PPO --env=CartPole-v0 ``

![单机跑，有点费时间](https://upload-images.jianshu.io/upload_images/1016401-338bb9394a77f200.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


>参考的不同分布式算法对应不同环境/并行条件的配置
代码位置：Lab9/config
参考命令：
``cd Lab9 ``
``rllib train -f config/xxx-xxx.yaml``

跑不起来
一直失败
