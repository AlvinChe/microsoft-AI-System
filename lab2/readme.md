2021-07-12
地址： [microsoft](https://github.com/microsoft)/**[AI-System](https://github.com/microsoft/AI-System)**

课程内容，讲座+实验

Lecture2：[System perspective of Systems for AI](https://github.com/microsoft/AI-System/blob/main/docs/SystemforAI-1-2-Introduction%20and%20System%20Perspective.pdf)

Lecture2 对Systems for AI的大概介绍
课程安排，先修知识
最有价值的几张PPT
![技术栈](https://upload-images.jianshu.io/upload_images/1016401-dc859a0c18454998.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![技术生态](https://upload-images.jianshu.io/upload_images/1016401-c7ff649d180b17d7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![详细图](https://upload-images.jianshu.io/upload_images/1016401-96b378f04bf48302.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![难点](https://upload-images.jianshu.io/upload_images/1016401-de3ab7e62a5a50b3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![AI system](https://upload-images.jianshu.io/upload_images/1016401-b04462e37605a6e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

Lab2：[Customize operators](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab2/README.md) 定制一个新的张量运算

![实验要求](https://upload-images.jianshu.io/upload_images/1016401-0e6b9c161fd00bbf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

1-4.在MNIST的模型样例中，选择线性层（Linear）张量运算进行定制化实现
线性张量：$y = ax$这样的算式？

前向传播：
output = input*weights
反向传播：
output= grad_output * weights^T
grad_weight = input^T * grad_output

数学推倒参考：[Numpy实现神经网络框架(3)——线性层反向传播推导及实现](https://zhuanlan.zhihu.com/p/67854272)
这里还是谜哈

一开始没看懂“基本单位：Function和Module”指什么，然后去看了答案，是实现以 ``torch.autograd.Function``,``nn.Module ``为基类，继承之后实现自己功能的类
[代码参考](https://github.com/microsoft/AI-System/blob/main/Labs/BasicLabs/Lab2/mnist_custom_linear.py)

```
#继承torch.autograd.Function，写一个linear的函数
class myLinearFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight):
        # 存下input 和weight
        ctx.save_for_backward(input, weight)
        # y = ax + b, b呢，可以通过参数加上，参考这里的实现(https://zhuanlan.zhihu.com/p/67854272)
        # 注意这里的weight加了转置
        output = input.mm(weight.t())
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        #获得正向传播的input和weight
        input, weight = ctx.saved_tensors
        # 这一句没看明白
        grad_input = grad_weight = None
        # 这里的注释也没明白
        #if ctx.needs_input_grad[0]:
        # grad_input =  grad_output*weights
        grad_input = grad_output.mm(weight)
        #if ctx.needs_input_grad[1]:
        # grad_weight = grad_output^T * input
        grad_weight = grad_output.t().mm(input)
        return grad_input, grad_weight

#继承nn.Module，写一个linear Module
# 输入输出定义里，参数为什么是output_features, input_features两个的在一起的tensor？
class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return myLinearFunction.apply(input, self.weight)
```

然后在代码中将第一行替换成自己的Linear
```
        # self.fc2 = nn.Linear(128, 10)
        self.fc2 = myLinear(128, 10)
```

其他的地方不需要变动

5.实现C++版本的定制化张量运算
c++我不太熟悉，还是直接看答案的，区别就是计算过程换成了``mylinear_cpp.forward``

```
class myLinearFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        #output = input.mm(weight.t())
        output = mylinear_cpp.forward(input, weight)
        return output[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        #grad_input = grad_weight = None
        #grad_input = grad_output.mm(weight)
        #grad_weight = grad_output.t().mm(input)
        grad_input, grad_weight = mylinear_cpp.backward(grad_output, input, weight)
        return grad_input, grad_weight

class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return myLinearFunction.apply(input, self.weight)
```
c++实现代码
```c++
#include <torch/extension.h>

#include <iostream>
#include <vector>

std::vector<torch::Tensor> mylinear_forward(
    torch::Tensor input,
    torch::Tensor weights) 
{
   // 前向传播，就input*weights
    auto output = torch::mm(input, weights.transpose(0, 1));
    //返回结果
    return {output};
}

//反向传播
std::vector<torch::Tensor> mylinear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights
    ) 
{
    // 这里没看懂
    auto grad_input = torch::mm(grad_output, weights);
    auto grad_weights = torch::mm(grad_output.transpose(0, 1), input);

    return {grad_input, grad_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mylinear_forward, "myLinear forward");
  m.def("backward", &mylinear_backward, "myLinear backward");
}
```
这里的推导和代码是不匹配的，但是我不像细细推理谁是谁了，大家感兴趣的可以仔细看看



![实验结果](https://upload-images.jianshu.io/upload_images/1016401-a387cfcecdcb41d9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

----
完整代码：https://github.com/microsoft/AI-System/tree/main/Labs/BasicLabs/Lab2
实验报告：https://www.jianshu.com/p/4268f9e1c55b
