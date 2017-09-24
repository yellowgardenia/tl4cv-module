# 图像预处理
原始数据在输入网络进行训练之前，通常需要进行预处理或者扩增。

## [对tensor进行预处理](prepro4tensor.py)
这个模块可以在训练或测试时，对输入的tensor进行一系列预处理操作。你可以通过编辑图像预处理列表（txt）来对这些操作进行设置。当你需要改变你的预处理流程时，只需对图像预处理列表进行修改即可。

### 图像预处理列表
在定义时，需要首先使用`train`和`test`来区分预处理命令作用时机，后跟所使用的命令。每一条预处理命令需要遵循语法：

`func`, `param1`=`val1`, `param2`=`val2` ...

其中，`func`为预处理函数的名称，后跟所需要的参数字典，如果不添加参数字典则使用默认参数，如以下例子：

> preprocess_list.txt

```
  train
  random_crop
  random_flip_h
  reduce_mean, mean=90.
  test
  crop_central
  reduce_mean, mean=90.
```

### 预处理命令
预处理操作 | 函数名 | 参数
:---------:|:------:|:----:
中心裁剪 | crop_central | size（输出图像尺寸）
随机裁剪 | random_crop | size（输出图像尺寸）
随机水平翻转 | random_flip_h | -
随机竖直翻转 | random_flip_v | -
随机亮度变化 | random_brightness | max_delta
随机对比度变化 | random_contrast | lower, upper
标准化 | standardization | -
减去均值 | reduce_mean | mean（图像均值）
缩放 | resize | size（输出图像尺寸）
