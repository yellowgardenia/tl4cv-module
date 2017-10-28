# Params for tensorlayer
TensorLayer（TensorFlow）中对权重参数进行操作的样例代码。

## **Freeze** some layers
想要冻结某些参数，首先需要得到训练的参数列表。第一种方法是使用网络中的`all_params`。`all_params`可以返回网络的全部权重参数变量，选择其中需要训练的即可。例如要选择第5个变量之后的进行训练，可以用：

```python
train_params = network.all_params[5:]
```

第二种方法是使用变量的名字。比如要选择所有名字中前缀为`dense`的变量进行训练，可以用：

```python
train_params = tl.layers.get_variables with names('dense', train_only=True, printable=True)

fc_p = [v for v in all_params if 'fc' in v.name]
fc_wp = [v for v in fc_p if 'weights' in v.name]
fc_bp = [v for v in fc_p if 'biases' in v.name]
```

得到参数列表之后，定义如下优化器即可：

```python
train_op = tf.train.AdamOptimizer(0.001).minimize(cost, var_list=train_params)
```
