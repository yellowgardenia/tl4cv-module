# TensorBoard
TensorBoard为TensorFlow提供了一个可视化方式，它可以把复杂的神经网络训练过程给可视化，可以更好地理解，调试并优化程序。

TensorBoard主要通过`tf.summary`来获取数据。`tf.summary`是一个`op`，在使用时，首先需要定义一个`writer`：

```python
writer = tf.summary.FileWriter("log/")
```


## 查看graph结构
对网络结构中的关键结构（如卷积层、池化层、全连接层等）进行命名，可以优化graph的显示效果，再将grapy传入`writer`即可显示：

```python
writer = tf.summary.FileWriter("log/", sess.graph)

writer = tf.summary.FileWriter("log/")
writer.add_graph(sess.graph)
```

## summary
A TensorFlow op that output protocol buffers containing "summarized" data

Examples:

* tf.summary.scalar
* tf.summary.image
* tf.summary.audio
* tf.summary.histogram
* **tf.summary.tensor**

1、scalar针对单一变量，如loss、accuracy等<br>
2、image可以输出图片，可用来检查CNN中每一层的输出结果<br>
3、audio可以查看生成的音频数据<br>
4、histogram可以查看数据的分布<br>
5、**tensor可以查看任意变量**<br>

## 添加summary
```python
tf.summary.scalar('acc', accuracy)
tf.summary.image('input', x_image, 3)
tf.summary.histogram(param.name, param)

#用一个 merge 把所有的 summary 合成一个 target
merged = tf.summary.merge_all()

#运行写入（i是一个计数器）
result = sess.run(merged, feed_dict=feed_dict)
writer.add_summary(result, i)
```