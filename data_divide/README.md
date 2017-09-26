# 数据分组
对于深度学习（机器学习）任务来说，通常需要将原始数据集分为训练集和测试集。分组的方式有很多，可以根据具体数据集的情况以及实验需求来选择。

## 单一标签的数据集
单一标签的数据集主要应用在对单一目标的分类以及分割任务中，为了保证训练集和测试集的均衡，通常须对于每一类别进行相同的分组方法。

例如，对于一个数据集有三类，数据量分别为：100（1类），80（2类），90（3类），使用二分法（ratio=0.5）的结果为：

类别 | 1类 | 2类 | 3类
:---:|:---:|:---:|:---:
1组 | 50 | 40 | 45
2组 | 50 | 40 | 45

### App
可以使用[sl_divide_method](sl_divide_method.py)中的分组函数对数据进行处理，也可以直接使用[sl_divide_app](sl_divide_app.py)

```
  python sl_divide_app.py \
      --divide_method="divide_into_two_parts, ratio=0.5"
	  
  python sl_divide_app.py \
      --divide_method="divide_into_n_parts, r=5:1:4"

  python sl_divide_app.py \
      --divide_method="k_fold_cross_validation, k=5"
```

### divide_method
分组方法 | 函数名 | 参数
:-------:|:------:|:----:
二分法 | divide_into_two_parts | ratio（组1比例[0.5]）
n分法 | divide_into_n_parts | r（n组比例[5:1:4]）
k折交叉验证 | k_fold_cross_validation | k（k fold[5]）