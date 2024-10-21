#

## 计算两向量L2距离

### 2loops
```python
def calcL2(X, Y):
    dists = np.zeros((X.shape[0], Y.shape[0]))
    diff = X[i] - Y[j]  ## Compute as a whole
    dists[i, j] = np.sum(diff ** 2)
    dists[i, j] = np.sqrt(dists[i, j])
    return dists
```

### no loop!
```python
def calcL2(X, Y):
    dists = np.zeros((X.shape[0], Y.shape[0]))
    dists = np.sqrt(np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)) #将计算过程
    return dists
```

## 数据分组，从中抽出1个test_data
```python
#这个基于数组切片
def split_data(data, num_folds, fold):
    num_val = len(data) / num_folds
    val_data = data[fold * num_val: (fold + 1) * num_val]
    train_data = np.vstack([data[:fold * num_val], data[(fold + 1) * num_val:]])
    return train_data, val_data
```

``` python
# 这个基于数组2重循环
def split_data(data, num_folds, fold):
    remaining_X = [X_train_folds[j][l] for j in range(len(X_train_folds)) if j != i for l in range(len(X_train_folds[j])) ]
    X_train_i = np.array(remaining_X)
    remaining_y = [y_train_folds[j][l] for j in range(len(y_train_folds)) if j != i for l in range(len(y_train_folds[j])) ]
    y_train_i = np.array(remaining_y)
    classifier.train(X_train_i, y_train_i)
```

## KNN不是线性分类器，至少L2距离计算就已经使其复杂

## 给第i个样本匹配y[i]的分数
高级索引：```scores[np.arange(num_train), y]```
后续广播时需要维数一致，因此要扩展成2维，即.reshape(-1, 1)就可以了

## 求梯度

类似这样，整列同时加X[i] ```dW[:, j] += X[i]```
Jacobian矩阵，每一行是一个梯度，每一列是一个变量

vectorized: if语句如何快速实现
原先：
```python
if margin > 0:
    loss += margin
    dW[:, j] += X[i]
    dW[:, y[i]] -= X[i]
```
向量化操作：
```python
binary = margins
binary[margins > 0] = 1
row_sum = np.sum(binary, axis=1)
binary[np.arange(num_train), y] = -row_sum
dW = X.T.dot(binary) # sum of all train samples
```
引入binary项，负责处理逻辑

## max()和maximum()的区别
max()是函数用于求某数组中的最大值，maximum()是numpy的广播函数，用于求两者中较大的值

## np.where(A, B, C)
- A是一个逻辑式
- B是A为True时的返回值
- C是A为False时的返回值
- 通常B, C是array或其中有一个是array，实现条件赋值

## 如何调参

- 输出loss的变化，发现是线性下降的，说明learning_rate还可以更大一些
- 如果是震荡的，那就要减小learning_rate
- 训练准确率与验证准确率之间没有差距，说明模型可能容量较低——即模型过于简单，无法捕捉数据中的潜在模式
- 训练准确率高，验证准确率低，说明过拟合，此时模型容量就已经足够大，需要通过增加正则化项等方式来减小过拟合

### 常见参数
1. 隐藏层大小：增加神经元或层的数量可以增强模型学习复杂模式的能力，但过多可能会导致过拟合。
1. 学习率：试验不同的学习率可以帮助找到收敛速度与稳定性之间的平衡。较高的学习率可能导致快速收敛，但风险是可能会越过最优解；而较低的学习率则确保稳定，但可能需要更长时间才能收敛。
1. 训练轮数（Epochs）：这是指学习算法将整个训练数据集遍历的次数。更多的轮数可以提高性能，但也增加了过拟合的风险。
1. 正则化强度：正则化技术（如 L2 正则化或 dropout）通过对大权重施加惩罚或在训练期间随机丢弃单元，帮助防止过拟合。
1. 学习率衰减：实施学习率衰减可以改善收敛效果。通过逐渐降低学习率，可以让模型在训练初期采取较大的步伐，后期再进行微调。