# HW2笔记

## 多层神经网络初始化

```enumerate(hidden_dims, 1)``` 从1开始枚举，hidden_dims是一个list，里面存放的是每一层的神经元个数。

## 不同层数的神经网络

- 深层网络通常对初始化规模更敏感。这是因为深层网络的梯度传播可能会受到影响，较大的初始化可能导致梯度消失或爆炸，从而使得训练变得更加困难。
- 较小的初始化规模有助于保持网络的稳定性，尤其是在深层网络中。

## 新的更新参数方式SGD

### Momentum
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position

### RMSprop
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)

### Adam
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)

## Batchnormalization

- 求导过程，注意mean和var对在一个batch中都起作用，要求和！
- 初始化相关：提升鲁棒性

## Dropout

- 训练时，每个神经元以概率p被保留，以概率1-p被丢弃。
- 遗留问题：为什么lr要调到1e-6那么小！

## 调参

- 调不好，怎么搞都只有0.42的validation accuracy

## CNN

- 实际实现时，有slice函数，可以集成切片“...:...”的表示
- 需要扩充维数方便广播，np.expand_dims()函数

## 导入库时路径问题

- vscode会以工作目录为基准，而不是执行的那个文件所在的目录

## CNN的batchnorm

- 可以复用之前神经网络的batchnorm代码
- 注意C这一维度是特征量，不能混为一谈，以C为分类标准，每一类中的N*H*W进行归一化

## 版本兼容问题

- TypeError: expected np.ndarray (got numpy.ndarray)
  - 将im_as_ten = torch.from_numpy(im_as_arr).float()改成im_as_ten = torch.Tensor(im_as_arr).float()
  - 
## Pytorch

- 关于形状：用view代替numpy的reshape即可
- 关于乘法：用.mm代替numpy的dot或@
- 卷积：F.conv2d()，直接传3D/4D的参数进去，不必flatten
- **Kaiming**初始化！`weight_scale = * np.sqrt(2. / fan_in)`，fan_in是输入神经元的数量
- 更新梯度时要进入with torch.no_grad()模式，更新完记得清零梯度`w.grad.zero_()`

### API调用

- 先定义一个类，继承自nn.Module，然后在构造函数中定义网络结构，forward函数中定义前向传播过程

