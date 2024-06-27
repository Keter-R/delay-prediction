2024.6.27

<span style="color: red; "> 修复了数据混淆的问题，之前模型结果均无效！ </span>

新数据下应对样本不平衡问题，从加权采样改为加权损失函数

当前测试结果如下

|  Result  |  FCN  |   LSTM   |   GCN    | ST-GCN |
|:--------:|:-----:|:--------:|:--------:|:------:|
| Accuracy | 0.791 | untested | untested | 0.827  |
|   AUC    | 0.836 | untested | untested | 0.854  |

2024.6.21

<h2>图结构数据来源</h2>

- 由[TTC](www.ttc.ca/routes-and-schedules)官网获取StreetCar线路图

- 线路包含501, 503-512（截图见`data/routes info/stops/{route}.png`）

- 通过OCR方式提取出站台名和对应的站台编号

- 将站台名和站编号存储在`data/routes info/stops/{route}.csv`中

- 对于同一路线的站台进行前后相连得到多个链

- 不同线路会有相同站台，因此多个链会相交形成图结构

<h2>数据中标注站台</h2>

- 由于数据集中仅有"Location"字段，似乎是事发地点街与道的信息，而不是站台名

- 采用匹配的方式对"Station"字段进行表注。具体而言，对当前路线的所有站台名和当前Location进行词集包含性测试，重复度最高的站台名（且Confidence>80%）即为标注结果

- 同时标注站台名和站台号，其中站台名用于人工校对，站台号用于后续的图结构构建（需要重新映射到0起始标号）

<h2>数据集处理</h2>

- 对`Month`, `Day`, `Time//15`, `Route`, `Incident` 字段进行one-hot处理

- 对`Station ID`字段进行重新映射，使得所有站台号从0开始，用于GCN的输入的数据构建

- 使用`seq_len = 16`的滑动窗口对数据集进行切分，每个窗口内的数据作为一个样本

- 前`seq_len-1`行数据每行表示了当次列车的信息及是否晚点

- 最后一行数据表示了需要预测的列车的车次信息，但不包含是否晚点的信息

- 对正负样本进行权重调整，使得每个batch中采出的正负样本的比例接近1:1

- 对数据集进行划分，按照时间顺序和0.8的比例划分为训练集、验证集

<h2>模型设计</h2>

<h3>FCN</h3>

全连接神经网络

- 输入为`[batch_size, seq_len, num_features]`的数据

- 仅保留`seq_len`维度的最后一个维度，即`[batch_size, num_features-1]`，即当次列车的信息

- 通过全连接层进行特征提取，最后通过sigmoid函数输出是否晚点的概率

- 输出为`[batch_size, 1]`的数据


<h3>LSTM</h3>

LSTM神经网络

- 输入为`[batch_size, seq_len, num_features]`的数据

- 由于`seq_len`维度最后一行数据不包含是否晚点的信息，因此仅保留`seq_len-1`维度的数据用于LSTM的输入

- 通过LSTM层进行特征提取，得到序列特征`[batch_size, hidden_size]`

- 将序列特征和当次列车的信息进行拼接，通过全连接层进行特征提取，最后通过sigmoid函数输出是否晚点的概率

- 输出为`[batch_size, 1]`的数据


<h3>GCN</h3>

图卷积神经网络

- 输入为`[batch_size, seq_len, num_features]`的数据

- 由于`seq_len`维度最后一行数据不包含是否晚点的信息，因此仅保留`seq_len-1`维度的数据用于GCN的输入

- 将数据集中的`Station ID`字段作为节点的特征，构建图结构。具体而言，记每个样本的前`seq_len-1`行数据中的`Station ID`字段作为节点编号，将对应行的特征值填入节点特征矩阵的对应行中

- 通过GCN层进行特征提取，得到节点特征`[batch_size, node_num, hidden_size]`

- 提取预测节点的特征，得到`[batch_size, hidden_size]`的数据

- 将节点特征和当次列车的信息进行拼接，通过全连接层进行特征提取，最后通过sigmoid函数输出是否晚点的概率


<h3>GCN2</h3>

图卷积神经网络2

- 输入为`[batch_size, seq_len, num_features]`的数据

- 由于`seq_len`维度最后一行数据不包含是否晚点的信息，因此舍弃所有是否晚点的信息，得到`[batch_size, seq_len, num_features-1]`的数据

- 将每个样本的信息填入节点特征矩阵的对应行中

- 通过GCN层进行特征提取，得到节点特征`[batch_size,node_num, hidden_size]`

- 提取预测节点的特征，得到`[batch_size, hidden_size]`的数据

- 不进行拼接，直接通过全连接层进行特征提取，最后通过sigmoid函数输出是否晚点的概率


<h2>测试结果</h2>

`batch_size = 32, lr = 0.01, epoch = 50, hidden_size = 512`

`seq_len = 16, node_num = 343, num_features = 140`

`lr_gamma = 0.97`

|  Result  |  FCN  | LSTM  |  GCN  | GCN2  |
|:--------:|:-----:|:-----:|:-----:|:-----:|
| Accuracy | 0.981 | 0.978 | 0.976 | 0.975 |
|   AUC    | 0.926 | 0.920 | 0.928 | 0.913 |

`batch_size = 32, lr = 0.003, epoch = 50, hidden_size = 512`

`seq_len = 16, node_num = 343, num_features = 140`

`lr_gamma = 0.997`

|  Result  |  FCN  | LSTM  |  GCN  | GCN2  |
|:--------:|:-----:|:-----:|:-----:|:-----:|
| Accuracy | 0.973 | 0.954 | 0.968 | 0.969 |
|   AUC    | 0.889 | 0.860 | 0.940 | 0.969 |


<h2>下一步工作</h2>

- 尝试在合适的`seq_len`对相同`Time`的数据进行合并，以实现同时具有时间序列和图结构的特征

- 进而引入更加复杂的模型，同时从时间序列和图结构两个维度进行特征提取

- 当前问题在于站台共计343个，但`seq_len`较小，因此图的特征矩阵包含大量的0值







