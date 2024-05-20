数据使用整年的{Month, DayType, Time, Incident} e.g. {July, Monday, 12:00:00, Held by}

数据处理1:{int(unique encode), int(unique encode), int(minutes count),int(unique encode)}

数据处理2:{float(unique encode normalized), float(unique encode normalized), int(minutes count),int(unique encode)}

数据处理3:{vector(one-hot encode), vector(one-hot encode), int(minutes count),vector(one-hot encode)}

数据处理4:{vector(one-hot encode), vector(one-hot encode), vector(3 labels: morning,evening or non-peek),vector(one-hot encode)}

模型1: LSTM(使用seq_len=12的连续十二行特征输入)

模型2: 全连接神经网络(使用seq_len=1的单行特征输入)

|Acc/AUC|数据处理1|数据处理2|数据处理3|数据处理4|
|:--:|:--:|:--:|:--:|:--:|
|模型1|Extremely Low|~73%/53%|Not Tested|Not Tested|
|模型2|Extremely Low|Extremely Low|83%/67%|90+%/90+%|

由于数据标签非常不平衡（1：10），因此二分类模型训练不使用Sampler时，先验信息将导致结果不可信

使用特值编码特征时，全连接网络几乎无法提取任何特征，LSTM在某些情况下可以给出一定正确的答案，但AUC仍非常低

使用one-hot编码时，全连接网络可以产生一定表现，但开始时loss收敛很慢，且后续不断震荡不收敛

使用one-hot编码基础上，对时间的所属时间段进行手动标记，Acc与AUC直线上升，loss快速收敛（WHY？？）

下一步打算在全连接的基础上，将输入改为连续十二行特征输入，构建LSTM提取序列信息为现有全连接提供额外特征输入
