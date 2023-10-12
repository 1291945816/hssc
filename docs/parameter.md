> 参考论文：
>
> Topcuoglu H, Hariri S, Wu M Y. Performance-effective and low-complexity task scheduling for heterogeneous computing[J]. IEEE transactions on parallel and distributed systems, 2002, 13(3): 260-274.
>
>[获取论文](https://ieeexplore.ieee.org/document/993206)
### 输入参数（影响DAG图）

- **v**：图中任务（Task）中数量

- **alpha**：图的形状参数，其中它的**高度（深度）**从一个**平均值**为下式的**均匀分布**中获得：
  $$
  \lfloor \frac{\sqrt{v}}{\alpha}\rfloor
  $$
  而**每一层宽度**也是从一个**平均值**为下式的**均匀分布**中获得：
  $$
  \lfloor\sqrt{v} \cdot {\alpha} \rfloor
  $$
  当`alpha`远大于1时，这是一个带高并行程度的短图，而当`alpha`远小于1时，则是一个带低并行程度的长图。

- **max_out**：图中节点的**最大出度**。

- **CCR**：通信计算比，当`CCR`非常小的时候，说明这是一个计算密集型应用。

- **beta**：处理器上计算成本的范围百分比，是一个异构性因子，代表不同处理器之间的异构性。每个任务的**平均计算开销**从一个**均匀分布**中随机选择。而每个任务在每个处理器上的计算开销由下式决定：
  $$
  \overline{w_{i}} \cdot(1 - \frac{\beta}{2}) \le w_{i,j} \le \overline{w_{i}} \cdot (1 + \frac{\beta}{2})
  $$
  而平均计算开销由**全图的平均计算开销**决定，这是由用户决定的：
  $$
  \overline{w_i} \in [0, 2 \cdot \overline{w_{DAG}}]
  $$



此外在代码中还涵盖了一个额外的`p`参数，代表处理器数目。

