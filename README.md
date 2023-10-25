
### 异构计算平台任务调度仿真平台未来规划

introduction: **This is a task scheduling simulation platform for general heterogeneous computing platforms.**

> 在该平台中，既能完成`DAG`任务的图的生成，也能完成任务的调度仿真，同时采用更加合理的方式进行可视化展示。

#### 第一阶段

> 在该阶段主要负责完成基本的DAG图生成、解析。

- [x] 分析网上的DAG生成代码，整理实现思路
- [x] DAG任务的构建生成
- [x] DAG任务的解析

#### 第二阶段

> 复现各类经典的、新型的**启发式调度算法。**

|      算法       |                                                                   论文                                                                    |
|:-------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|
| HEFT算法、CPOP算法 |   [Performance-effective and low-complexity task scheduling for heterogeneous computing](https://ieeexplore.ieee.org/document/993206)   |
|    PEFT算法     | [List Scheduling Algorithm for HeterogeneousSystems by an Optimistic Cost Table](https://ieeexplore.ieee.org/abstract/document/6471969) |
|       -       |                                                                    -                                                                    |



