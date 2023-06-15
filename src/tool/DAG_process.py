import argparse  # 解析命令行参数
import math
import os.path
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='default', type=str,
                    help="There are two ways to generate a DAG graph: default and custom. Customization requires "
                         "manual input of parameters")
parser.add_argument('--v', default=30, type=int,
                    help="Number of tasks in the graph")
parser.add_argument('--alpha', default=1, type=float,
                    help="Shape parameter of the graph")
parser.add_argument('--max_out', default=2, type=int,
                    help="Max out_degree of the node")
parser.add_argument('--ccr', default=0.1, type=float,
                    help="Communication to computation ratio")
parser.add_argument('--p', default=4, type=int,
                    help="Number of processors")
parser.add_argument('--beta', default=0.1, type=float,
                    help="Range percentage of computation costs on processors")


# 设置随机种子
# random.seed(10)


class GenerateDag:
    def __init__(self, args):
        self.__avg_w_dag = None  # 整个图中的平均计算开销
        self.__v = args.v
        self.__alpha = args.alpha
        self.__max_out = args.max_out
        self.__ccr = args.ccr
        self.__p = args.p
        self.__beta = args.beta

        self.__dag_v_layer = []  # 每个顶点的dag列表
        self.__edges = []  # 顶点之间的边集

        self.__in_degree = [0 for i in range(self.__v)]  # 入度
        self.__out_degree = [0 for i in range(self.__v)]  # 出度

        mean_height = math.ceil(math.sqrt(self.__v) / self.__alpha)  # DAG Height

        mean_width = math.ceil(math.sqrt(self.__v) * self.__alpha)  # DAG Width

        self.__height = random.randint(1, 2 * mean_height - 1)  # randomly generate height with uniform distribution

        # randomly generate width with uniform distribution
        self.__width = [random.randint(1, 2 * mean_width - 1) for i in range(self.__height)]

        generate_num = sum(self.__width)
        if generate_num > self.__v:
            for i in range(generate_num - self.__v):
                index = random.randrange(0, self.__height, 1)  # 随机生成一层的索引
                if self.__width[index] == 1:
                    re_index = random.randrange(0, self.__height, 1)  # 重新随机生成一层的索引
                    while re_index == index or self.__width[re_index] == 1:
                        re_index = random.randrange(0, self.__height, 1)
                    self.__width[re_index] -= 1
                else:
                    self.__width[index] -= 1

        if generate_num < self.__v:
            for i in range(self.__v - generate_num):
                index = random.randrange(0, self.__height, 1)  # 随机生成一层的索引
                self.__width[index] += 1

    def __dag_init(self):
        """
        用于生成有序的dag列表
        :return:
        """
        cur_index = 1
        self.__dag_v_layer = []
        for i in range(self.__height):
            self.__dag_v_layer.append(list(range(cur_index, cur_index + self.__width[i])))
            cur_index += self.__width[i]

    def __edges_init(self):
        """
        生成dag顶点集中的边集关系
        :return:
        """
        pred = 0  # 用于记录当前节点
        for i in range(self.__height - 1):
            second_layer_index = list(range(self.__width[i + 1]))
            for j in range(self.__width[i]):
                out_degree_ = random.randrange(1, self.__max_out + 1, 1)  # 随机生成一个出度
                out_degree_ = self.__width[i + 1] if self.__width[i + 1] < out_degree_ else out_degree_
                edge_links = np.random.choice(self.__width[i + 1], out_degree_, replace=False)  # 从第二层中随机选几个作为将要连接的
                for k in edge_links:
                    self.__edges.append((self.__dag_v_layer[i][j], self.__dag_v_layer[i + 1][k]))
                    # 出度 & 入度 更新
                    self.__in_degree[pred + self.__width[i] + k] += 1
                    self.__out_degree[pred + j] += 1
            pred += self.__width[i]

        # 处理虚拟的开始节点和结束节点
        start_out = 0  # 虚拟开始节点的出度
        end_in = 0  # 虚拟结束节点的入度
        for index, in_degree in enumerate(self.__in_degree):
            # 入度为0的点
            if in_degree == 0:
                self.__edges.append((0, index + 1))  # +1 是指某个点  在list的映射下标是-1
                self.__in_degree[index] += 1
                start_out += 1

        for index, out_degree in enumerate(self.__out_degree):
            # 出度 为 0 点
            if out_degree == 0:
                self.__edges.append((index + 1, self.__v))
                self.__out_degree[index] += 1
                end_in += 1

        # ## 还要针对性质的处理 只有一个节点时 是否还需要虚拟节点
        # # 虚拟节点的出度大于1，就说明有开始节点和结束节点都是不为1的
        # if start_out > 1 and end_in > 1:
        #     self.__in_degree = [0] + self.__in_degree + [end_in]  # 加入0入度的虚拟开始节点  加入 end_in 入度的虚拟结束节点
        #     self.__out_degree = [start_out] + self.__out_degree + [0]  # 加入start_out入度的虚拟开始节点  加入 0入度的虚拟结束节点
        # elif start_out > 1 and

    def __set_randomly_avg_w_dag(self, min_value, max_value):
        self.__avg_w_dag = random.randint(min_value, max_value)
        return self.__avg_w_dag

    def __comcost(self):
        """
        计算开销
        :return:
        """
        avg_w = random.randint(1, 2 * self.__avg_w_dag)
        wij = random.randint(math.ceil(avg_w * (1 - self.__beta / 2)), math.ceil(avg_w * (1 + self.__beta / 2)))
        return wij

    def __commcost(self):
        """
        通信开销
        :return:
        """
        # 获取平均通信开销
        avg_cc = math.ceil(self.__ccr * self.__avg_w_dag)
        comm_cost = random.randint(1, 2 * avg_cc - 1)
        return comm_cost

    def dag_construct(self, file_path, index = 0, min_value=5, max_value=20):
        self.__dag_init()
        self.__edges_init()

        # 生成图的平均DAg
        wDAG = self.__set_randomly_avg_w_dag(min_value, max_value)

        # 创建目录
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path += str(index) + ".txt"
        f = open(file_path, mode='w')


if __name__ == '__main__':
    lis1 = [0] + [1, 2, 3] + [3]
    lis2 = [1, 2, 3] + [0]
    print("{} {}".format(lis1, lis2))

    # arg = parser.parse_args()
    # dag = GenerateDag(arg)
    # dag.dag_construct()
