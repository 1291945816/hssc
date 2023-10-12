import argparse  # 解析命令行参数
import math
import os.path
import random
import numpy as np

# todo 后续可以优化为从某一个配置文件中读取内容
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='default', type=str,
                    help="There are two ways to generate a DAG graph: default and custom. Customization requires "
                         "manual input of parameters")
parser.add_argument('--v', default=10, type=int,
                    help="Number of tasks in the graph")
parser.add_argument('--alpha', default=1, type=float,
                    help="Shape parameter of the graph")
parser.add_argument('--max_out', default=3, type=int,
                    help="Max out_degree of the node")
parser.add_argument('--ccr', default=1.0, type=float,
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
        self.__nodes = args.v  # 实际节点数目（用以构建文件名称）
        self.__alpha = args.alpha
        self.__max_out = args.max_out
        self.__ccr = args.ccr
        self.__p = args.p
        self.__beta = args.beta
        self.__created_start = False
        self.__created_end = False

        self.__dag_v_layer = []  # 每个顶点的dag列表
        self.__edges = []  # 顶点之间的边集

        self.__in_degree = [0 for _ in range(self.__v)]  # 入度
        self.__out_degree = [0 for _ in range(self.__v)]  # 出度
        self.__height = 0
        self.__width = []

        # todo 这些提前处理可能会存在一些问题
        # 最小值 取下界
        mean_height = math.ceil(math.sqrt(self.__v) / self.__alpha)  # DAG Height
        mean_width = math.ceil(math.sqrt(self.__v) * self.__alpha)  # DAG Width

        # randomly generate height with uniform distribution
        self.__height = random.randint(1, 2 * mean_height - 1)

        # randomly generate width with uniform distribution
        self.__width = [random.randint(1, 2 * mean_width - 1) for _ in range(self.__height)]

        generate_num = sum(self.__width)
        if generate_num > self.__v:
            for i in range(generate_num - self.__v):
                index = random.randrange(0, self.__height, 1)  # 随机生成某一层的索引【用于删除节点】
                if self.__width[index] == 1:
                    re_index = random.randrange(0, self.__height, 1)  # 重新随机生成某一层的索引
                    while re_index == index or self.__width[re_index] == 1:
                        re_index = random.randrange(0, self.__height, 1)
                    self.__width[re_index] -= 1  # 删去某一层的节点
                else:
                    self.__width[index] -= 1

        if generate_num < self.__v:
            for i in range(self.__v - generate_num):
                index = random.randrange(0, self.__height, 1)  # 随机生成某一层的索引
                self.__width[index] += 1  # 在某一层增加节点

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
            # 入度为0的点【属于开始节点】
            if in_degree == 0:
                self.__edges.append((0, index + 1))  # +1 是指 索引是从0开始的,而边的关系则是从1开始的
                self.__in_degree[index] += 1
                start_out += 1

        # 虚拟开始节点的入度大于1 需要创建一个虚拟节点
        if start_out > 1:
            self.__in_degree = [0] + self.__in_degree
            self.__out_degree = [start_out] + self.__out_degree
            self.__v += 1
            self.__created_start = True
        else:
            self.__in_degree[self.__edges[-1][-1] - 1] -= 1
            self.__edges.pop()

        for index, out_degree in enumerate(self.__out_degree):
            # 出度 为 0 点
            if out_degree == 0:
                # 要单独处理 考虑是否新增了一个开始节点
                if self.__created_start:
                    self.__edges.append((index, self.__v))  # 新建了一个开始节点，这里处理的时候 会考虑下标会与实际的边相同
                else:
                    self.__edges.append((index + 1, self.__v + 1))  # 而不新增的时候则会从1开始 所以需要考虑是否新增了开始节点
                self.__out_degree[index] += 1
                end_in += 1

        # 虚拟结束节点的入度大于1 需要创建一个虚拟节点
        if end_in > 1:
            self.__in_degree = self.__in_degree + [end_in]
            self.__out_degree = self.__out_degree + [0]
            self.__v += 1
            self.__created_end = True
        else:
            if self.__created_start:
                self.__out_degree[self.__edges[-1][0]] -= 1
            else:
                self.__out_degree[self.__edges[-1][0] - 1] -= 1
            self.__edges.pop()

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

    def dag_construct(self, file_path, index=0, min_value=5, max_value=20):
        """
        构架DAG图
        :param file_path: 生成的DAG数据集存放的目录（最后需要加入 /）
        :param index: 在同一个目录下的下标值
        :param min_value: DAG图的平均计算开销的最小值
        :param max_value: DAG图的平均计算开销的最大值
        :return:
        """

        self.__dag_init()
        self.__edges_init()

        # 生成图的平均DAg
        wDAG = self.__set_randomly_avg_w_dag(min_value, max_value)

        # 节点之间的通信矩阵 -1 表示没有关系
        comm_v = np.array([[-1 for _ in range(self.__v)] for _ in range(self.__v)])  # v x v
        comp = np.array([[-1 for _ in range(self.__p)] for _ in range(self.__v)])  # v x p

        # 随机生成虚拟节点的通信值
        if self.__created_start and self.__created_end:
            '''
            创建了虚拟开始节点 和 虚拟 结束节点
            '''
            for edge in self.__edges:
                if edge[0] == 0 or edge[1] == (self.__v - 1):
                    comm_v[edge[0]][edge[1]] = 0
                else:
                    comm_v[edge[0]][edge[1]] = self.__commcost()
        elif self.__created_start and not self.__created_end:
            '''
            仅创建了虚拟开始节点
            '''
            for edge in self.__edges:
                if edge[0] == 0:
                    comm_v[edge[0]][edge[1]] = 0
                else:
                    comm_v[edge[0]][edge[1]] = self.__commcost()
        elif not self.__created_start and self.__created_end:
            '''
            创建了虚拟结束节点 此刻 0 代表实际节点
            '''
            for edge in self.__edges:
                if edge[1] == self.__v:
                    comm_v[edge[0] - 1][edge[1] - 1] = 0
                else:
                    comm_v[edge[0] - 1][edge[1] - 1] = self.__commcost()
        else:
            '''
            没有创建虚拟节点
            '''
            for edge in self.__edges:
                comm_v[edge[0] - 1][edge[1] - 1] = self.__commcost()

        # 对于每一个任务 处理器都生成一个值
        for i in range(self.__p):
            for j in range(self.__v):
                comp[j][i] = self.__comcost()

        # -1 表示处理器不处理这个任务
        if self.__created_start:
            comp[0, :] = 0
        if self.__created_end:
            comp[self.__v - 1, :] = 0

        #
        file_path = file_path \
                    + "V_" + str(self.__nodes) \
                    + "_Alpha_" + str(self.__alpha) \
                    + "_Maxout_" + str(self.__max_out) \
                    + "_CCR_" + str(self.__ccr) \
                    + "_Beta_" + str(self.__beta) + "/"

        # 创建目录
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path += str(index) + ".txt"
        f = open(file_path, mode='w')

        # 这里需要更新实际的V 考虑到有实际的 虚拟节点
        # 增加基本的描述信息
        f.write(f"Graph # {index} \n"
                f"\tv={self.__nodes}, alpha={self.__alpha}, CC Ratio={self.__ccr}, Average Computational Cost={wDAG} \n")
        f.write(f"\tVirtual start node={self.__created_start}, Virtual end node={self.__created_end}\n")

        f.write("\n#######\tcommunication matrix\t#######\n\n")
        f.write("--- From / to ---\t")

        # 表头
        for i in range(self.__v):
            f.write(f"#{i}\t")
        f.write("\n")

        for i in range(self.__v):
            f.write(f"\tTask#{i}\t\t\t")
            for j in range(self.__v):
                f.write(f"{comm_v[i][j]}\t")
            f.write("\n")

        f.write("\n#######\tcomputation matrix\t#######\n\n")
        f.write("---------\t")
        for i in range(self.__p):
            f.write(f"P{i + 1}\t")
        f.write("\n")

        for i in range(self.__v):
            f.write(f"\tTask#{i}\t")
            for j in range(self.__p):
                f.write(f"{comp[i][j]}\t")
            f.write("\n")
        f.close()


if __name__ == '__main__':
    for i in range(10):
        arg = parser.parse_args()
        dag = GenerateDag(arg)
        dag.dag_construct("E:\\heterogeneous_simu_code\\data_gen\\", index=i)
