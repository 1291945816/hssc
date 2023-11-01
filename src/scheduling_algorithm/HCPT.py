from sortedcontainers import SortedList
import sys

# 这里的路径要根据实际情况进行替换
sys.path.append("E:\heterogeneous_simu_code\src")

from tool.DAG_process import ProcessDag


class Task:
    def __init__(self, id):
        self.id = id
        self.processor_id = None
        self.aest = -1
        self.alst = -1
        self.comp_cost = []
        self.avg_comp = None
        self.duration = {'start': None, 'end': None}


class Processor:
    def __init__(self, id):
        self.id = id
        self.task_list = []


class HCPT:
    def __init__(self, input_list=None, verbose=False):
        if len(input_list) == 4:
            self.num_tasks, self.num_processors, comp_cost, self.graph = input_list
        else:
            print('Enter filename or input params')
            raise Exception()

        if verbose:
            print("No. of Tasks: ", self.num_tasks)
            print("No. of processors: ", self.num_processors)
            print("Computational Cost Matrix:")
            for i in range(self.num_tasks):
                print(comp_cost[i])
            print("Graph Matrix:")
            for line in self.graph:
                print(line)

        self.tasks = [Task(i) for i in range(self.num_tasks)]
        self.processors = [Processor(i) for i in range(self.num_processors)]

        for i in range(self.num_tasks):
            self.tasks[i].comp_cost = comp_cost[i]
            self.tasks[i].avg_comp = sum(comp_cost[i]) / self.num_processors

        self.__get_aest(self.tasks[-1])
        self.tasks[-1].alst = self.tasks[-1].aest
        self.__get_alst(self.tasks[0])
        # self.__listed_stage()
        self.__alloc_proc()

        # self.__allocatProcessor()
        self.makespan = max([t.duration['end'] for t in self.tasks])

    def __get_aest(self, t):
        if t.id == 0:
            t.aest = 0
            return
        current_aest = -1
        for pre in self.tasks:
            if self.graph[pre.id][t.id] != -1:
                if pre.aest == -1:
                    self.__get_aest(pre)
                current_aest = max(pre.aest + pre.avg_comp + self.graph[pre.id][t.id], current_aest)
        t.aest = current_aest

    def __get_alst(self, t):
        current_min_alst = float('inf')
        for succ in self.tasks:
            if self.graph[t.id][succ.id] != -1:
                if succ.alst == -1:
                    self.__get_alst(succ)
                current_min_alst = min(current_min_alst, succ.alst - self.graph[t.id][succ.id])
        t.alst = current_min_alst - t.avg_comp

    def __listed_stage(self):
        """
        列表阶段 获取所有的关键节点，并放入栈中（后进先出）
        :return:
        """
        S = []
        L = []
        for t in self.tasks:
            if abs(t.alst - t.aest) <= 1e-3:
                S.append(t)  # 压入列表中
        S = sorted(S, key=lambda x: x.alst, reverse=True)


        while len(S) != 0:  # 判断父节点
            top = S[-1]
            flag = False
            H_L = []
            for parent in self.tasks:

                if self.graph[parent.id][top.id] != -1:
                    e_isexist = False
                    for x in L:
                        if x.id == parent.id:  # 元素存在
                            e_isexist = True
                            break
                    if not e_isexist:  # 元素不存在
                        flag = True
                        H_L.append(parent)
            H_L = sorted(H_L, key=lambda x: x.alst, reverse=True)
            while len(H_L) != 0:
                S.append(H_L[0])
                H_L.pop(0)

            if not flag:
                L.append(top)
                S.pop()
        return L

    def __alloc_proc(self):
        """
        分配处理器
        :return:
        """
        L = self.__listed_stage()

        while len(L) != 0:
            front = L[0]  # 队头
            aft = float("inf")
            best_p = None

            for p in self.processors:
                eest = self.__get_eest(front, p)
                eeft = eest + front.comp_cost[p.id]

                if eeft < aft:
                    aft = eeft
                    best_p = p.id

            front.processor_id = best_p
            front.duration['start'] = aft - front.comp_cost[best_p]
            front.duration['end'] = aft
            self.processors[best_p].task_list.append(front)
            self.processors[best_p].task_list.sort(key=lambda x: x.duration['start'])
            L.pop(0)    # 移除

    def __get_eest(self, t, p):
        """
        类似于其他算法的获取最早开始时间
        :param t: 任务节点
        :param p: 处理器
        :return:
        """
        eest = 0
        # 从众多最长节点中的选择耗时最长的
        for pre in self.tasks:
            if self.graph[pre.id][t.id] != -1:
                c = self.graph[pre.id][t.id] if pre.processor_id != p.id else 0
                try:
                    eest = max(eest, pre.duration['end'] + c)
                except:
                    self.error = False
                    return -1

        free_times = []
        if len(p.task_list) == 0:  # no task has yet been assigned to processor
            free_times.append([0, float('inf')])
        else:
            for i in range(len(p.task_list)):
                if i == 0:
                    if p.task_list[i].duration['start'] != 0:  # if p is not busy from time 0
                        free_times.append([0, p.task_list[i].duration['start']])
                else:
                    free_times.append([p.task_list[i - 1].duration['end'], p.task_list[i].duration['start']])
            free_times.append([p.task_list[-1].duration['end'], float('inf')])
        # 插入策略
        # return eest

        for slot in free_times:
            if eest < slot[0] and slot[0] + t.comp_cost[p.id] <= slot[1]:
                return slot[0]
            if eest >= slot[0] and eest + t.comp_cost[p.id] <= slot[1]:
                return eest

    def __str__(self):
        print_str = ""
        for p in self.processors:
            print_str += 'Processor {}:\n'.format(p.id + 1)
            for t in p.task_list:
                print_str += 'Task {}: start = {}, end = {}\n'.format(t.id + 1, t.duration['start'], t.duration['end'])
        print_str += "Makespan = {}\n".format(self.makespan)
        return print_str


if __name__ == "__main__":
    input_list = ProcessDag.get_input_list("F:\\heterogeneous_simu_code\\data_gen\\V_10_Alpha_1.0_Maxout_4_CCR_0"
                                           ".1_Beta_0.25\\11.txt")
    hcpt = HCPT(input_list=input_list)
    print(hcpt)
