
from sortedcontainers import SortedList
import sys

# 这里的路径要根据实际情况进行替换
sys.path.append("E:\heterogeneous_simu_code\src")

from tool.DAG_process import ProcessDag


class Task:
    def __init__(self, id):
        self.id = id
        self.processor_id = None

        self.ranku = None
        self.rankd = None

        self.rank = None
        self.comp_cost = []
        self.avg_comp = None
        self.duration = {'start': None, 'end': None}


class Processor:
    def __init__(self, id):
        self.id = id
        self.task_list = []


class CPOP:
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

        self.__computeRank()
        self.__allocatProcessor()
        self.makespan = max([t.duration['end'] for t in self.tasks])


    def __computeRank(self):
        self.__computeRanku(self.tasks[0])
        self.__computeRankd(self.tasks[-1])

        for task in self.tasks:
            # task.rank = (int((task.ranku + task.rankd) * 1000 + 0.5)) / 1000.0
            task.rank = task.ranku + task.rankd

    def __computeRanku(self, task):
        curr_rank = 0
        for succ in self.tasks:
            if self.graph[task.id][succ.id] != -1:
                if succ.ranku is None:
                    self.__computeRanku(succ)
                curr_rank = max(curr_rank, self.graph[task.id][succ.id] + succ.ranku)
        task.ranku = task.avg_comp + curr_rank
        # task.ranku = (int((task.avg_comp + curr_rank) * 1000 + 0.5)) / 1000.0

    # 当前路径与入口节点的距离
    def __computeRankd(self, t):
        if t.id == 0:
            t.rankd = 0
            return
        t.rankd = 0
        for pre in self.tasks:
            if self.graph[pre.id][t.id] != -1:
                if pre.rankd is None:
                    self.__computeRankd(pre)
                t.rankd = max(t.rankd, pre.rankd + pre.avg_comp + self.graph[pre.id][t.id])

    def __get_est(self, t, p):
        """
        与HEFT算法一致
        指的是如果调度到这个节点上能够得到最早开始时间 EST
        :param t: 任务节点
        :param p: 处理器
        :return:
        """
        est = 0
        # 从众多最长节点中的选择耗时最长的
        for pre in self.tasks:
            if self.graph[pre.id][t.id] != -1:
                c = self.graph[pre.id][t.id] if pre.processor_id != p.id else 0
                try:
                    est = max(est, pre.duration['end'] + c)
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
        for slot in free_times:  # free_times is already sorted based on avaialbe start times
            if est < slot[0] and slot[0] + t.comp_cost[p.id] <= slot[1]:
                return slot[0]
            if est >= slot[0] and est + t.comp_cost[p.id] <= slot[1]:
                return est

    def __get_cp_set(self):
        SetCP = [self.tasks[0]]
        Nk = self.tasks[0]
        CP = Nk.rank
        while Nk.id != self.num_tasks - 1:
            for succ in self.tasks:
                # 与入口节点的CP值进行比较，如果相等，则认为是关键路径的上的节点
                if self.graph[Nk.id][succ.id] != -1 and abs(succ.rank - CP) <= 1e-3:
                    SetCP.append(succ)
                    Nk = succ
                    break
        return SetCP

    def __allocatProcessor(self):

        cp_set = self.__get_cp_set()
        for task in cp_set:
            print(task.id)
        cp_p = None
        s = float("inf")

        for p in self.processors:
            total_w = 0
            for t in cp_set:  # 对关键路径上节点 计算它们的开销和
                total_w += t.comp_cost[p.id]
            if total_w < s:
                cp_p = p
                s = total_w

        # 采用入口节点初始化关键路径节点[降序排列]
        ready_list = SortedList([self.tasks[0]], key=lambda x: -x.rank)


        while len(ready_list) != 0:

            t = ready_list.pop(0)

            if t in cp_set:  # 分配关键路径上的节点到关键路径处理器
                est = self.__get_est(t, cp_p)
                aft = est + t.comp_cost[cp_p.id]

                t.processor_id = cp_p.id
                t.duration['start'] = est
                t.duration['end'] = aft
                self.processors[cp_p.id].task_list.append(t)
                self.processors[cp_p.id].task_list.sort(key=lambda x: x.duration['start'])
            else:
                aft = float("inf")
                for p in self.processors:
                    est = self.__get_est(t, p)
                    eft = est + t.comp_cost[p.id]
                    if eft < aft:
                        aft = eft
                        best_p = p.id

                t.processor_id = best_p
                t.duration['start'] = aft - t.comp_cost[best_p]
                t.duration['end'] = aft
                self.processors[best_p].task_list.append(t)
                self.processors[best_p].task_list.sort(key=lambda x: x.duration['start'])

            # 更新后继节点到准备列表中
            for succ in self.tasks:
                if self.graph[t.id][succ.id] != -1:

                    flag = True
                    for succ_pre in self.tasks:
                        if self.graph[succ_pre.id][succ.id] != -1 and succ_pre.processor_id is None:
                            flag = False
                            break

                    if flag:
                        ready_list.add(succ)

    def __str__(self):
        print_str = ""
        for p in self.processors:
            print_str += 'Processor {}:\n'.format(p.id + 1)
            for t in p.task_list:
                print_str += 'Task {}: start = {}, end = {}\n'.format(t.id + 1, t.duration['start'], t.duration['end'])
        print_str += "Makespan = {}\n".format(self.makespan)
        return print_str


if __name__ == "__main__":
    input_list = ProcessDag.get_input_list("E:\\heterogeneous_simu_code\\data_gen\\V_10_Alpha_1.0_Maxout_4_CCR_0"
                                           ".1_Beta_0.25\\4.txt")
    cpop = CPOP(input_list=input_list)

    for p in cpop.processors:
        print("P#{}".format(p.id))
        for task in p.task_list:
            print("task#{}: {} -> {}".format(task.id, task.duration['start'], task.duration['end']))

    print(cpop.makespan)
