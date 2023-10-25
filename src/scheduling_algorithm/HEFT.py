import operator
# 导入其他模块的内容
import sys
# 这里的路径要根据实际情况进行替换
sys.path.append("E:\heterogeneous_simu_code\src")

from tool.DAG_process import ProcessDag


#
class Task:
    def __init__(self, id):
        self.id = id
        self.processor_id = None
        self.rank = None
        self.comp_cost = []  # 执行开销 多个处理器
        self.avg_comp = None  # 平均执行开销
        self.duration = {'start': None, 'end': None}  # 执行时间


class Processor:
    def __init__(self, id):
        self.id = id
        self.task_list = []


class HEFT:
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

            # 计算处理器的平均处理器开销
            self.tasks[i].avg_comp = sum(comp_cost[i]) / self.num_processors

        # 第 0 个任务 默认为初始任务
        self.__computeRanks(self.tasks[0])
        if verbose:
            for task in self.tasks:
                print("Task ", task.id, "-> Rank: ", task.rank)
        self.tasks.sort(key=lambda x: x.rank, reverse=True)  # 降序排
        print("sorted tasklist: ")
        for task in self.tasks:
            print(f"Task {task.id}->Rank: {task.rank}")
        self.error = True
        self.__allotProcessor()
        self.makespan = None

        if self.error:
            self.makespan = max([t.duration['end'] for t in self.tasks])

    def __computeRanks(self, task):
        """
        计算rank\n
        假设1 task[0]是初始任务\n
        假设2 处理器之间的通信速率是相等 也就是忽略了处理器之间的通信开销\n
        递归计算 从后往上回溯 [upward_rank]\n
        :param task: Dag图中的任务节点
        :return:
        """
        curr_rank = 0
        for succ in self.tasks:
            if self.graph[task.id][succ.id] != -1:  # 非-1才代表连通
                if succ.rank is None:
                    self.__computeRanks(succ)
                curr_rank = max(curr_rank, self.graph[task.id][succ.id] + succ.rank)
        task.rank = task.avg_comp + curr_rank

    def __get_est(self, t, p):
        est = 0
        for pre in self.tasks:
            if self.graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = self.graph[pre.id][t.id] if pre.processor_id != p.id else 0
                try:
                    est = max(est, pre.duration['end'] + c)  # 实际完成的时间
                except:
                    # raise Exception()
                    self.error = False
                    return -1

        # 切分处理器上任务的空闲时间[可获得的时间]
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

        # 可获得时间
        for slot in free_times:  # free_times is already sorted based on avaialbe start times
            if est < slot[0] and slot[0] + t.comp_cost[p.id] <= slot[1]:
                return slot[0]
            if est >= slot[0] and est + t.comp_cost[p.id] <= slot[1]:
                return est

    # 分配处理器
    def __allotProcessor(self):
        for t in self.tasks:
            if t == self.tasks[0]:  # the one with the highest rank
                p, w = min(enumerate(t.comp_cost), key=operator.itemgetter(1))
                # p, w = min(enumerate(t.comp_cost), key=lambda e_i: e_i[1]) # 比较第二个元素的最小值
                t.processor_id = p
                t.duration['start'] = 0
                t.duration['end'] = w
                self.processors[p].task_list.append(t)

            else:
                aft = float("inf")
                for p in self.processors:
                    est = self.__get_est(t, p)  # est 包括了通信开销
                    # print("Task: ", t.id, ", Proc: ", p.id, " -> EST: ", est)
                    eft = est + t.comp_cost[p.id]
                    if eft < aft:  # found better case of processor
                        aft = eft
                        best_p = p.id

                t.processor_id = best_p
                t.duration['start'] = aft - t.comp_cost[best_p]
                t.duration['end'] = aft
                self.processors[best_p].task_list.append(t)
                self.processors[best_p].task_list.sort(key=lambda x: x.duration['start'])

    def __str__(self):
        print_str = ""


        for p in self.processors:
            print_str += 'Processor {}:\n '.format(p.id + 1)
            for t in p.task_list:
                print_str += 'Task {}: start = {}, end = {}\n'.format(t.id, t.duration['start'], t.duration['end'])
        print_str += "Makespan = {}\n".format(self.makespan)
        return print_str


if __name__ == "__main__":
    input_list = ProcessDag.get_input_list("F:\\heterogeneous_simu_code\\data_gen\\V_10_Alpha_1.0_Maxout_4_CCR_0"
                                           ".1_Beta_0.25\\11.txt")
    heft = HEFT(input_list=input_list)
    print(heft)


    # from argparse import ArgumentParser
    #
    # ap = ArgumentParser()
    # ap.add_argument('-i', '--input', required=True, help="DAG description as a .dot file")
    # args = ap.parse_args()
    # new_sch = HEFT(file=args.input, verbose=True, p=4, b=0.1, ccr=0.1)
    # print(new_sch)


