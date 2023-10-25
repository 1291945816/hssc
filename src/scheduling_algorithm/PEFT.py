
from collections import defaultdict
from sortedcontainers import SortedList
import numpy as np
import sys
# 这里的路径要根据实际情况进行替换
sys.path.append("F:\heterogeneous_simu_code\src")

from tool.DAG_process import ProcessDag

class Task:
    def __init__(self, id):
        self.id = id
        self.processor_id = None
        self.rank = None
        self.comp_cost = []
        self.avg_comp = None
        self.duration = {'start': None, 'end': None}
        self.CNP = False
        self.b_level = None
        self.t_level = None


class Processor:
    def __init__(self, id):
        self.id = id
        self.task_list = []


class PEFT:
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

        self.__computeRanks()

        self.Dag_succ = defaultdict(list)
        self.Dag_pred = defaultdict(list)

        for i in range(len(self.graph)):
            for j in range(len(self.graph[i])):
                if self.graph[i][j] != -1:
                    self.Dag_succ[i].append(j)
                    self.Dag_pred[j].append(i)

        if verbose:
            # print('OCT:\n', self.OCT)
            for task in self.tasks:
                print("Task {} -> Rank: {}".format(task.id + 1, task.rank))
        self.error = True
        self.__allotProcessor()
        self.makespan = None

        if self.error:
            self.makespan = max([t.duration['end'] for t in self.tasks])

    def computeSpeedup(self):
        self.speedup = float('inf')
        for p in self.processors:
            su = 0.0
            for task in self.tasks:
                su += task.comp_cost[p.id]
            self.speedup = min(self.speedup, su)

        self.speedup = self.speedup / self.makespan

    def computeSLR(self):
        self.slr = 0.0
        for task in self.tasks:
            if task.CNP:
                self.slr += np.min(np.array(task.comp_cost))
        self.slr = self.makespan / self.slr

    def computeStack(self):
        pass

    def populate_AEST(self, t):
        if t == self.tasks[0]:
            self.AEST[0] = 0
            return
        aest_preds = []
        for pre in self.tasks:
            if self.graph[pre.id][t.id] != -1:
                if self.AEST[pre.id] == -1:
                    self.populate_AEST(pre)
                aest_preds.append(self.AEST[pre.id] + pre.avg_comp + self.graph[pre.id][t.id])
        self.AEST[t.id] = max(aest_preds)

    def populate_ALST(self, t):
        if t == self.tasks[self.num_tasks - 1]:
            self.ALST[t.id] = self.AEST[t.id]
            return
        alst = float('inf')
        for succ in self.tasks:
            if self.graph[t.id][succ.id] != -1:
                if self.ALST[succ.id] == -1:  # ALST not calculated yet
                    self.populate_ALST(succ)
                c_im = self.graph[t.id][succ.id]
                alst = min(alst, self.ALST[succ.id] - c_im)
        self.ALST[t.id] = alst - t.avg_comp

    def populate_OCT(self, t, p):
        if t == self.tasks[self.num_tasks - 1]:
            self.OCT[t.id][p.id] = 0
            return
        pct = -float('inf')

        for succ in self.tasks:
            if self.graph[t.id][succ.id] != -1:
                min_proc_ppts = float('inf')
                for pm in self.processors:
                    if self.OCT[succ.id][pm.id] == -1:
                        self.populate_OCT(succ, pm)  # 递归计算
                    c_ij = self.graph[t.id][succ.id] if p.id != pm.id else 0

                    new_pct = self.OCT[succ.id][pm.id] + succ.comp_cost[pm.id] + c_ij
                    min_proc_ppts = min(min_proc_ppts, new_pct)
                pct = max(pct, min_proc_ppts)

        self.OCT[t.id][p.id] = pct

    def __computeRanks(self):
        self.OCT = np.full((self.num_tasks, self.num_processors), -1)
        for p in self.processors:
            self.populate_OCT(self.tasks[0], p)

        avg_pct = np.sum(self.OCT, axis=1) / self.num_processors
        for t in self.tasks:
            t.rank = avg_pct[t.id]

    def __get_est(self, t, p):
        est = 0
        for pre in self.tasks:
            if self.graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = self.graph[pre.id][t.id] if pre.processor_id != p.id else 0
                try:
                    # print(pre.duration['end'])
                    est = max(est, pre.duration['end'] + c)
                except:
                    # print(pre.id)
                    # print(t.id)
                    # raise Exception()
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
        for slot in free_times:  # free_times is already sorted based on avaialbe start times
            if est < slot[0] and slot[0] + t.comp_cost[p.id] <= slot[1]:
                return slot[0]
            if est >= slot[0] and est + t.comp_cost[p.id] <= slot[1]:
                return est

    def __allotProcessor(self):

        ready_list = SortedList([self.tasks[0]], key=lambda x: -x.rank)  # 降序排列
        schedule_list = []
        while len(ready_list) != 0:
            t = ready_list.pop(0)

            curr_eft_cnct = float("inf")
            for p in self.processors:
                est = self.__get_est(t, p)
                eft = est + t.comp_cost[p.id]

                eft_cnct = eft + self.OCT[t.id][p.id]

                if eft_cnct < curr_eft_cnct:  # found better case of processor
                    curr_eft_cnct = eft_cnct
                    aft = eft
                    best_p = p.id

            t.processor_id = best_p
            t.duration['start'] = aft - t.comp_cost[best_p]
            t.duration['end'] = aft
            self.processors[best_p].task_list.append(t)
            self.processors[best_p].task_list.sort(key=lambda x: x.duration['start'])

            schedule_list.append(t.id)
            for tt in self.Dag_succ[t.id]:
                flag = True
                for tt_pre in self.Dag_pred[tt]:
                    if tt_pre not in schedule_list:
                        flag = False
                        break
                if flag:
                    ready_list.add(self.tasks[tt])

    def __str__(self):
        print_str = ""
        for p in self.processors:
            print_str += 'Processor {}:\n '.format(p.id + 1)
            for t in p.task_list:
                print_str += 'Task {}: start = {}, end = {}\n'.format(t.id + 1, t.duration['start'], t.duration['end'])
        print_str += "Makespan = {}\n".format(self.makespan)
        return print_str


if __name__ == '__main__':
    input_list = ProcessDag.get_input_list("F:\\heterogeneous_simu_code\\data_gen\\V_10_Alpha_1.0_Maxout_4_CCR_0"
                                           ".1_Beta_0.25\\11.txt")
    peft = PEFT(input_list)
    print(peft)