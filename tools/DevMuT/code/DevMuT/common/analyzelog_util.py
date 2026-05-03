import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import time
import numpy as np
from scipy import spatial

NAN_FLAG = -100
loss_id = 0
acc_id = 0
memory_id = 0


class train_result_analyze:

    def __init__(self, model_name, epochs, loss_ms, loss_torch, memories_ms, eval_ms, eval_torch, memories_torch,loss_truth, acc_truth, memory_truth, train_logger):
        self.model_name, self.epochs, self.loss_ms, self.loss_torch, self.path = model_name, epochs, loss_ms, loss_torch, train_logger.log_file[:-8]
        self.memories_ms, self.memories_torch = memories_ms, memories_torch
        self.loss_truth = loss_truth
        self.acc_ms, self.acc_torch = eval_ms, eval_torch
        self.acc_truth = acc_truth
        self.memory_truth = memory_truth
        self.train_logger = train_logger
        self.generation = train_logger.generation

    def nan_fix(self, metrics):
        metrics_fix = []
        for metrics_ in metrics:
            if np.isnan(metrics_):
                metrics_fix.append(NAN_FLAG)
            else:
                metrics_fix.append(metrics_)
        return metrics_fix
    
    
    def loss_pic(self):
        global loss_id
        epoch_axis = [i for i in range(self.epochs)]
    
        loss_ms_fix = self.nan_fix(self.loss_ms)
        loss_torch_fix = self.nan_fix(self.loss_torch)
        fig = plt.figure(num=1, figsize=(10, 10))
    
        ax_loss = fig.add_subplot(1, 1, 1)
        ax_loss.plot(epoch_axis, loss_ms_fix, label='mindspore loss')
        ax_loss.plot(epoch_axis, loss_torch_fix, label='torch loss')
    
        y_minor_locator = MultipleLocator(1000)
        ax_loss.yaxis.set_minor_locator(y_minor_locator)
    
        ax_loss.legend()
        ax_loss.set_title(f'{self.model_name} loss diagram {loss_id}')
        plt.savefig(f'{self.path}/{self.model_name}_loss_{loss_id}')
        plt.close()
        loss_id += 1
    
    
    def acc_pic(self):
        global acc_id
        epoch_axis = [i for i in range(self.epochs)]
    
        acc_ms_fix = self.nan_fix(self.acc_ms)
        acc_torch_fix = self.nan_fix(self.acc_torch)
    
        fig = plt.figure(num=1, figsize=(10, 10))
        ax_acc = fig.add_subplot(1, 1, 1)
        ax_acc.plot(epoch_axis, acc_ms_fix, label='mindspore accuracy')
        ax_acc.plot(epoch_axis, acc_torch_fix, label='torch accuracy')
    
        ax_acc.legend()
        ax_acc.set_title(f'{self.model_name} accuracy diagram {acc_id}')
        plt.savefig(f'{self.path}/{self.model_name}_acc_{acc_id}')
        plt.close()
        acc_id += 1
    
    
    def memory_pic(self):
        global memory_id
        epoch_axis = [i for i in range(len(self.memories_ms))]
    
        fig = plt.figure(num=1, figsize=(10, 10))
        ax_mem = fig.add_subplot(1, 1, 1)
        ax_mem.plot(epoch_axis, self.memories_ms, label='mindspore memory usage')
        ax_mem.plot(epoch_axis, self.memories_torch, label='torch memory usage')
        ax_mem.legend()
        ax_mem.set_title(f'{self.model_name} memory diagram {memory_id}')
        plt.savefig(f'{self.path}/{self.model_name}_memory_{memory_id}')
        plt.close()
        memory_id += 1

    def cal_frechet_distance(self, curve_a: np.ndarray, curve_b: np.ndarray):
        # 距离公式，两个坐标作差，平方，累加后开根号
        def euc_dist(pt1, pt2):
            return np.sqrt(np.square(pt2[0] - pt1[0]) + np.square(pt2[1] - pt1[1]))

        # 用递归方式计算，遍历整个ca矩阵
        def _c(ca, i, j, P, Q):  # 从ca左上角开始计算，这里用堆的方法把计算序列从右下角压入到左上角，实际计算时是从左上角开始
            if ca[i, j] > -1:
                return ca[i, j]
            elif i == 0 and j == 0:  # 走到最左上角，只会计算一次
                ca[i, j] = euc_dist(P[0], Q[0])
            elif i > 0 and j == 0:  # 沿着Q的边边走
                ca[i, j] = max(_c(ca, i - 1, 0, P, Q), euc_dist(P[i], Q[0]))
            elif i == 0 and j > 0:  # 沿着P的边边走
                ca[i, j] = max(_c(ca, 0, j - 1, P, Q), euc_dist(P[0], Q[j]))
            elif i > 0 and j > 0:  # 核心代码：在ca矩阵中间走，寻找对齐路径
                ca[i, j] = max(min(_c(ca, i - 1, j, P, Q),  # 正上方
                                   _c(ca, i - 1, j - 1, P, Q),  # 斜左上角
                                   _c(ca, i, j - 1, P, Q)),  # 正左方
                               euc_dist(P[i], Q[j]))
            else:  # 非法的无效数据，算法中不考虑，此时 i<0,j<0
                ca[i, j] = float("inf")
            return ca[i, j]

        # 这个是给我们调用的方法
        def frechet_distance(P, Q):
            ca = np.ones((len(P), len(Q)))
            ca = np.multiply(ca, -1)
            dis = _c(ca, len(P) - 1, len(Q) - 1, P, Q)  # ca为全-1的矩阵，shape = ( len(a), len(b) )
            return dis

        # 这里构造计算序列
        curve_line_a = list(zip(range(len(curve_a)), curve_a))
        curve_line_b = list(zip(range(len(curve_b)), curve_b))
        return frechet_distance(curve_line_a, curve_line_b)

    def get_cos_similar(self, v1: list, v2: list):
        return 1 - spatial.distance.cosine(v1, v2)

    def mem_usage_security(self, memories, threshold):
        safe_num = 0
        for i in range(1, len(memories)):
            if float(memories[i]) - float(memories[i - 1]) < float(threshold):
                safe_num += 1
        return safe_num / (len(memories) - 1)

    def loss_acc_compare(self, losses_ms, losses_torch, accs_ms, accs_torch):
        loss_ms = losses_ms[-1]
        loss_torch = losses_torch[-1]
        acc_ms = accs_ms[-1]
        acc_torch = accs_torch[-1]
        return (loss_ms <= loss_torch and acc_ms >= acc_torch) or (loss_ms >= loss_torch and acc_ms <= acc_torch)

    def get_report(self):
        GROUND_TRUTH_LOSS = self.loss_truth
        GROUND_TRUTH_ACC = self.acc_truth
        GROUND_TRUTH_MEM = self.memory_truth

        generation = self.generation

        loss_dis = self.cal_frechet_distance(np.array(self.loss_ms), np.array(self.loss_torch))
        acc_dis = self.get_cos_similar(self.acc_ms, self.acc_torch)
        mem_safe_ratio_ms = self.mem_usage_security(self.memories_ms, GROUND_TRUTH_MEM)
        mem_safe_ratio_torch = self.mem_usage_security(self.memories_torch, GROUND_TRUTH_MEM)
        has_no_conflict = self.loss_acc_compare(self.loss_ms, self.loss_torch, self.acc_ms, self.acc_torch)
        has_loss_acc_conflict = not has_no_conflict


        self.train_logger.info(f'this is the {generation} mutation generation')
        if np.isnan(loss_dis):
            self.train_logger.error(f'loss distance is {loss_dis}, nan')
        elif loss_dis > GROUND_TRUTH_LOSS:
            self.train_logger.error(f'loss distance is {loss_dis}, exceed the threshold:{GROUND_TRUTH_LOSS}')
        else:
            self.train_logger.info(f'loss distance is {loss_dis}, normal float number')

        if np.isnan(acc_dis):
            self.train_logger.error(f'accuracy distance is {acc_dis}, nan')
        if acc_dis > GROUND_TRUTH_ACC:
            self.train_logger.error(f'accuracy distance is {acc_dis}, exceed the threshold:{GROUND_TRUTH_ACC}')
        else:
            self.train_logger.info(f'accuracy distance is {acc_dis}, normal float number')

        if mem_safe_ratio_ms < 0.6 or mem_safe_ratio_torch < 0.6:
            self.train_logger.error(
                f'mindspore memory security ration is {mem_safe_ratio_ms}, '
                f'torch memory security ration is {mem_safe_ratio_torch}, '
                f'memory growth exceed the threshold:{GROUND_TRUTH_MEM}')
        else:
            self.train_logger.info(f'mindspore memory security ration is {mem_safe_ratio_ms},'
                    f'torch memory security ration is {mem_safe_ratio_torch}, normal float number')

        if any([loss < 0 for loss in self.loss_ms]):
            self.train_logger.error(f'exists negative loss in mindspore')
        if any([loss < 0 for loss in self.loss_torch]):
            self.train_logger.error(f'exists negative loss in torch')
        if any(np.isnan(self.loss_ms)):
            self.train_logger.error(f'exists NAN loss in mindspore')
        if any(np.isnan(self.loss_torch)):
            self.train_logger.error(f'exists NAN loss in torch')

        if has_loss_acc_conflict:
            self.train_logger.error(f'exist conflict between loss and accuracy')

    def analyze_main(self):
        self.loss_pic()
        if not self.acc_ms==None:
            self.acc_pic()
        self.memory_pic()
        self.get_report()



