import os
import glob
import json
import random
from openpyxl import Workbook

import numpy as np
from matplotlib import pyplot as plt


class RecordsAnalyser:
    def __init__(self, state_num):
        self.root_path = './pool10'
        self.state_num = state_num

    def analyse(self):
        sarsd_records = self.count_records()
        self.analyse_sarsd(sarsd_records)
        self.analyse_meta()

    def count_records(self):
        # 记录池
        with open(self.root_path + '/sarsd/history', 'r') as f:
            records = f.readlines()
        print('{0} history records found for your pool.'.format(len(records)))
        tf_meta = len(os.listdir(self.root_path + '/meta/tf'))
        torch_meta = len(os.listdir(self.root_path + '/meta/torch'))
        tf_w = len(os.listdir(self.root_path + '/weights/tf'))
        torch_w = len(os.listdir(self.root_path + '/weights/torch'))
        if tf_meta == torch_meta == tf_w == torch_w:
            print('{0} available model cases found for you.'.format(tf_meta))
        else:
            raise Exception('History records inconsistency occurs.')
        return records

    def analyse_sarsd(self, records):
        state_num = [0] * self.state_num
        action_num = [0] * self.state_num
        reward = 0.
        positive_reward = 0.
        positive_num = 0
        reward_lst = []
        zero_reward_num = 0
        for r in records:
            tup = eval(r)
            state_num[np.argmax(np.asarray(tup[0])).item()] += 1
            action_num[tup[1]] += 1
            reward += tup[2]
            if tup[2] > 0.:
                positive_reward += tup[2]
                positive_num += 1
            elif tup[2] == 0.:
                zero_reward_num += 1
            reward_lst.append(tup[2])
            pre = tup[4]
        avg_reward = reward / len(records)
        avg_positive_reward = positive_reward / positive_num
        print('The total average reward value of the records is {0}'.format(avg_reward))
        print('The total average positive reward value of the records is {0}'.format(avg_positive_reward))
        print('{0} zero reward episode.'.format(zero_reward_num))

        # 画图 st
        x = [i for i in range(self.state_num)]
        plt.bar(x=x, width=0.2, height=state_num, align='center')
        plt.xticks(np.arange(len(x)), x, fontsize=3)
        plt.title('st distribution of the existed records')
        plt.show()

        # 画图 a
        plt.bar(x=x, width=0.2, height=action_num, align='center')
        plt.xticks(np.arange(len(x)), x, fontsize=3)
        plt.title('a distribution of the existed records')
        plt.show()

        # 画图 reward趋势
        plt.title('Trend of reward during episode')
        plt.plot(np.arange(0, len(reward_lst)), reward_lst)
        plt.show()

    def analyse_meta(self):
        avg_duration = 0.
        layer_num = [0] * 15
        type_num = [0] * 6
        type_reward = [0.] * 6
        crashed_lst = []
        value_encoder = {
            'INCONSISTENCY': 0,
            'LOSS-NAN/INF': 1,
            'OUTPUT-NAN/INF': 2,
            'TF-OUTPUT-NAN/INF': 3,
            'TORCH-OUTPUT-NAN/INF': 4,
            'CRASH': 5
        }
        value_decoder = {
            'INCONSISTENCY': 0,
            'LOSS-NAN/INF': 1,
            'OUTPUT-NAN/INF': 2,
            'TF-OUTPUT-NAN/INF': 3,
            'TORCH-OUTPUT-NAN/INF': 4,
            'CRASH': 5
        }
        c = 0
        for name in glob.iglob(self.root_path + '/meta/tf/*'):
            c += 1
            with open(name, 'r') as f:
                record = json.load(f)
            # 耗时
            duration = record['time']
            avg_duration += duration
            # 层数
            network = record['json']
            num = len(network['network'])
            layer_num[num] += 1
            # type reward
            type_idx = value_encoder[record['type']]
            if type_idx == 3:
                crashed_lst.append(name)
            reward = record['reward']
            type_num[type_idx] += 1
            type_reward[type_idx] += reward
        for i in range(4):
            type_reward[i] = type_reward[i] / type_num[i] if type_num[i] != 0. else 0.
        avg_duration /= c
        print('{0}s is cost for finding a model with error in average.'.format(avg_duration))
        # 错误类型bar
        x = ['INCONSISTENCY', 'LOSS-NAN/INF', 'OUTPUT-NAN/INF', 'TF-OUTPUT-NAN/INF', 'TORCH-OUTPUT-NAN/INF', 'CRASH']
        plt.bar(x, type_num, align='center')
        plt.title('error type distribution of the generated models')
        plt.show()
        # 层数bar
        res = 0
        for i in range(9):
            res += (i + 1) * layer_num[i]
        res = res / c
        x = [i + 1 for i in range(15)]
        plt.bar(x, layer_num, align='center')
        plt.title('depth distribution of the generated models')
        plt.show()
        print('The average depth of the generated model is {0}.'.format(res))
        # type reward
        for i in range(4):
            print('The average reward value for error {0} is {1}'.format(value_decoder[i], type_reward[i]))
        # 写入crash案例路径
        if len(crashed_lst) != 0:
            with open('./crash', 'w+') as f:
                for p in crashed_lst:
                    f.write(p + '\n')

    def check_concrete_st_meta(self, st):
        if isinstance(st, int):
            st = str(st)
        for name in glob.iglob(self.root_path + '/meta/tf/*'):
            with open(name, 'r') as f:
                record = json.load(f)
            if record['st+1'] == st:
                print('reward {0} with type {1}'.format(record['reward'], record['type']))

    def check_concrete_a_sarsd(self, records, a):
        for r in records:
            tup = eval(r)
            if a == tup[1]:
                print('reward: {0}'.format(tup[2]))

    def count_independent_model(self):
        exist_lst = []
        for name in glob.iglob(self.root_path + '/meta/tf/*'):
            with open(name, 'r') as f:
                record = json.load(f)
            network = record['json']
            network = network['network']
            flag = True
            for x in exist_lst:
                if x == network:
                    flag = False
                    break
            if flag:
                exist_lst.append(network)
        print('{0} different models are generated.'.format(len(exist_lst)))




if __name__ == '__main__':
    analyser = RecordsAnalyser(85)
    analyser.count_records()
    analyser.count_independent_model()
    # analyser.analyse_meta()
    # analyser.analyse()
    # analyser.check_concrete_a_sarsd(analyser.count_records(), 38)
    # analyser.check_concrete_st_meta(38)
