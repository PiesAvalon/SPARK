import os
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import tensorflow as tf

from preliminary_trials.env.cnn import CNNEnvironment


SCALE_FACTOR = 5.
DURATION_BY_SECONDS = 4 * 60 * 60


def count_model(root_path='./pool'):
    tf_meta = len(os.listdir(root_path + '/meta/tf'))
    torch_meta = len(os.listdir(root_path + '/meta/torch'))
    tf_w = len(os.listdir(root_path + '/weights/tf'))
    torch_w = len(os.listdir(root_path + '/weights/torch'))
    if tf_meta == torch_meta == tf_w == torch_w:
        return tf_meta
    else:
        raise Exception('History records inconsistency occurs.')


def run_episode(env, start_time=None):
    step = 0
    obs = env.reset(random_start=True)

    while True:
        step += 1
        action = random.randint(0, 83)
        _, _, done = env.step(action)

        obs = _
        if done:
            break

        if start_time is not None:
            cur_time = time.time()
            if cur_time - start_time >= DURATION_BY_SECONDS:
                tf_meta = count_model()
                with open('./num0', 'w+') as f:
                    f.write(str(tf_meta) + '\n')
                    f.write(str((cur_time - start_time) / 3600))
                sys.exit()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.config.experimental_run_functions_eagerly(True)

    action_dim = 84
    obs_shape = 85

    env = CNNEnvironment('mnist', 'train', 5, 100, scale_factor=SCALE_FACTOR)

    start_time = time.time()

    max_episode = 10000

    # 开始训练
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        total_reward = 0
        for i in range(0, 10):
            print('i={0}'.format(i))
            run_episode(env, start_time)
            episode += 1

        print('episode:{0}'.format(episode))
