import os
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

#
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import random
import collections
import tensorflow as tf

from preliminary_trials.env.cnn import CNNEnvironment


LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 50  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 1e-4  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
SCALE_FACTOR = 5.
TRAP_OCCURRED = False
DURATION_BY_SECONDS = 4 * 60 * 60
DEADLINEACHIEVED = False


def count_model(root_path='./pool'):
    tf_meta = len(os.listdir(root_path + '/meta/tf'))
    torch_meta = len(os.listdir(root_path + '/meta/torch'))
    tf_w = len(os.listdir(root_path + '/weights/tf'))
    torch_w = len(os.listdir(root_path + '/weights/torch'))
    if tf_meta == torch_meta == tf_w == torch_w:
        return tf_meta
    else:
        raise Exception('History records inconsistency occurs.')


class Model:
    def __init__(self, obs_n, act_dim, checkpoint=None):
        self.act_dim = act_dim
        self.obs_n = obs_n
        if checkpoint is None:
            self._build_model()
        else:
            self._build_model_with_checkpoint(checkpoint)

    def _build_model(self):
        hid1_size = 128
        hid2_size = 128
        # ------------------ build evaluate_net ------------------
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.obs_n)))
        # encoder_params = np.load('../autoencoder/primary_encoder.npy', allow_pickle=True)
        # encoder_dense = tf.keras.layers.Dense(32, trainable=False)
        # model.add(encoder_dense)
        # encoder_dense.set_weights(encoder_params)
        # encoder_sigmoid = tf.keras.layers.Activation('sigmoid', trainable=False)
        # model.add(encoder_sigmoid)

        model.add(tf.keras.layers.Dense(hid1_size, activation='relu', name='l1'))
        model.add(tf.keras.layers.Dense(hid2_size, activation='relu', name='l2'))
        model.add(tf.keras.layers.Dense(32, name='l3'))
        # decoder_params = np.load('../autoencoder/primary_decoder.npy', allow_pickle=True)
        decoder_dense = tf.keras.layers.Dense(self.act_dim)
        model.add(decoder_dense)
        # decoder_dense.set_weights(decoder_params)
        # decoder_sigmoid = tf.keras.layers.Activation('sigmoid')
        # decoder_sigmoid = tf.keras.layers.Softmax(-1)
        # model.add(decoder_sigmoid)
        model.summary()
        self.model = model
        # ------------------ build target_model ------------------
        target_model = tf.keras.Sequential()
        target_model.add(tf.keras.layers.Input(shape=(self.obs_n)))
        # encoder_params = np.load('../autoencoder/primary_encoder.npy', allow_pickle=True)
        # encoder_dense = tf.keras.layers.Dense(32, trainable=False)
        # target_model.add(encoder_dense)
        # encoder_dense.set_weights(encoder_params)
        # encoder_sigmoid = tf.keras.layers.Activation('sigmoid', trainable=False)
        # target_model.add(encoder_sigmoid)

        target_model.add(tf.keras.layers.Dense(hid1_size, activation='relu', name='l1'))
        target_model.add(tf.keras.layers.Dense(hid2_size, activation='relu', name='l2'))
        target_model.add(tf.keras.layers.Dense(32, name='l3'))
        # decoder_params = np.load('../autoencoder/primary_decoder.npy', allow_pickle=True)
        decoder_dense = tf.keras.layers.Dense(self.act_dim)
        target_model.add(decoder_dense)
        # decoder_dense.set_weights(decoder_params)
        # decoder_sigmoid = tf.keras.layers.Activation('sigmoid')
        # decoder_sigmoid = tf.keras.layers.Softmax(-1)
        # target_model.add(decoder_sigmoid)
        target_model.summary()
        self.target_model = target_model

        # x_np = []
        # for i in range(85):
        #     tmp = [0.] * 85
        #     tmp[i] = 1.
        #     x_np.append(tmp)
        # x = np.asarray(x_np, dtype=np.float32)
        # y = self.model(x).numpy()
        # y_max = np.max(y, -1)
        # y_min = np.min(y, -1)
        # print(y_max)
        # print(y_min)

    def _build_model_with_checkpoint(self, checkpoint):
        hid1_size = 128
        hid2_size = 128
        # ------------------ build evaluate_net ------------------
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.obs_n)))
        # encoder_dense = tf.keras.layers.Dense(32, trainable=False)
        # model.add(encoder_dense)
        # encoder_sigmoid = tf.keras.layers.Activation('sigmoid', trainable=False)
        # model.add(encoder_sigmoid)

        model.add(tf.keras.layers.Dense(hid1_size, activation='relu', name='l1'))
        model.add(tf.keras.layers.Dense(hid2_size, activation='relu', name='l2'))
        model.add(tf.keras.layers.Dense(32, name='l3'))
        # decoder_params = np.load('../autoencoder/primary_decoder.npy', allow_pickle=True)
        decoder_dense = tf.keras.layers.Dense(self.act_dim)
        model.add(decoder_dense)
        # decoder_dense.set_weights(decoder_params)
        # decoder_sigmoid = tf.keras.layers.Activation('sigmoid')
        # decoder_sigmoid = tf.keras.layers.Softmax(-1)
        # model.add(decoder_sigmoid)
        model.summary()
        model.load_weights('./dqn_model{0}'.format(checkpoint))
        self.model = model
        # ------------------ build target_model ------------------
        target_model = tf.keras.Sequential()
        target_model.add(tf.keras.layers.Input(shape=(self.obs_n)))
        # encoder_dense = tf.keras.layers.Dense(32, trainable=False)
        # target_model.add(encoder_dense)
        # encoder_sigmoid = tf.keras.layers.Activation('sigmoid', trainable=False)
        # target_model.add(encoder_sigmoid)

        target_model.add(tf.keras.layers.Dense(hid1_size, activation='relu', name='l1'))
        target_model.add(tf.keras.layers.Dense(hid2_size, activation='relu', name='l2'))
        target_model.add(tf.keras.layers.Dense(32, name='l3'))
        # decoder_params = np.load('../autoencoder/primary_decoder.npy', allow_pickle=True)
        decoder_dense = tf.keras.layers.Dense(self.act_dim)
        target_model.add(decoder_dense)
        # decoder_dense.set_weights(decoder_params)
        # decoder_sigmoid = tf.keras.layers.Activation('sigmoid')
        # decoder_sigmoid = tf.keras.layers.Softmax(-1)
        # target_model.add(decoder_sigmoid)
        target_model.summary()
        target_model.load_weights('./target_model{0}'.format(checkpoint))
        self.target_model = target_model
        print('*****************************************************************')
        print('History model loaded!!!')
        print('*****************************************************************')

        x_np = []
        for i in range(85):
            tmp = [0.] * 85
            tmp[i] = 1.
            x_np.append(tmp)
        x = np.asarray(x_np, dtype=np.float32)
        y = self.model(x).numpy()
        y_max = np.max(y, -1)
        y_min = np.min(y, -1)
        print(y_max)
        print(y_min)
        print(np.argmax(y, -1))


class DQN:
    def __init__(self, model, gamma=0.9, learning_rate=0.01):
        self.model = model.model
        self.target_model = model.target_model
        self.gamma = gamma
        self.lr = learning_rate
        # --------------------------训练模型--------------------------- #
        self.model.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.model.loss_func = tf.losses.MeanSquaredError()
        # self.model.train_loss = tf.metrics.Mean(name="train_loss")
        # ------------------------------------------------------------ #
        self.global_step = 0
        self.update_target_steps = 25  # 每隔200个training steps再把model的参数复制到target_model中

    def predict(self, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.predict(obs)

    def _train_step(self, action, features, labels):
        global LEARNING_RATE
        """ 训练步骤
        """
        with tf.GradientTape() as tape:
            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            predictions = self.model(features, training=True)
            enum_action = list(enumerate(action))
            pred_action_value = tf.gather_nd(predictions, indices=enum_action)
            loss = self.model.loss_func(labels, pred_action_value)
            print('With model loss: {0}'.format(loss.numpy()))
            # # 动态学习率
            # if tf.reduce_any(loss < 1.):
            #     LEARNING_RATE = 1e-6
            #     print('Learning_rate changed with value: {0}'.format(LEARNING_RATE))
            # elif tf.reduce_any(loss < 3.):
            #     LEARNING_RATE = 1e-5
            #     print('Learning_rate changed with value: {0}'.format(LEARNING_RATE))
            # elif tf.reduce_any(loss < 5.):
            #     LEARNING_RATE = 1e-4
            #     print('Learning_rate changed with value: {0}'.format(LEARNING_RATE))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # self.model.train_loss.update_state(loss)

    def _train_model(self, action, features, labels, epochs=1):
        """ 训练模型
        """
        for epoch in tf.range(1, epochs+1):
            self._train_step(action, features, labels)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        global TRAP_OCCURRED

        # 如果trap了 看一下dqn输出
        try:
            if TRAP_OCCURRED:
                x_np = []
                for i in range(85):
                    tmp = [0.] * 85
                    tmp[i] = 1.
                    x_np.append(tmp)
                x = np.asarray(x_np, dtype=np.float32)
                y = self.model(x).numpy()
                y_max = np.max(y, -1)
                y_min = np.min(y, -1)
                with open('./trap', 'a+') as f:
                    f.write(str(y_max))
                    f.write('\n')
                    f.write(str(y_min))
                    f.write('\n')
                    f.write(str(np.argmax(y, -1)))
                    f.write('\n')
                y = self.target_model(x).numpy()
                y_max = np.max(y, -1)
                y_min = np.min(y, -1)
                with open('./trap', 'a+') as f:
                    f.write(str(y_max))
                    f.write('\n')
                    f.write(str(y_min))
                    f.write('\n')
                    f.write(str(np.argmax(y, -1)))
                    f.write('\n')
                    f.write('\n')
        except Exception as e:
            print(e)

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.predict(next_obs)
        best_v = tf.reduce_max(next_pred_value, axis=1)
        terminal = tf.cast(terminal, dtype=tf.float32)
        target = reward + self.gamma * (1.0 - terminal) * best_v

        # 训练模型
        self._train_model(action, obs, target, epochs=1)
        self.global_step += 1
        print('Model trained once with GLOBAL_STEP={0}.'.format(self.global_step))

        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0 or TRAP_OCCURRED:
            self.replace_target()
            TRAP_OCCURRED = False

    def replace_target(self):
        print('-----------------------------------------')
        print('Try to update target_model.')
        print('-----------------------------------------')
        self.target_model.get_layer(name='l1').set_weights(self.model.get_layer(name='l1').get_weights())
        self.target_model.get_layer(name='l2').set_weights(self.model.get_layer(name='l2').get_weights())
        self.target_model.get_layer(name='l3').set_weights(self.model.get_layer(name='l3').get_weights())


class DDQN(DQN):
    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        global TRAP_OCCURRED

        # 如果trap了 看一下dqn输出
        try:
            if TRAP_OCCURRED:
                x_np = []
                for i in range(85):
                    tmp = [0.] * 85
                    tmp[i] = 1.
                    x_np.append(tmp)
                x = np.asarray(x_np, dtype=np.float32)
                y = self.model(x).numpy()
                y_max = np.max(y, -1)
                y_min = np.min(y, -1)
                with open('./trap', 'a+') as f:
                    f.write(str(y_max))
                    f.write('\n')
                    f.write(str(y_min))
                    f.write('\n')
                    f.write(str(np.argmax(y, -1)))
                    f.write('\n')
                y = self.target_model(x).numpy()
                y_max = np.max(y, -1)
                y_min = np.min(y, -1)
                with open('./trap', 'a+') as f:
                    f.write(str(y_max))
                    f.write('\n')
                    f.write(str(y_min))
                    f.write('\n')
                    f.write(str(np.argmax(y, -1)))
                    f.write('\n')
                    f.write('\n')
        except Exception as e:
            print(e)

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        # ddqn区别于dqn在于：在model中选出st最大的a -> 计算target Q
        next_pred_value_by_model = self.model(next_obs)
        action_max_by_model = tf.argmax(next_pred_value_by_model, -1)
        next_pred_value = self.target_model.predict(next_obs)
        best_v = next_pred_value[:, action_max_by_model]
        terminal = tf.cast(terminal, dtype=tf.float32)
        target = reward + self.gamma * (1.0 - terminal) * best_v

        # 训练模型
        self._train_model(action, obs, target, epochs=1)
        self.global_step += 1
        print('Model trained once with GLOBAL_STEP={0}.'.format(self.global_step))

        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0 or TRAP_OCCURRED:
            self.replace_target()
            TRAP_OCCURRED = False


class Agent:
    def __init__(self, act_dim, algorithm, e_greed=0.1, e_greed_decrement=0, e_greed_decrement_t=0):
        self.act_dim = act_dim
        self.algorithm = algorithm
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
        # 恢复历史数据时，同样恢复e_greedy指数
        self.e_greed = max(0.1, self.e_greed - self.e_greed_decrement * e_greed_decrement_t)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            sample1 = np.random.rand()
            if sample1 < 0.4:
                act = self.predict(obs)
            else:
                act_lst = self.predict_top_k(obs, 7)
                act = act_lst[:, np.random.randint(1, 7)].item()
            # act = self.predict(obs)

        self.e_greed = max(
            0.1, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):
        obs = tf.expand_dims(obs, axis=0)
        if obs.dtype != tf.float32:
            obs = tf.cast(obs, tf.float32)
        action = self.algorithm.model(obs)
        return np.argmax(action)

    def predict_top_k(self, obs, k):
        obs = tf.expand_dims(obs, axis=0)
        if obs.dtype != tf.float32:
            obs = tf.cast(obs, tf.float32)
        action = self.algorithm.model(obs)
        sort_res = np.argsort(action)
        return sort_res[-k:]


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)
        with open('./pool/sarsd/history', 'a') as f:
            f.write(str(exp) + '\n')

    def load_history(self):
        # 历史数据池存储在pool的sarsd下
        path = './pool/sarsd/history'
        record_nums = 0
        if os.path.isfile(path):
            with open(path, 'r') as f:
                records = f.readlines()
            for r in records:
                tup = eval(r)
                if not isinstance(tup, tuple):
                    raise Exception('Some impurity occurs in history data of pool.')
                self.buffer.append(tup)
                record_nums += 1
        print('*****************************************************************')
        print('{0} samples loaded from history pool data!!!'.format(record_nums))
        print('*****************************************************************')
        return record_nums

    def sample(self, batch_size):
        # 判断是否发生trap
        if TRAP_OCCURRED:
            # 确保抑制trap的信息被训练
            restrain_exp = self.buffer[-1]
            # 虽然有小概率选中两个抑制当前trap项
            # 不过问题不大
            mini_batch = random.sample(self.buffer, batch_size - 1)
            s, a, r, s_p, done = restrain_exp
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [s], [a], [r], [s_p], [done]

            for experience in mini_batch:
                s, a, r, s_p, done = experience
                obs_batch.append(s)
                action_batch.append(a)
                reward_batch.append(r)
                next_obs_batch.append(s_p)
                done_batch.append(done)
        else:
            mini_batch = random.sample(self.buffer, batch_size)
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

            for experience in mini_batch:
                s, a, r, s_p, done = experience
                obs_batch.append(s)
                action_batch.append(a)
                reward_batch.append(r)
                next_obs_batch.append(s_p)
                done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('int32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)


def run_episode(env, algorithm, agent, rpm, start_time=None, start_time_without_pool=None):
    global TRAP_OCCURRED, DEADLINEACHIEVED
    step = 0
    total_reward = 0
    obs = env.reset(random_start=True)

    while True:
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # 判断trap有无发生
        # 加一个判断warmup判断以防万一
        # 但一般是不会用到的，因为还未开始训练的模型不太可能出现trap问题
        if len(rpm) > MEMORY_WARMUP_SIZE and reward == -SCALE_FACTOR * 4 and done:
            print('TRAP OCCURRED!!!!!')
            TRAP_OCCURRED = True

        # 到固定训练数或trap发生了就训练一次
        if len(rpm) > MEMORY_WARMUP_SIZE and (step % LEARN_FREQ == 0 or TRAP_OCCURRED):
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = rpm.sample(BATCH_SIZE)
            algorithm.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        obs = next_obs
        total_reward += reward
        if done:
            break
        # 此时已经进入模型生成阶段，要判断时间
        if start_time_without_pool is not None:
            cur_time = time.time()
            if DEADLINEACHIEVED:
                # 包括池准备时间已经到了4小时
                if cur_time - start_time_without_pool >= DURATION_BY_SECONDS:
                    with open('./num1', 'w+') as f:
                        f.write(str((cur_time - start_time_without_pool) / 3600))
                    sys.exit()
            else:
                if cur_time - start_time >= DURATION_BY_SECONDS:
                    tf_meta = count_model()
                    with open('./num0', 'w+') as f:
                        f.write(str(tf_meta) + '\n')
                        f.write(str((cur_time - start_time) / 3600))
                    DEADLINEACHIEVED = True
                    sys.exit()

    return total_reward


def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done = env.step(action)
            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


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

    env = CNNEnvironment('cifar100', 'train', 5, 100, scale_factor=SCALE_FACTOR)
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池
    history_records_num = rpm.load_history()
    model = Model(obs_shape, action_dim)
    algorithm = DDQN(model, gamma=GAMMA, learning_rate=LEARNING_RATE)
    agent = Agent(action_dim, algorithm, e_greed=0.5, e_greed_decrement=1e-2, e_greed_decrement_t=history_records_num)

    start_time = time.time()

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, algorithm, agent, rpm)
        print('Current size of rpm: {0}'.format(len(rpm)))

    max_episode = 1000

    # 开始训练
    episode = 0
    start_time_without_pool = time.time()
    print(start_time_without_pool)
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        total_reward = 0
        for i in range(0, 10):
            print('i={0}'.format(i))
            episode_reward = run_episode(env, algorithm, agent, rpm, start_time, start_time_without_pool)
            print('Current episode get reward with values {0}'.format(episode_reward))
            total_reward += episode_reward
            episode += 1

        # 测试
        # eval_reward = evaluate(env, agent)
        print('episode:{}   e_greed:{}   reward:{}'.format(episode, agent.e_greed, total_reward))
        # 保存
        # save_path = './dqn_model' + str(episode)
        # target_save_path = './target_model' + str(episode)
        # model.model.save_weights(save_path)
        # model.target_model.save_weights(target_save_path)

    # 训练结束，保存模型
    save_path = './dqn_model'
    target_save_path = './target_model'
    model.model.save_weights(save_path)
    model.target_model.save_weights(target_save_path)
