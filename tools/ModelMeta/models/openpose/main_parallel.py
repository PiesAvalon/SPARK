import json
import cv2
import mindspore
import numpy as np
from tqdm import tqdm
from eval import detect, draw_person_pose, evaluate_mAP
from src.loss import openpose_loss, BuildTrainNetwork
from src.utils import get_lr
from src.model_utils.config import config
from ast import literal_eval as liter
import os
os.environ['GLOG_v'] = '3'
from mindspore.nn.optim import Adam, Momentum
from src.dataset import create_dataset, valdata
from src.openposenet import OpenPoseNet
config.lr = liter(config.lr)
config.outputs_dir = config.save_model_path
config.outputs_dir = os.path.join(config.outputs_dir, "ckpt_0/")
config.rank = 0
device_num = 1
config.max_epoch = config.max_epoch_train
config.lr_steps = list(map(int, config.lr_steps.split(',')))
config.group_size = 1


if __name__ == '__main__':
    print('start create network')
    criterion = openpose_loss()
    criterion.add_flags_recursive(fp32=True)
    network = OpenPoseNet(vggpath=config.vgg_path, vgg_with_bn=config.vgg_with_bn)
    train_net = BuildTrainNetwork(network, criterion)
    de_dataset_train = create_dataset(config.jsonpath_train, config.imgpath_train, config.maskpath_train,
                                      batch_size=config.batch_size,
                                      rank=config.rank,
                                      group_size=1,
                                      num_worker=1,
                                      multiprocessing=False,
                                      shuffle=True,
                                      repeat_num=1)
    steps_per_epoch = de_dataset_train.get_dataset_size()
    lr_stage, lr_base, lr_vgg = get_lr(config.lr * device_num,
                                       config.lr_gamma,
                                       steps_per_epoch,
                                       config.max_epoch,
                                       config.lr_steps,
                                       device_num,
                                       lr_type=config.lr_type,
                                       warmup_epoch=config.warmup_epoch)
    print("steps_per_epoch: ", steps_per_epoch)
    vgg19_base_params = list(filter(lambda x: 'base.vgg_base' in x.name, train_net.trainable_params()))
    base_params = list(filter(lambda x: 'base.conv' in x.name, train_net.trainable_params()))
    stages_params = list(filter(lambda x: 'base' not in x.name, train_net.trainable_params()))

    group_params = [{'params': vgg19_base_params, 'lr': lr_vgg},
                    {'params': base_params, 'lr': lr_base},
                    {'params': stages_params, 'lr': lr_stage}]

    if config.optimizer == "Momentum":
        opt = Momentum(group_params, learning_rate=lr_stage, momentum=0.9)
    elif config.optimizer == "Adam":
        opt = Adam(group_params)
    else:
        raise ValueError("optimizer not support.")


    def forward_fn(data, label, l1, l2):
        loss = train_net(data, label, l1, l2)
        return loss


    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)


    def train_step(data, label, l1, l2):
        (loss), grads = grad_fn(data, label, l1, l2)
        loss = mindspore.ops.depend(loss, opt(grads))
        return loss


    dataset = valdata(config.ann, config.imgpath_val, config.rank, config.group_size, mode='val')
    dataset_size = dataset.get_dataset_size()
    de_dataset = dataset.create_tuple_iterator()

    print("eval dataset size: ", dataset_size)
    epoch_num = 60
    per_batch = 200
    losses_ms_avg1 = []
    for epoch in range(epoch_num):
        nums = 0
        losses_ms = []
        for data in de_dataset_train:
            print("data[0].shape: ", data[0].shape)
            nums += data[0].shape[0]
            loss_ms = train_step(data[0], data[1], data[2], data[3])
            if nums % per_batch == 0:
                print("batch:" + str(nums) + " ms_loss1:" + str(
                    loss_ms.asnumpy()))
            losses_ms.append(loss_ms.asnumpy())
            break
        losses_ms_avg1.append(np.mean(losses_ms))
        print("epoch {}: ".format(epoch), " ms_loss1: ",
              str(np.mean(losses_ms)))

        kpt_json = []
        for _, (img, img_id) in tqdm(enumerate(de_dataset), total=dataset_size):
            img = img.asnumpy()
            img_id = int((img_id.asnumpy())[0])
            poses, scores = detect(img, network)

            if poses.shape[0] > 0:
                for index, pose in enumerate(poses):
                    data = dict()

                    pose = pose[[0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10, 1], :].round().astype('i')

                    keypoints = pose.reshape(-1).tolist()
                    keypoints = keypoints[:-3]
                    data['image_id'] = img_id
                    data['score'] = scores[index]
                    data['category_id'] = 1
                    data['keypoints'] = keypoints
                    kpt_json.append(data)
            else:
                print("Predict poses size is zero.", flush=True)
            img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)

            save_path = os.path.join(config.output_img_path, str(img_id) + ".png")
            cv2.imwrite(save_path, img)

        result_json = 'eval_result.json'
        with open(os.path.join(config.output_img_path, result_json), 'w') as fid:
            json.dump(kpt_json, fid)
        res = evaluate_mAP(os.path.join(config.output_img_path, result_json), ann_file=config.ann)
        print('result: ', res)
