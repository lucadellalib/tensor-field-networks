# -*- coding: utf-8 -*-
"""
    Evaluate classification performance with optional voting.
"""

# Source code modified from:
# 	Title: charlesq34/pointnet2
# 	Author: Charles R. Qi (charlesq34)
# 	Date: 2017
# 	Availability: https://github.com/charlesq34/pointnet2/tree/42926632a3c33461aebfbee2d829098b30a23aaa

import os
import sys
import argparse
import math
from datetime import datetime
import importlib

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "modelnet"))

import preprocessing
import dataloader


parser = argparse.ArgumentParser()
#parser.add_argument("--gpu", type=int, default=0, help="GPU to use [default: GPU 0]")
parser.add_argument("--model", default="tfn_4_channels", help="Model name [default: tfn_4_channels]")
parser.add_argument("--num_point", type=int, default=16, help="Number of points [default: 16]")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size during testing [default: 32]")
parser.add_argument("--model_path", default="log/tfn_4_channels/model.ckpt", help="model checkpoint file path [default: log/tfn_4_channels/model.ckpt]")
parser.add_argument("--dump_dir", default="dump/tfn_4_channels", help="Dump dir [default: dump]")
parser.add_argument("--num_votes", type=int, default=1, help="Aggregate classification scores from multiple rotations [default: 1]")
FLAGS = parser.parse_args()

#GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, "log_evaluate.txt"), "w")
LOG_FOUT.write(str(FLAGS) + "\n")

NUM_CLASSES = 10
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(ROOT_DIR, "modelnet/data/modelnet40_normal_resampled/modelnet10_shape_names.txt"))]

# ModelNet10 official train/test split
assert(NUM_POINT <= 10000)
DATA_PATH = os.path.join(ROOT_DIR, "modelnet/data/modelnet40_normal_resampled")
TRAIN_DATASET = dataloader.ModelNetDataloader(root=DATA_PATH, npoints=NUM_POINT, split="train", modelnet10=True, normal_channel=False, batch_size=BATCH_SIZE)
TEST_DATASET = dataloader.ModelNetDataloader(root=DATA_PATH, npoints=NUM_POINT, split="test", modelnet10=True, normal_channel=False, batch_size=BATCH_SIZE)


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


def evaluate(num_votes):
    is_training = False
     
    #with tf.device("/gpu:"+str(GPU_INDEX)):
    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
    is_training_pl = tf.placeholder(tf.bool, shape=())

    # model
    pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
    MODEL.get_loss(pred, labels_pl, end_points)
    losses = tf.get_collection("losses")
    total_loss = tf.add_n(losses, name="total_loss")

    # add ops to save and restore all the variables.
    saver = tf.train.Saver()
        
    # create a session
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.allow_soft_placement = True
    #config.log_device_placement = False
    sess = tf.Session()

    # restore variables from disk
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {"pointclouds_pl": pointclouds_pl,
           "labels_pl": labels_pl,
           "is_training_pl": is_training_pl,
           "pred": pred,
           "loss": total_loss}

    eval_one_epoch(sess, ops, num_votes)


def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False

    # make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        print("Batch: %03d, batch size: %d"%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            # shuffle point order to achieve different farthest samplings
            shuffled_indices = np.arange(NUM_POINT)
            np.random.shuffle(shuffled_indices)
           
            rotated_data = preprocessing.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                    vote_idx / float(num_votes) * np.pi * 2)
            feed_dict = {ops["pointclouds_pl"]: rotated_data,
                         ops["labels_pl"]: cur_batch_label,
                         ops["is_training_pl"]: is_training}
            loss_val, pred_val = sess.run([ops["loss"], ops["pred"]], feed_dict=feed_dict)
            batch_pred_sum += pred_val
        pred_val = np.argmax(batch_pred_sum, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
    
    log_string("eval mean loss: %f" % (loss_sum / float(batch_idx)))
    log_string("eval accuracy: %f"% (total_correct / float(total_seen)))
    log_string("eval avg class acc: %f" % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string("%10s:\t%0.3f" % (name, class_accuracies[i]))


if __name__=="__main__":
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
