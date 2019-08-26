# -*- coding: utf-8 -*-
"""
    Single-GPU training.
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
parser.add_argument("--max_epoch", type=int, default=250, help="Epoch to run [default: 250]")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size during training/evaluation [default: 32]")
parser.add_argument("--optimizer", default="adam", help="Optimizer (adam | momentum) [default: adam]")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate [default: 0.001]")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (used only for momentum optimizer) [default: 0.9]")
parser.add_argument("--decay_step", type=int, default=200000, help="Decay step for learning rate decay [default: 200000]")
parser.add_argument("--decay_rate", type=float, default=0.7, help="Decay rate for learning rate decay [default: 0.7]")
parser.add_argument("--log_dir", default="log/tfn_4_channels", help="Log dir [default: log/tfn_4_channels]")
FLAGS = parser.parse_args()

EPOCH_CNT = 0

#GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, "models", FLAGS.model + ".py")
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
OPTIMIZER = FLAGS.optimizer
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system("cp %s %s" % (MODEL_FILE, LOG_DIR)) # backup of model definition
os.system("cp train.py %s" % (LOG_DIR)) # backup of traininig procedure
LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "w")
LOG_FOUT.write(str(FLAGS)+"\n")

NUM_CLASSES = 10

# ModelNet10 official train/test split
assert(NUM_POINT <= 10000)
DATA_PATH = os.path.join(ROOT_DIR, "modelnet/data/modelnet40_normal_resampled")
TRAIN_DATASET = dataloader.ModelNetDataloader(root=DATA_PATH, npoints=NUM_POINT, split="train", modelnet10=True, normal_channel=False, batch_size=BATCH_SIZE)
TEST_DATASET = dataloader.ModelNetDataloader(root=DATA_PATH, npoints=NUM_POINT, split="test", modelnet10=True, normal_channel=False, batch_size=BATCH_SIZE)


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # base learning rate
                        batch * BATCH_SIZE,  # current index into the dataset
                        DECAY_STEP,          # decay step
                        DECAY_RATE,          # decay rate
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE
    return learning_rate        


def train():
    with tf.Graph().as_default():
        # with tf.device("/gpu:" + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        
        batch = tf.get_variable("batch", [],
            initializer=tf.constant_initializer(0), trainable=False)

        # get model and loss 
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection("losses")
        total_loss = tf.add_n(losses, name="total_loss")
        tf.summary.scalar("total_loss", total_loss)
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name, l)

        correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
        tf.summary.scalar("accuracy", accuracy)

        print("--- Get training operator")
        # get training operator
        learning_rate = get_learning_rate(batch)
        tf.summary.scalar("learning_rate", learning_rate)
        if OPTIMIZER == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=batch)
        
        # add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # create a session
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #config.allow_soft_placement = True
        #config.log_device_placement = False
        sess = tf.Session()#config=config)

        # add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "train"), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "test"), sess.graph)

        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {"pointclouds_pl": pointclouds_pl,
               "labels_pl": labels_pl,
               "is_training_pl": is_training_pl,
               "pred": pred,
               "loss": total_loss,
               "train_op": train_op,
               "merged": merged,
               "step": batch,
               "end_points": end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string("**** EPOCH %03d ****" % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            curr_acc = eval_one_epoch(sess, ops, test_writer)

            # save the variables with best test accuracy to disk.
            if curr_acc > best_acc:
                best_acc = curr_acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
    log_string("Model saved in file: %s" % save_path)
  

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # make sure batch data is of the same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0

    log_string(str(datetime.now()))
    log_string("---- EPOCH %03d TRAINING ----"%(EPOCH_CNT))

    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)
        #batch_data = preprocessing.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops["pointclouds_pl"]: cur_batch_data,
                     ops["labels_pl"]: cur_batch_label,
                     ops["is_training_pl"]: is_training}
        summary, step, _, loss_val, pred_val = sess.run([ops["merged"], ops["step"],
            ops["train_op"], ops["loss"], ops["pred"]], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        if (batch_idx + 1) % 10 == 0:
            log_string(" ---- batch: %03d ----" % (batch_idx + 1))
            log_string("mean loss: %f" % (loss_sum / 10))
            log_string("accuracy: %f" % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT, TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string("---- EPOCH %03d EVALUATION ----"%(EPOCH_CNT))
    
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops["pointclouds_pl"]: cur_batch_data,
                     ops["labels_pl"]: cur_batch_label,
                     ops["is_training_pl"]: is_training}
        summary, step, loss_val, pred_val = sess.run([ops["merged"], ops["step"],
            ops["loss"], ops["pred"]], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
    
    log_string("eval mean loss: %f" % (loss_sum / float(batch_idx)))
    log_string("eval accuracy: %f"% (total_correct / float(total_seen)))
    log_string("eval avg class acc: %f" % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return total_correct / float(total_seen)


if __name__ == "__main__":
    log_string("pid: %s"%(str(os.getpid())))
    train()
    LOG_FOUT.close()
