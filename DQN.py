#!/usr/bin/env python
from __future__ import print_function

import re
import tensorflow as tf
import cv2
import math
import sys
sys.path.append("game/")
import Wormy as game
import random
import numpy as np
from collections import deque

LR = 1e-6
GAME = 'wormy'
ACTIONS = 4
GAMMA = 0.99
OBSERVE = 50000
EXPLORE = 3000000
REPLAY_MEMORY = 1000000
BATCH = 64
FRAME_PER_ACTION = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def createNetwork():
    W_conv1 = weight_variable([7, 7, 12, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    W_fc1 = weight_variable([2048, 512])
    b_fc1 = bias_variable([512])
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    s = tf.placeholder("float", [None, 64, 64, 12])
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 2048])
    h_fc1 = tf.matmul(h_conv3_flat, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(h_fc1)
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout

def trainNetwork(s, readout, sess):
    loss = 0
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(LR).minimize(cost)

    game_state = game.GameState()
    D = deque()
    D1 = deque()


    do_nothing = np.zeros(ACTIONS)
    x_t, r_0, terminal, flag_timeout = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t,(64,64)),cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(x_t,cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv,np.array([0,43,46]),np.array([179,255,255]))
    x_t = cv2.bitwise_and(x_t,x_t,mask = mask)
    s_t = np.concatenate((x_t,x_t,x_t,x_t),2)
    
    t = 1
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        string = re.findall("\d*",checkpoint.model_checkpoint_path)
        t = int(string[25])
    else:
        print("Could not find old network weights")


    while game_state.getLifes() <= game.TOTAL_LIFES:
        epsilon = game_state.getEpsilon()
        Alpha = game_state.getAlpha()
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                while(True):
                    action_index = random.randrange(ACTIONS)
                    if(action_index + game_state.getDirection() != 3):
                        break
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                if(action_index + game_state.getDirection() == 3):
                    action_index = game_state.getDirection()
                a_t[action_index] = 1
        
        x_t1_colored, r_t, terminal, flag_timeout = game_state.frame_step(a_t)
        x_t1_origin = cv2.cvtColor(x_t1_colored,cv2.COLOR_BGR2RGB)
        x_t1 = cv2.resize(x_t1_origin,(64,64))
        hsv = cv2.cvtColor(x_t1,cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv,np.array([0,43,46]),np.array([179,255,255]))
        
        x_t1_final = cv2.bitwise_and(x_t1,x_t1,mask = mask)
        
        s_t1 = np.append(x_t1_final,s_t[:,:,:9],axis=2)
        if abs(r_t) >= 0.5 and flag_timeout == False:
            D.append((s_t, a_t, r_t, s_t1, terminal))
        elif r_t != 0:
            if(flag_timeout):
                for i in range( math.floor( game_state.getLength() * 0.7 + 10 ) ):
                    temp = D1.pop()
                    Ps_t = temp[0]; Pa_t = temp[1]; Ps_t1 = temp[3]; Pterminal = temp[4]
                    Pr_t = temp[2] - 0.5/game_state.getLength()
                    if(Pr_t < -1):
                        Pr_t = -1
                    D1.append((Ps_t, Pa_t, Pr_t, Ps_t1, Pterminal))
            D1.append((s_t, a_t, r_t, s_t1, terminal))
        
        if len(D) > (REPLAY_MEMORY/2):
            D.popleft()
        if len(D1) > (REPLAY_MEMORY/2):
            D1.popleft()

        if t > OBSERVE and len(D) + len(D1) > BATCH:
            if len(D) >= math.ceil(Alpha * BATCH):
                minibatch1 = random.sample(D, math.ceil(Alpha * BATCH))
                minibatch2 = random.sample(D1,math.floor((1-Alpha) * BATCH))
            else:
                minibatch1 = random.sample(D, len(D))
                minibatch2 = random.sample(D1, BATCH - len(D))
                
            minibatch = minibatch1 + minibatch2
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
            loss = cost.eval(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})


        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        if action_index == 0:
            ACT = "left"
        elif action_index ==1:
            ACT = "up"
        elif action_index ==2:
            ACT = "down"
        elif action_index ==3:
            ACT = "right"
        print("TIMESTEP", t, "/ Lifes", game_state.getLifes(), \
            "/ EPSILON", epsilon, "/ Loss", loss, "/ ACTION", ACT, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        s_t = s_t1
        t += 1
        
def playGame():
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
