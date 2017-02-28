# coding = utf-8
"""author = jingyuan zhang"""
from data_util import DataUtil
from config import Config
import tensorflow as tf
import os
import random
import numpy as np
from stock_scraper import StockScraper
from config import ASingleStockConfig
import time

class Reinforcer:
    def __init__(self):
        self.config = Config()
        self.sc = StockScraper(ASingleStockConfig())
        self.du = DataUtil(self.config)
        self.memories = []
        self.W1 = tf.get_variable('W1', [self.config.INPUT, self.config.M1])
        self.b1 = tf.get_variable('b1', [self.config.M1])
        self.W2 = tf.get_variable('W2', [self.config.M1, self.config.M2])
        self.b2 = tf.get_variable('b2', [self.config.M2])
        self.W3 = tf.get_variable('W3', [self.config.M2, 1])
        self.b3 = tf.get_variable('b3', [1])
        self.fund = 1000000
        # self.init_op = tf.initialize_all_variables()
        self.init_placeholder()
        scores = self.batch_scoring_op()
        next_step_scores = self.batch_predict_op()
        self.add_loss_n_train_op(scores, next_step_scores)
        self.add_step_predict_op()
        self.saver = tf.train.Saver()
        self.init_op = tf.initialize_all_variables()

    def init_placeholder(self):
        self.states = tf.placeholder(tf.float32)
        self.rewards = tf.placeholder(tf.float32)
        self.states_next = tf.placeholder(tf.float32)

    def batch_scoring_op(self):
        x = tf.reshape(self.states, (self.config.BATCH_SIZE, self.config.INPUT))
        scores = self.Q_network_op(x)
        return scores

    def add_loss_n_train_op(self, scores, next_scores):
        self.predict_scores = self.config.gamma * tf.reduce_max(next_scores, 1)
        self.viewing_scores = scores
        '''sarsa reward better?'''
        self.losses = (self.rewards + self.predict_scores - scores) ** 2
        self.loss = tf.reduce_sum(self.losses)
        optimizer = tf.train.RMSPropOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)

    def add_step_predict_op(self):
        x = tf.reshape(self.states_next, (self.config.STOCK_AMOUNT, self.config.INPUT))
        scores = self.Q_network_op(x)
        self.prediction = tf.argmax(tf.reshape(scores, (-1, self.config.STOCK_AMOUNT + 1)), axis=1)[0]

    def Q_network_op(self, x):
        fc1 = tf.matmul(x, self.W1) + self.b1
        relu1 = tf.nn.tanh(fc1)
        relu1 = tf.nn.dropout(relu1, self.config.DROPOUT)
        fc2 = tf.matmul(relu1, self.W2) + self.b2
        relu2 = tf.nn.tanh(fc2)
        relu2 = tf.nn.dropout(relu2, self.config.DROPOUT)
        scores = tf.matmul(relu2, self.W3) + self.b3
        scores = tf.squeeze(scores)
        return scores

    def batch_predict_op(self):
        x = tf.reshape(self.states_next, (self.config.BATCH_SIZE * (self.config.STOCK_AMOUNT + 1), self.config.INPUT))
        scores = self.Q_network_op(x)
        scores = tf.reshape(scores, (self.config.BATCH_SIZE, self.config.STOCK_AMOUNT + 1))
        return scores

    def build_feed_dict(self, random_memories):
        feed = {}
        feed[self.states] = [m[0] for m in random_memories]
        # print feed[self.states], len(feed[self.states])
        states_next = []
        next_rewards = []
        for rm in random_memories:
            step_ind = rm[-1]
            portfolio = rm[-2]
            for i in range(self.config.STOCK_AMOUNT + 1):
                if i == self.config.STOCK_AMOUNT:
                    # print sum(portfolio)
                    updated_state, reward, _, _ = self.du.calc_state(portfolio, step_ind, True)
                else:
                    # print sum(portfolio)

                    _, updated_state, reward, _, _ = self.du.update_state_after_operation(portfolio, i, step_ind + 1)
                states_next.append(updated_state)
                # next_rewards.append(reward)
                # print len(updated_state)

        feed[self.states_next] = states_next
        # print len(feed[self.states_next])
        feed[self.rewards] = [m[2] for m in random_memories]
        # print feed[self.rewards], len(feed[self.rewards])
        return feed

    def run_epoch(self, session, save=None, load=None):
        if not os.path.exists('./save'):
            os.makedirs('./save')
        if load:
            self.saver.restore(session, load)
        else:
            session.run(self.init_op)

        timestamp = 0
        while True:
            time.sleep(self.sc.config.time_interval)
            data = self.sc.request_api()
            self.config.INPUT = len(data)
            self.memories.append(data)
            if len(self.memories) > 10*self.config.BATCH_SIZE:
                startind = random.randint(0,len(self.memories)-self.config.BATCH_SIZE)
                batch = self.memories[startind:startind+self.config.BATCH_SIZE]
                feed = self.build_feed_dict(batch)






if __name__ == '__main__':
    cc = Reinforcer()
    with tf.Session() as sess:
        cc.run_epoch(sess)






