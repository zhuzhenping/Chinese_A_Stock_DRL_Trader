# coding = utf-8
"""author = jingyuan zhang"""
from config import Config
import tensorflow as tf
import os
import random
import numpy as np
from stock_scraper import StockScraper
from data_util import DataUtil
from config import ASingleStockConfig
import time
import sys

class Reinforcer:
    def __init__(self):
        self.config = Config()
        self.du = DataUtil(self.config)
        self.sc = StockScraper(ASingleStockConfig())
        self.memories = []
        self.W1 = tf.get_variable('W1', [self.config.INPUT, self.config.M1])
        self.b1 = tf.get_variable('b1', [self.config.M1])
        self.W2 = tf.get_variable('W2', [self.config.M1, self.config.M2])
        self.b2 = tf.get_variable('b2', [self.config.M2])
        self.W3 = tf.get_variable('W3', [self.config.M2, 1])
        self.b3 = tf.get_variable('b3', [1])
        self.fund = 500000
        self.stock_quantity = 10000
        self.stock_value = 0
        self.current_stock_price = 0
        self.total = -1
        self.current_data = []
        self.current_state = []
        # self.init_op = tf.initialize_all_variables()
        self.init_placeholder()
        scores = self.batch_scoring_op()
        next_step_scores = self.batch_predict_op()
        self.add_loss_n_train_op(scores, next_step_scores)
        self.add_step_predict_op()
        self.saver = tf.train.Saver()
        self.actions = [5000, 1000, 0, -1000, -5000]
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
        Q_scores = self.Q_network_op(x)
        Q_scores = tf.reshape(Q_scores, (self.config.BATCH_SIZE, self.config.STOCK_AMOUNT + 1))
        return Q_scores

    def build_feed_dict(self, random_memories):
        feed = {}

        return feed

    def run_epoch(self, session, save=None, load=None):
        if not os.path.exists('./save'):
            os.makedirs('./save')
        if load:
            self.saver.restore(session, load)
        else:
            session.run(self.init_op)



        while True:
            if self.total == -1:
                init_data = self.sc.request_api()

                if init_data[self.config.open_price_ind]:
                    assert init_data[self.config.open_price_ind] == init_data[self.config.current_ind]
                    self.current_stock_price = init_data[self.config.current_ind]
                    self.stock_value = self.stock_quantity * self.current_stock_price
                    self.total = self.stock_value + self.fund
                    self.current_data = init_data
                    self.current_state = self.du.preprocess_state(init_data, 0, self.stock_quantity, self.stock_value, self.fund, self.total)
                else:
                    print "market closed or stock halts"
                    sys.exit(0)
                self.config.INPUT = len(self.current_state)

            # else:
            #     data, rate = self.sc.request_api()
            #     self.current_stock_price = data[self.config.current_ind]

            is_exploration = random.random()
            assert self.current_stock_price != 0
            if is_exploration <= self.config.EPSILON:
                buy_quantity = random.choice(self.actions)

            else:
                buy_quantity = sess.run(self.prediction, feed_dict={self.states_next: })


            if buy_quantity >= 0:
                if buy_quantity * self.current_stock_price > self.fund:
                    buy_quantity = self.fund / self.current_stock_price
                self.stock_quantity += buy_quantity
                self.fund -= buy_quantity * self.current_stock_price
                assert (self.current_stock_price * self.stock_quantity + self.fund) == self.total

            time.sleep(self.sc.config.time_interval)
            new_data = self.sc.request_api()
            new_price = new_data[self.config.current_ind]
            new_total = self.stock_quantity * new_price + self.fund
            new_state = self.du.preprocess_state(new_data, self.stock_quantity, self.stock_value, self.total)
            reward = 100.0 * (new_total - self.total) / self.total

            self.memories.append((self.current_state, buy_quantity, reward, new_state))
            self.current_state = new_state
            self.current_data = new_data
            self.current_stock_price = new_price
            self.total = new_total

            if len(self.memories) > 10*self.config.BATCH_SIZE:
                startind = random.randint(0,len(self.memories)-self.config.BATCH_SIZE)
                batch = self.memories[startind:startind+self.config.BATCH_SIZE]
                feed = self.build_feed_dict(batch)




if __name__ == '__main__':
    cc = Reinforcer()
    with tf.Session() as sess:
        cc.run_epoch(sess)






