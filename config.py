# coding = utf-8
"""author = jingyuan zhang"""


class Config:
    def __init__(self):
        self.BATCH_SIZE = 32
        self.UNCHANGED_TOLERENCE = 10
        self.EPSILON = 0.1
        self.MAXSTEP = 100
        self.MAXTIMESTEP = 200
        self.START_TIMESTEP = 50
        self.INPUT = 13
        self.M1 = 20
        self.M2 = 5
        self.lr = 1e-6
        self.gamma = 0.015
        self.STOCK_AMOUNT = 16
        self.TRANSACTION_AMOUNT = 0.1
        self.BENCHMARK_RETURN_INDEX = 14
        self.DROPOUT = 0.5
        self.INTERVAL = 5


class ASingleStockConfig:
    def __init__(self):
        self.BATCH_SIZE = 32
        self.DROPOUT = 0.5
        self.code = '601766'
        self.type = 'sh'
        self.time_interval = 1
        self.outfile = '601766_60.txt'
