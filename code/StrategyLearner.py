import datetime as dt
import pandas as pd
import util as ut
import numpy as np
import random
import QLearner

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.q = QLearner.QLearner(num_states=1000, num_actions=3)
        self.train = True


    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices


        # cash = sv
        data = self.compute_indicators(prices)

        states = np.apply_along_axis(lambda x: x[0] * 100 + x[1] * 10 + x[2], 1, self.discretize(data.iloc[:, :-1], 0).values)
        # print states
        rets = data.iloc[:,-1].values
        print rets

        assert len(states) == len(rets)

        curr_a = self.q.querysetstate(states[0])

        count = 0
        while count < 1500:
            holding = 0
            for i in range(1, len(states)):
                if curr_a == 0 and (holding == 0 or holding == 200):
                    holding -= 200
                elif curr_a == 1:
                    holding = holding
                elif curr_a == 2 and (holding == 0 or holding == -200):
                    holding += 200
                curr_r = holding * rets[i]
                curr_a = self.q.query(states[i], curr_r)
            count += 1

    def discretize(self, indicators, holding):
        start_point = 2.4
        interval = 0.5

        indicators = (indicators + start_point)/interval
        indicators[np.isnan(indicators)] = 4

        indicators = indicators.astype(int)
        # print indicators
        indicators[indicators > 9] = 9
        indicators[indicators < 0] = 0
        return indicators

    def compute_indicators(self, prices):
        indicators = ['bband', 'mom', 'vol']
        holdings = [200, 0, -200]

        window_size = 3
        reward_interval = 1

        prices_SMA = pd.rolling_mean(prices, window_size)
        prices_STD = pd.rolling_std(prices, window_size)

        bollinger = (prices - prices_SMA)/prices_STD
        momentum = (prices[window_size:]/prices[:-window_size].values) - 1
        volatility = pd.rolling_std((prices[1:]/prices[:-1].values) - 1, window_size)

        train_set = pd.DataFrame(data=None, index=prices.index)


        if self.train:
            self.b_mu = bollinger.mean()
            self.b_std = bollinger.std()

            self.mom_mu = momentum.mean()
            self.mom_std = momentum.std()

            self.vol_mu = volatility.mean()
            self.vol_std = volatility.std()

        bollinger_norm = (bollinger - self.b_mu)/self.b_std
        momentum_norm = (momentum - self.mom_mu)/self.mom_std
        vol_norm = (volatility - self.vol_mu)/self.vol_std

        ret = (prices[reward_interval:]/prices[:-reward_interval].values) - 1

        bollinger_norm.columns = [s + "_bband" for s in bollinger_norm.columns]
        train_set = train_set.join(bollinger_norm, how='left')
        momentum_norm.columns = [s + "_mom" for s in momentum_norm.columns]
        train_set = train_set.join(momentum_norm, how='left')
        vol_norm.columns = [s + "_vol" for s in vol_norm.columns]
        train_set = train_set.join(vol_norm, how='left')
        ret.columns = [s + "_ret" for s in vol_norm.columns]
        train_set = train_set.join(ret, how='left')
        train_set[ret.columns] = train_set[ret.columns].fillna(value=0)

        return train_set


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # self.train = False

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        indicators = self.compute_indicators(trades)
        states = np.apply_along_axis(lambda x: x[0] * 100 + x[1] * 10 + x[2], 1,
                                     self.discretize(indicators.iloc[:, :-1], 0).values)
        # rets = indicators.iloc[:, -1].values
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing

        init_a = self.q.querysetstate(states[0])
        holding = (init_a - 1) * 200
        trades.values[0,:] = holding

        assert len(states) == len(trades)

        for i in range(1, len(states)):
            action = self.q.querysetstate(states[i])

            taken = 0
            if action == 0 and (holding == 0 or holding == 200):
                holding -= 200
                taken = -200
            elif action == 2 and (holding == 0 or holding == -200):
                holding += 200
                taken = 200

            trades.values[i, :] = taken

        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
