import datetime as dt
import pandas as pd
# import util as ut
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math


if __name__ == '__main__':
	# dates = pd.date_range(dt.datetime(2008,1,1), dt.datetime(2009,1,1))
	# prices_all = ut.get_data(['AAPL'], dates)

	# prices = prices_all[['AAPL']]

	# # print prices

	# indicators = ['bband', 'mom', 'vol']
	# window_size = 3
	# reward_interval = 1

	# prices_SMA = pd.rolling_mean(prices, window_size)
	# prices_STD = pd.rolling_std(prices, window_size)

	# bollinger = (prices - prices_SMA)/prices_STD
	# momentum = (prices[window_size:]/prices[:-window_size].values) - 1
	# volatility = pd.rolling_std((prices[1:]/prices[:-1].values) - 1, window_size)

	# bollinger_norm = (bollinger - bollinger.mean())/bollinger.std()
	# momentum_norm = (momentum - momentum.mean())/momentum.std()
	# vol_norm = (volatility - volatility.mean())/volatility.std()

	# prices = prices.join(bollinger_norm, how='left', rsuffix='_bband')
	# prices = prices.join(momentum_norm, how='left', rsuffix='_mom')
	# prices = prices.join(vol_norm, how='left', rsuffix='_vol')

	# prices = prices.fillna(0)

	# prices.plot(x=prices.index, subplots=True)
	# plt.show()

	mu = 0
	variance = 1
	sigma = math.sqrt(variance)
	x = np.linspace(mu-3*variance,mu+3*variance, 100)
	plt.plot(x,mlab.normpdf(x, mu, sigma))
	plt.fill_between([0, 0.5],[0,0], mlab.normpdf(x, mu, sigma), facecolor='green', interpolate=True)

	plt.show()
