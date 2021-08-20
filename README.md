# Strategy-Learner
A Dyna Q-Learner that attempts to learn a trading strategy off reinforcement learning.

Q-learning parameters:

  - State: composition of bollinger bands, momentum, volatility
  - Actions: 0,1,2 for selling (short), holding, or buying (long)
  - Rewards: daily return

![data1](figure_1.png)

Please see ![the notebook](Strategy%20Learner.ipynb) and ![the report](report.pdf) for the results.
