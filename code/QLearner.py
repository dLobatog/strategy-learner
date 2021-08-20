import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, num_states= 100, num_actions = 4, alpha = 0.2, gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.Q = np.zeros((num_states, num_actions), dtype=np.float64)
        self.TC = np.ones((num_states, num_actions, num_states), dtype=np.float64) * 0.00001
        self.R = np.zeros((num_states, num_actions))
        # self.seen_s = np.array([])
        # self.seen_a = np.array([])

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        # self.seen_s = np.append(self.seen_s, self.s)
        action = rand.randint(0, self.num_actions-1) if rand.uniform(0.0, 1.0) <= self.rar else np.argmax(self.Q[self.s])
        # self.seen_a = np.append(self.seen_a, self.a)
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        action = self.updateQ(self.s, self.a, s_prime, r)
        if self.dyna != 0:
        	self.updateT_R(self.s, self.a, s_prime, r)
        	for i in range(self.dyna):
        		s = rand.randint(0, self.num_states-1)
        		a = rand.randint(0, self.num_actions-1)
        		self.updateQ(s, a, np.argmax(self.TC[s][a]), self.R[np.argmax(self.TC[s][a])][a])

        self.s = s_prime
        self.a = action
        self.rar *= self.radr

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return self.a

    def updateT_R(self, s, a, s_prime, r):
        self.TC[s][a][s_prime] += 1
        self.R[s_prime][a] += self.alpha * (r - self.R[s_prime][a])

    def updateQ(self, s, a, s_prime, r):
    	self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[s_prime]) - self.Q[s][a])
        action = rand.randint(0, self.num_actions-1) if rand.uniform(0.0, 1.0) <= self.rar else np.argmax(self.Q[s_prime])

        return action


if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
