''' 
MATH 519: Poker Project
Authors: Nathaniel Sheets & Lauren Proctor

PART 3:
------------------------------------------------------------------------------
Game 1: 
- Each player antes an amount a forming a pot p=2a.
- Each player receives an independently chosen, uniformly distributed random
  integer between 1 and N.
- Player 1 checks or bets b.
- If player 1 checks, there is a showdown for p.
- If player 1 bets, player 2 may fold or call.
- If player 2 calls there is a showdown for p+2b.

Method (#2):
- In q-learning, the goal is to learn iteratively the optimal Q-value function 
  using the Bellman Optimality Equation. To do so, we store all the Q-values in 
  a table that we will update at each time step using the Q-Learning iteration:

Deliverables:
- Generate two graphs:
    1. the betting fraction for each card i as a function of i
    2. the calling fraction for each card i as a function of i
------------------------------------------------------------------------------
'''
import matplotlib as plt
import numpy as np
import random

class Qlearning:
	def __init__(self, num, ante, bet, episodes): ##############################
		''' initialize variables (# cards, ante, bet, number of iterations) '''

		# halfstreet stuff
		self.N = num
		self.a = ante
		self.b = bet
		self.I = episodes

		# exploration probabilities
		self.exp = 1
		self.minexp = 0.01
		self.decay = 0.001

		# learning rate & discount factor
		self.lr = 0.1
		self.df = 0.6
	###########################################################################

	def p1_payoff(self, p1c, p2c, bet, fold): #################################
		''' calculates the reward of player1 based on cards & strats played '''
		# player 1 tries to bet
		if (bet == 1): 
			if (fold == 2): # player 2 folds
				return self.a
			elif (p1c > p2c):
				return (self.a + self.b)
			elif (p1c < p2c):
				return (-self.a - self.b)
			else:
				return (0.0)
		
		# player 1 tries to check
		else:
			if (p1c > p2c):
				return self.a
			elif (p1c < p2c):
				return (-self.a)
			else:
				return (0.0)
	###########################################################################

	def p2_payoff(self, p1c, p2c, fold): ######################################
		''' calculates the reward of player2 based of cards & strats played '''
		# player 2 folds
		if (fold == 2):
			return (-self.a)
		else:
			if (p1c > p2c):
				return (-self.a - self.b)
			elif (p1c < p2c):
				return (self.a + self.b)
			else:
				return (0.0)
	###########################################################################

	def train(self): #########################################################################
		''' trains the agent to play the game via qLearning algorithm'''

		# create table to hold q values
		q_table = np.zeros([self.N,4])

		# create struct to hold rewards
		totRewards = []
		
		# take an iterative approach to the q-learning (grid search impractical)
		for e in range(self.I):
			# sum of the total rewards from agent:
			eReward1 = 0
			eReward2 = 0

			for i in range(self.N):
				for j in range(self.N):

					# for player 1 we either explore or go w/ prev knowledge
					if (np.random.uniform(0,1) < self.exp):
						strat1 = random.randrange(2)
					else:
						strat1 = np.argmax(q_table[i,:2])

					# for player 2 we either explore or go w/ prev knowledge
					if (np.random.uniform(0,1) < self.exp):
						strat2 = random.randrange(2,4)
					else:
						strat2 = np.argmax(q_table[j,2:])

					rwd1 = self.p1_payoff(i, j, strat1, strat2)
					rwd2 = self.p2_payoff(i, j, strat2)

					eReward1 += rwd1
					eReward2 += rwd2

					q_table[i, strat1] = (1-self.lr) * q_table[i, strat1] + self.lr * rwd1
					q_table[j, strat2] = (1-self.lr) * q_table[j, strat2] + self.lr * rwd2				

			self.exp = max(self.minexp, np.exp(-self.decay))

			totRewards.append([eReward1,eReward2])

		return q_table
	######################################################################################	

if __name__ == "__main__":

	# create games and run Q-learning with multiple iteration numbers:
	g1 = Qlearning(100, 1, 1, 1000)
	gMatrix = g1.train()
	print(gMatrix)
	