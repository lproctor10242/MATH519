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
import matplotlib.pyplot as plt
import numpy as np
import random

class Qlearning:
	def __init__(self, num, ante, bet, episodes): #############################
		''' initialize variables (# cards, ante, bet, number of iterations) '''

		# halfstreet stuff
		self.N = num
		self.a = ante
		self.b = bet
		self.I = episodes

		# exploration probabilities
		self.exp = 1
		self.init = 0.3
		self.end = 0.001

		# learning rate & discount factor
		self.lr = 1
		self.min = 0.001
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
		else: # else compare cards
			if (p1c > p2c):
				return (-self.a - self.b)
			elif (p1c < p2c): # lose or gain bet & ante
				return (self.a + self.b)
			else: # else we tie and 0 gain
				return (0.0)
	###########################################################################

	def train(self): ###############################################################
		''' trains the agent to play the game via qLearning algorithm'''

		# create table to hold q values
		q_table = np.zeros([self.N,4])
		#strat_counter = np.zeros([self.N, 4])
		
		# take an iterative approach to the q-learning (grid search impractical)
		for e in range(self.I):

			# iterate the possible values of p1c and p2c
			for i in range(self.N):
				for j in range(self.N):
					strat1 = 0
					strat2 = 3

					# we first see if we go for both random
					if (np.random.uniform(0,1) < self.exp):
						strat1 = random.randrange(2)
						strat2 = random.randrange(2,4)
					# if we not exploring, just go with prev for both
					else:
						strat1 = np.argmax(q_table[i,:2])
						if (strat1 == 1):
							strat2 = np.argmax(q_table[j,2:])
							strat2 += 2

					# for player 1 we either explore or go w/ prev knowledge
					'''elif (np.random.uniform(0,1) < self.exp):
						strat1 = random.randrange(2)
						if (strat1 == 1):
							strat2 = np.argmax(q_table[j,2:])
							strat2 += 2

					# if we not exploring for 1, we either explore or go w/ prev
					elif (np.random.uniform(0,1) < self.exp):
					#else:
						strat1 = np.argmax(q_table[i,:2])
						if (strat1 == 1):
							strat2 = random.randrange(2,4)

					# if we not exploring, just go with prev for both
					else:
						strat1 = np.argmax(q_table[i,:2])
						if (strat1 == 1):
							strat2 = np.argmax(q_table[j,2:])
							strat2 += 2'''

					# calculate rewards
					rwd1 = self.p1_payoff(i, j, strat1, strat2)
					if (strat1 == 1): # calculate r2
						rwd2 = self.p2_payoff(i, j, strat2)

					# update q_values using formula
					q_table[i,strat1]=(1-self.lr)*q_table[i,strat1]+self.lr*rwd1
					if (strat1 == 1):
						q_table[j,strat2]=(1-self.lr)*q_table[j,strat2]+self.lr*rwd2						

			# calculate new exploration probabilities
			#r = max(0, (self.N-e)/self.N)
			#self.exp = (self.init-self.end)*r+self.end
			
			# calculate the new learning rate
			self.lr = max(self.min,1/((e+2)**(0.69)))
			self.exp = self.lr
			#self.lr = self.exp
			#print(self.lr)

		# make our strategies based on the q-value
		bFrac = [0] * self.N
		cFrac = [0] * self.N
		for i in range(len(q_table)):
			# make comparison for betting fraction
			if (q_table[i,1] > q_table[i,0]):
				bFrac[i] = 1
			else:
				bFrac[i] = 0

			# make comparison for calling fraction
			if (q_table[i,3] > q_table[i,2]):
				cFrac[i] = 1
			else:
				cFrac[i] = 0
				
		print(q_table)
		return bFrac, cFrac
	################################################################################

if __name__ == "__main__": ###########################################

	# create games and run Q-learning with multiple iteration numbers:
	g1 = Qlearning(100, 1, 1, 20000)
	bFrac, cFrac = g1.train()
	#print(bFrac)
	#print(cFrac)

    # plot betting fraction plot
	plt.figure(figsize=(12,8))
	plt.title('Player 1 Betting Fraction as a Function of Their Hand')
	plt.xlabel('Card Number')
	plt.ylabel('Betting Fraction')
	plt.plot(bFrac,label='100 iter.')
	plt.legend()
	plt.savefig('bettingValues.png')
	
    # plot calling fraction plot
	plt.clf()
	plt.figure(figsize=(12,8))
	plt.title('Player 2 Calling Fraction as a Function of Their Hand')
	plt.xlabel('Card Number')
	plt.ylabel('Calling Fraction')
	plt.plot(cFrac, label='100 iter.')
	plt.legend()
	plt.savefig('callingValues.png')
######################################################################