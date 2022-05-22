''' 
MATH 519: Poker Project
Authors: Nathaniel Sheets & Lauren Proctor

PART 4:
------------------------------------------------------------------------------
Game 2: 
- Each player antes an amount a forming a pot p=2a.
- Each player receives an independently chosen, uniformly distributed random
  integer between 1 and N.
- Player 1 checks or bets b.
- If player 1 checks, player 2 may check or bet. If player 2 checks we go to
  showdown. If player 2 bets, player 1 may fold or call.
- If player 1 bets, player 2 may fold or call.

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
		self.init = 0.6
		self.end = 0.001

		# learning rate & discount factor
		self.lr = 1
		self.min = 0.001
	###########################################################################

	def p1_payoff(self, p1c, p2c, p1bet, p1fold, p2strat): ############
		''' calculates the reward of player1 based on cards & strats played '''

		# player 1 tries to bet
		if (p1bet == 1):

			# player 2 folds
			if (p2strat == 2):
				return self.a
			# player 2 does not fold
			elif (p1c > p2c):
				return (self.a + self.b)
			elif (p1c < p2c):
				return (-self.a - self.b)
			else: # we tied
				return (0.0)
		
		# else player 1 didn't try to bet and player 2 tries to check
		elif (p2strat == 4):
			
			# we have a showdown for the pot
			if (p1c > p2c):
				return self.a
			elif (p1c < p2c):
				return (-self.a)
			else:
				return (0.0)
		
		# else player 1 tries to fold assuming player 2 tried to bet
		elif (p1fold == 6):
			return (-self.a)
		
		# else we have showdown
		else: # (here p1fold should be 5 and then 7)
			if (p1c > p2c):
				return (self.a + self.b)
			elif (p1c < p2c):
				return (-self.a - self.b)
			else:
				return (0.0)
	###########################################################################

	def p2_payoff(self, p1c, p2c, p1bet, p1fold, p2strat): ############
		''' calculates the reward of player1 based on cards & strats played '''

		# player 1 tries to bet
		if (p1bet == 1):

			# player 2 folds
			if (p2strat == 2):
				return (-self.a)
			# player 2 does not fold
			elif (p1c > p2c):
				return (-self.a - self.b)
			elif (p1c < p2c):
				return (self.a + self.b)
			else: # we tied
				return (0.0)
		
		# else player 1 didn't try to bet and player 2 tries to check
		elif (p2strat == 4):
			
			# we have a showdown for the pot
			if (p1c > p2c):
				return (-self.a)
			elif (p1c < p2c):
				return self.a
			else:
				return (0.0)
		
		# else player 1 tries to fold assuming player 2 tried to bet
		elif (p1fold == 6):
			return self.a
		
		# else we have showdown as player 1 doubles down
		else: # (here p1fold should be 5 and then 7)
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
		q_table = np.zeros([self.N,8])
		#strat_counter = np.zeros([self.N, 4])
		
		# take an iterative approach to the q-learning (grid search impractical)
		for e in range(self.I):

			# iterate the possible values of p1c and p2c
			for i in range(self.N):
				for j in range(self.N):
					strat1 = 0
					strat2 = 3
					strat3 = 6
					isExplore = 0

					# we first see if we go for both random
					if (np.random.uniform(0,1) < self.exp):
						isExplore = 1
						strat1 = random.randrange(2)
						if (strat1 == 1):
							strat2 = random.randrange(2,4)
						else:
							strat2 = random.randrange(4,6)
					# if we not exploring, just go with prev for both
					else:
						strat1 = np.argmax(q_table[i,:2])
						if (strat1 == 1):
							strat2 = np.argmax(q_table[j,2:4])
							strat2 += 2
						else:
							strat2 = np.argmax(q_table[j,4:6])
							strat2 += 4

					# now we could have the case where p1 must call or fold
					if (strat1 == 0 and strat2 == 5):
						if (isExplore == 1):
						#if (np.random.uniform(0,1) < self.exp1):
							strat3 = random.randrange(6,8)
						else: # we either go exploring or go w/ prev knowledge
							strat3 = np.argmax(q_table[i,6:])
							strat3 += 6

					# calculate rewards
					rwd1 = self.p1_payoff(i, j, strat1, strat3, strat2)
					rwd2 = self.p2_payoff(i, j, strat1, strat3, strat2)

					if (strat1 == 0 and strat2 == 5): # if p1 has a choice
						rwd3 = self.p1_payoff(i, j, strat1, strat3, strat2)

					# update q_values using formula
					q_table[i,strat1]=(1-self.lr)*q_table[i,strat1]+self.lr*rwd1
					q_table[j,strat2]=(1-self.lr)*q_table[j,strat2]+self.lr*rwd2

					if (strat1 == 0 and strat2 == 5): # if p1 has a 2nd choice
						q_table[i,strat3]=(1-self.lr)*q_table[i,strat3]+self.lr*rwd3				

			# calculate new exploration probabilities
			#r = max(0, (self.N-e)/self.N)
			#self.exp1 = (self.init-self.end)*r+self.end
			
			# calculate the new learning rate
			self.lr = max(self.min,1/((e+2)**(0.69)))
			self.exp = self.lr

		# make our strategies based on the q-value
		p1bFrac = [0] * self.N
		p1cFrac = [0] * self.N
		p2bFrac = [0] * self.N
		p2cFrac = [0] * self.N
		for i in range(len(q_table)):
			# make comparison for player 1 betting fraction
			if (q_table[i,1] > q_table[i,0]):
				p1bFrac[i] = 1
			else:
				p1bFrac[i] = 0

			# make comparison for player 2 calling fraction
			if (q_table[i,3] > q_table[i,2]):
				p2cFrac[i] = 1
			else:
				p2cFrac[i] = 0

			# make comparison for player 2 betting fraction
			if (q_table[i,5] > q_table[i,4]):
				p2bFrac[i] = 1
			else:
				p2bFrac[i] = 0

			# make comparison for player 1 calling fraction
			if (q_table[i,7] > q_table[i,6]):
				p1cFrac[i] = 1
			else:
				p1cFrac[i] = 0
				
		print(q_table)
		print("p1bFrac")
		print(p1bFrac)
		print("p1cFrac")
		print(p1cFrac)
		print("p2bFrac")
		print(p2bFrac)
		print("p2cFrac")
		print(p2cFrac)
		#print(p1cFrac)
		return p1bFrac, p1cFrac, p2bFrac, p2cFrac
	######################################################################################	

if __name__ == "__main__": ###########################################

	# create games and run Q-learning with multiple iteration numbers:
	g1 = Qlearning(100, 1, 1, 20000)
	b1Frac, c1Frac, b2Frac, c2Frac = g1.train()

	# plot x betting fraction plot
	plt.figure(figsize=(12,8))
	plt.title('Player 1 Betting Fraction')
	plt.xlabel('Card Number')
	plt.ylabel('Betting Fraction')
	plt.plot(b1Frac)
	plt.savefig('xbettingFraction.png')

	# plot y betting fraction plot
	plt.figure(figsize=(12,8))
	plt.title('Player 2 Betting Fraction')
	plt.xlabel('Card Number')
	plt.ylabel('Betting Fraction')
	plt.plot(b2Frac)
	plt.savefig('ybettingFraction.png')

	# plot y calling fraction plot
	plt.figure(figsize=(12,8))
	plt.title('Player 2 Calling Fraction')
	plt.xlabel('Card Number')
	plt.ylabel('Calling Fraction')
	plt.plot(c2Frac)
	plt.savefig('ycallingFraction.png')

	# plot calling fraction plot
	plt.figure(figsize=(12,8))
	plt.title('Player 1 Calling Fraction')
	plt.xlabel('Card Number')
	plt.ylabel('Calling Fraction')
	plt.plot(c1Frac)
	plt.savefig('xcombinedFraction.png')