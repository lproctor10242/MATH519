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