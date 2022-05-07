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
- ...

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