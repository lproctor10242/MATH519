''' 
MATH 519: Poker Project
Authors: Nathaniel Sheets & Lauren Proctor

PART 1:
------------------------------------------------------------------------------
Game 1: 
- Each player antes an amount a forming a pot p=2a.
- Each player receives an independently chosen, uniformly distributed random
  integer between 1 and N.
- Player 1 checks or bets b.
- If player 1 checks, there is a showdown for p.
- If player 1 bets, player 2 may fold or call.
- If player 2 calls there is a showdown for p+2b.

Method (#1):
- In fictitious player, each player chooses the best pure strategy response to
the average play of their opponent over previous iterations of the game.

Deliverables:
- Generate two graphs:
    1. the betting fraction for each card i as a function of i
    2. the calling fraction for each card i as a function of i
------------------------------------------------------------------------------
'''

import matplotlib.pyplot as plt
import random
