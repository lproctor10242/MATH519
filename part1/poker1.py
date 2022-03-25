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

##########################################################################
# Meaning of Variables:
# ---------------------
# numCards:  defines a deck from 1 to N
# ante:      The contribution at beginning of round pot = 2a
# bet:       the betting amount of the players
# cardDrawn: the cards the players drew that turn as an array [p1c, p2c]
# avgStrat:  average strat as a vector (0 for fold/check & 1 for bet/call)
# amounts:   the amount each player has

class pokerRound:
    """sets up the poker round... can be modded throughout round"""
    N = 1
    p = 0
    a = 1
    b = 1
    d = [1,1]
    s = [0,0]
    am = [1,1]    

    def __init__(self, numCards, pot, ante, bet, drawCards, avgStrat, amounts):
        self.N = numCards
        self.p = pot            # pot
        self.a = ante           # ante
        self.b = bet            # bet
        self.d = drawCards      # player's cards
        self.s = avgStrat       # average strategy
        self.am = amounts       # amount for each player   
    
    def formPot(self):
        for i in range(2):
            self.am[i] -= self.a
            self.p += self.a # decrements ante and increase pot
        return self.p           

    def drawCards(self, cardDrawn):
        self.d = cardDrawn # draws cards that we give
        
    def drawRandCards(self):
        c1 = random.randrange(1, self.N)
        c2 = random.randrange(1, self.N)
        self.d = [c1, c2] # draws randomized cards
        
    def p1Checks(self): # return the winnings of the round
        pot = self.p

        if (self.d[0] > self.d[1]):
            self.am[0] += self.p # if p1 wins
            self.p = 0 # reset pot
            return [pot, 0]
            
        if (self.d[0] < self.d[1]):
            self.am[1] += self.p # if p2 wins
            self.p = 0 # reset pot
            return [0, pot]

    def p1Bets(self):
        self.am[0] -= self.b
        self.p += self.b # decrement amount increase pot

    def p2Folds(self): # return the winnings of the round
        self.am[0] += self.p
        pot = self.p
        self.p = 0 # add winning and reset pot
        return [pot, 0]
            
    def p2Calls(self): # return the winnings of the round
        self.am[1] -= self.b
        self.p += self.b # decrement amount increase pot
        pot = self.p

        if (self.d[0] > self.d[1]):
            self.am[0] += self.p # if p1 wins
            self.p = 0 # reset pot
            return [pot, 0]
            
        if (self.d[0] < self.d[1]):
            self.am[1] += self.p # if p2 wins
            self.p = 0 # reset pot
            return [0, pot]

# Creates half-street game with:
#   - Cards labeled 1 to 100
#   - Initial pot of 0
#   - Ante of 1 betting unit
#   - Bet of 2 betting units
#   - Initial drawing of cards is 1 for each player
#   - Each player has amount 10 betting units at start
pr = pokerRound(100, 0, 1, 2, [1,1], [0,0], [10,10])

pr.drawRandCards()
print(pr.d)