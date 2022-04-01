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

# fictitious play algorithm based off of class discussion
class fictitiousPlay:
    
    def __init__(self, num, ante, bet, numTry):
        
        # initialize variables
        self.N = num
        self.a = ante
        self.b = bet
        self.I = numTry
        self.bFracs = [1/2] * self.N
        self.cFracs = [1/2] * self.N
        self.bCount = [0] * self.N
        self.cCount = [0] * self.N

    def fictitiousAlg(self):

        #loop over number of tries
        for i in range(1,self.I):
            
            # reevaluate betting fraction for player X
            for j in range(1,self.N+1,1):

                checkUtil = 0.0
                betUtil = 0.0

                for k in range(1,self.N+1,1):

                    checkUtil += (1/float(i))*self.xPayoff(j,k,1)
                    betUtil += (1/float(i))*self.xPayoff(j,k,0)

                if betUtil > checkUtil:
                    self.bCount[j-1] += 1

            # set bFracs vector here
            self.bFracs = [(1/float(i))*b for b in self.bCount]
            
            # reevaluate calling fraction for player Y
            for j in range(1,self.N+1,1):
                
                foldUtil = 0.0
                callUtil = 0.0

                for k in range(1,self.N+1,1):

                    foldUtil += (1/float(i))*self.yPayoff(j,k,1)
                    callUtil += (1/float(i))*self.yPayoff(j,k,0)

                if callUtil > foldUtil:
                    self.cCount[j-1] += 1

            # set cFracs vector here
            self.cFracs = [(1/float(i))*c for c in self.cCount]
    
    # probably wrong
    def xPayoff(self,x,y,check):
        
        payoff = 0.0
        bY = self.cFracs[y-1]

        if check:
            payoff = (1/float(self.N))*((x-1)*(self.a)+0+(self.N-x)*(-self.a))
        else:
            payoff = (1/float(self.N))*(1-bY)*((x-1)*(self.a)+0+(self.N-x)*(-self.a))
            payoff += (1/float(self.N))*bY*((x-1)*(self.a+self.b)+0+(self.N-x)*(-(self.a+self.b)))

        return payoff
    
    # probably wrong
    def yPayoff(self,y,x,fold):

        payoff = 0.0
        aX = self.bFracs[x-1]

        if fold:
            payoff = -self.a
        else:
            payoff = (1/float(self.N))*aX*((y-1)*(self.a+self.b)+0+(self.N-y)*(-(self.a+self.b)))

        return payoff

    def plotFracs(self):
        
        # plot betting fraction plot
        plt.title('Player X Betting Fraction as a Function of Their Hand')
        plt.plot(self.bFracs)
        plt.savefig('bettingFraction.png')
        
        # plot calling fraction plot
        plt.clf()
        plt.title('Player Y Calling Fraction as a Function of Their Hand')
        plt.plot(self.cFracs)
        plt.savefig('callingFraction.png')

if __name__ == "__main__":

    game = fictitiousPlay(100,1,1,1000)
    game.fictitiousAlg()
    game.plotFracs()
    
    # Creates half-street game with:
    #   - Cards labeled 1 to 100
    #   - Initial pot of 0
    #   - Ante of 1 betting unit
    #   - Bet of 2 betting units
    #   - Initial drawing of cards is 1 for each player
    #   - Each player has amount 10 betting units at start
    # pr = pokerRound(100, 0, 1, 2, [1,1], [0,0], [10,10])

    # pr.drawRandCards()
    # print(pr.d)
