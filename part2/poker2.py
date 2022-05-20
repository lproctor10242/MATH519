''' 
MATH 519: Poker Project
Authors: Nathaniel Sheets & Lauren Proctor

PART 2:
------------------------------------------------------------------------------
Game 2: 
- Each player antes an amount a forming a pot p=2a.
- Each player receives an independently chosen, uniformly distributed random
  integer between 1 and N.
- Player 1 checks or bets b.
- If player 1 checks, player 2 may check or bet. If player 2 checks we go to
  showdown. If player 2 bets, player 1 may fold or call.
- If player 1 bets, player 2 may fold or call.

Method (#1):
- In fictitious player, each player chooses the best pure strategy response to
the average play of their opponent over previous iterations of the game.

Deliverables:
- Generate two graphs:
    1. the betting fraction for each card i as a function of i
    2. the calling fraction for each card i as a function of i
------------------------------------------------------------------------------
'''

# imports
import matplotlib.pyplot as plt

# fictitious play algorithm based off of class discussion
class fictitiousPlay:

    def __init__(self, num, ante, bet, numTry):

        # initialize variables
        self.N = num
        self.a = ante
        self.b = bet
        self.I = numTry

    def xPayoff(self,x,y,xbet,ycall,ycheck,xfold):
        payoff = 0
        if (xbet == 1):
            if (ycall == 0): 
                payoff = self.a
            elif (x > y):    
                payoff = self.a + self.b
            elif (x < y):   
                payoff = -self.a - self.b
            else:              
                payoff = 0.0
        elif (ycheck == 1):
            if (x > y):
                payoff = self.a
            elif (x < y):
                payoff = -self.a
            else:
                payoff = 0.0
        elif (xfold == 1):
            payoff = -self.a
        elif (x > y): 
            payoff = self.a + self.b
        elif (x < y):
            payoff = -self.a - self.b
        else:
            payoff = 0.0
        return payoff

    def fictitiousAlg(self):

        # count and fraction vectors
        xbFrac = [0.25] * self.N
        xcFrac = [0.25] * self.N
        ybFrac = [0.25] * self.N
        ycFrac = [0.25] * self.N
        xbCount = [0] * self.N
        xcCount = [0] * self.N
        xfCount = [0] * self.N
        ycCount = [0] * self.N
        ybCount = [0] * self.N

        # loop over number of tries
        for i in range(1,self.I+1):

            # reevaluate betting fraction for player X
            for j in range(self.N):

                checkFoldUtil = 0
                checkCallUtil = 0
                betUtil = 0

                for k in range(self.N):

                    aY = ycFrac[k]
                    bY = ybFrac[k]
                    checkFoldUtil += ((1-bY)*self.xPayoff(j,k,0,0,1,0) + bY*self.xPayoff(j,k,0,0,0,1))
                    checkCallUtil += ((1-bY)*self.xPayoff(j,k,0,0,1,0) + bY*self.xPayoff(j,k,0,0,0,0))
                    betUtil += ((aY*self.xPayoff(j,k,1,1,0,0)) + ((1-aY)*self.xPayoff(j,k,1,0,0,0)))

                if betUtil > max(checkFoldUtil,checkCallUtil):
                    xbCount[j] += 1
                elif checkCallUtil > checkFoldUtil:
                    xcCount[j] +=1
                else:
                    xfCount[j] += 1

                xbFrac[j] = (1/i)*xbCount[j]
                xcFrac[j] = xcCount[j]/max((xcCount[j] + xfCount[j]),1)

            # reevaluate calling fraction for player Y
            for j in range(self.N):

                foldUtil = 0
                callUtil = 0
                checkUtil = 0
                betUtil = 0

                for k in range(self.N):

                    aX = xbFrac[k]
                    cX = xcFrac[k]
                    foldUtil += (-1*aX*self.xPayoff(k,j,1,0,0,0))
                    callUtil += (-1*aX*self.xPayoff(k,j,1,1,0,0))
                    checkUtil += (-1*(1-aX)*self.xPayoff(k,j,0,0,1,0))
                    betUtil += (1-aX)*(-1*(1-cX)*self.xPayoff(k,j,0,0,0,1) + -1*cX*self.xPayoff(k,j,0,0,0,0))

                if callUtil > foldUtil:
                    ycCount[j] += 1
                if betUtil > checkUtil:
                    ybCount[j] += 1

                ycFrac[j] = (1/i)*ycCount[j]
                ybFrac[j] = (1/i)*ybCount[j]

        # return fraction vectors
        return xbFrac, xcFrac, ycFrac, ybFrac

if __name__ == "__main__":
    
    # create game and run fictitious play for different iteration nums
    game = fictitiousPlay(100,1,1,100000)
    xb, xc, yc, yb = game.fictitiousAlg()

    # plot x betting fraction plot
    plt.figure(figsize=(12,8))
    plt.title('Player X Betting Fraction')
    plt.xlabel('Card Number')
    plt.ylabel('Betting Fraction')
    plt.plot(xb)
    plt.scatter(xb)
    plt.savefig('xbettingFraction.png')

    # plot y betting fraction plot
    plt.figure(figsize=(12,8))
    plt.title('Player Y Betting Fraction')
    plt.xlabel('Card Number')
    plt.ylabel('Betting Fraction')
    plt.plot(yb)
    plt.scatter(yb)
    plt.savefig('ybettingFraction.png')

    # plot y calling fraction plot
    plt.figure(figsize=(12,8))
    plt.title('Player Y Calling Fraction')
    plt.xlabel('Card Number')
    plt.ylabel('Calling Fraction')
    plt.plot(yc)
    plt.scatter(yc)
    plt.savefig('ycallingFraction.png')

    #calculate check call and check fold fractions
    xchc = [(1-xb[i])*(xc[i]) for i in range(100)]
    xchf = [(1-xb[i])*(1-xc[i]) for i in range(100)]
    xbchc = [xb[i] + xchc[i] for i in range(100)]
    xbchf = [xb[i] + xchf[i] for i in range(100)]
    xchfchc = [xchf[i] + xchc[i] for i in range(100)]
    
    # plot calling fraction plot
    plt.clf()
    plt.figure(figsize=(12,8))
    plt.title('Player X Betting, Check Fold, and Check Call Fractions and Their Pairwise Sums')
    plt.xlabel('Card Number')
    plt.ylabel('Fractions')
    plt.plot(xb, label='Bet')
    plt.scatter(xb)
    plt.plot(xchc, label='Check Call')
    plt.scatter(xchc)
    plt.plot(xchf, label='Check Fold')
    plt.scatter(xchf)
    plt.plot(xbchc, label='Bet + Check Call')
    plt.scatter(xbchc)
    plt.plot(xbchf, label='Bet + Check Fold')
    plt.scatter(xbchf)
    plt.plot(xchfchc, label='Check Fold + Check Call')
    plt.scatter(xchfchc)
    plt.legend()
    plt.savefig('xcombinedFraction.png')
