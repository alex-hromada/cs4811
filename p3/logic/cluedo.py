'''cluedo.py - project skeleton for a propositional reasoner
for the game of Clue.  Unimplemented portions have the comment "TO
BE IMPLEMENTED AS AN EXERCISE".  The reasoner does not include
knowledge of how many cards each player holds.
Originally by Todd Neller
Ported to Python by Dave Musicant
Adapted to course needs by Laura Brown

Copyright (C) 2008 Dave Musicant

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Information about the GNU General Public License is available online at:
  http://www.gnu.org/licenses/
To receive a copy of the GNU General Public License, write to the Free
Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.'''

import cnf

class Cluedo:
    suspects = ['sc', 'mu', 'wh', 'gr', 'pe', 'pl']
    weapons  = ['kn', 'cs', 're', 'ro', 'pi', 'wr']
    rooms    = ['ha', 'lo', 'di', 'ki', 'ba', 'co', 'bi', 'li', 'st']
    casefile = "cf"
    hands    = suspects + [casefile]
    cards    = suspects + weapons + rooms

    """
    Return ID for player/card pair from player/card indicies
    """
    @staticmethod
    def getIdentifierFromIndicies(hand, card):
        return hand * len(Cluedo.cards) + card + 1

    """
    Return ID for player/card pair from player/card names
    """
    @staticmethod
    def getIdentifierFromNames(hand, card):
        return Cluedo.getIdentifierFromIndicies(Cluedo.hands.index(hand), Cluedo.cards.index(card))


# **************
#  Question 6 
# **************
def deal(hand, cards):
    "Construct the CNF clauses for the given cards being in the specified hand"
    "*** YOUR CODE HERE ***"
    hands = []
    
    for i in range(len(cards)):
        card = []
        id = Cluedo.getIdentifierFromNames(hand, cards[i])
        card.append(id)
        hands.append(card)


    return hands


# **************
#  Question 7 
# **************
def axiom_card_exists():
    """
    Construct the CNF clauses which represents:
        'Each card is in at least one place'
    """
    "*** YOUR CODE HERE ***"
    exist =[]
    for i in range(len(Cluedo.cards)):
        cardss = []

        for j in range(len(Cluedo.hands)):
            id = Cluedo.getIdentifierFromNames(Cluedo.hands[j], Cluedo.cards[i])
            cardss.append(id)

        exist.append(cardss)

    return exist


# **************
#  Question 7 
# **************
def axiom_card_unique():
    """
    Construct the CNF clauses which represents:
        'If a card is in one place, it can not be in another place'
    """
    "*** YOUR CODE HERE ***"
    unique = []
    for card in Cluedo.cards:

        for hand in Cluedo.hands:
            id1 = Cluedo.getIdentifierFromNames(hand, card)
            
            for otherHand in Cluedo.hands:
                
                if otherHand is not Cluedo.hands:
                    id2 = Cluedo.getIdentifierFromNames(otherHand, card)
                    temp = []
                    temp.append(-id1)
                    temp.append(-id2)
                    unique.append(temp)

    return unique


# **************
#  Question 7 
# **************
def axiom_casefile_exists():
    """
    Construct the CNF clauses which represents:
        'At least one card of each category is in the case file'
    """
    "*** YOUR CODE HERE ***"
    weps = []
    exist = []
    sus = []
    rom = []

    for i in range(len(Cluedo.weapons)):
        id1 = Cluedo.getIdentifierFromNames(Cluedo.casefile, Cluedo.weapons[i])
        weps.append(id1)

    for j in range(len(Cluedo.rooms)):
        id2 = Cluedo.getIdentifierFromNames(Cluedo.casefile, Cluedo.rooms[j])
        rom.append(id2)

    for k in range(len(Cluedo.suspects)):
        id3 = Cluedo.getIdentifierFromNames(Cluedo.casefile, Cluedo.suspects[k])
        sus.append(id3)
  
    exist.append(weps)
    exist.append(rom)
    exist.append(sus)
    return exist


# **************
#  Question 7 
# **************
def axiom_casefile_unique():
    """
    Construct the CNF clauses which represents:
        'No two cards in each category are in the case file'
    """
    "*** YOUR CODE HERE ***"

    unique = []
    for k in range(len(Cluedo.weapons)):
        id1 = Cluedo.getIdentifierFromNames(Cluedo.casefile, Cluedo.weapons[k])

        for l in range(len(Cluedo.weapons)):
            id2 = Cluedo.getIdentifierFromNames(Cluedo.casefile, Cluedo.weapons[l])

            if Cluedo.weapons[l] is not Cluedo.weapons:
                temp = []
                temp.append(-id1)
                temp.append(-id2)
                unique.append(temp)

    return unique


# **************
#  Question 8 
# **************
def suggest(suggester, card1, card2, card3, refuter, cardShown):
    "Construct the CNF clauses representing facts and/or clauses learned from a suggestion"
    "*** YOUR CODE HERE ***"
    kb = []

    if refuter is None and cardShown is None:
        
        for s in Cluedo.suspects:

            if s is not suggester:
                kb.append([-Cluedo.getIdentifierFromNames(s, card1)])
                kb.append([-Cluedo.getIdentifierFromNames(s, card2)])
                kb.append([-Cluedo.getIdentifierFromNames(s, card3)])

    elif cardShown is None:
        susin = Cluedo.suspects.index(suggester)
        refin = Cluedo.suspects.index(refuter)
        temp = susin

        for i in range(5,10):
            temp = temp + 1

            if temp >= 6:
                temp = 0

            sus = Cluedo.suspects[temp]

            if sus != suggester and sus != refuter:
                kb.append([-Cluedo.getIdentifierFromNames(sus, card1)])
                kb.append([-Cluedo.getIdentifierFromNames(sus, card2)])
                kb.append([-Cluedo.getIdentifierFromNames(sus, card3)])

            if sus == refuter:
                tempk = []
                tempk.append(Cluedo.getIdentifierFromNames(sus, card1))
                tempk.append(Cluedo.getIdentifierFromNames(sus, card2))
                tempk.append(Cluedo.getIdentifierFromNames(sus, card3))
                kb.append(tempk)
            
            if temp == refin:
                break

    else:
        kb.append([Cluedo.getIdentifierFromNames(refuter, cardShown)])        

    return kb


# **************
#  Question 9 
# **************
def accuse(accuser, card1, card2, card3, correct):
    "Construct the CNF clauses representing facts and/or clauses learned from an accusation"
    "*** YOUR CODE HERE ***"
    kb =[]
    if correct:

        kb.append([Cluedo.getIdentifierFromNames(Cluedo.casefile, card1)])
        kb.append([Cluedo.getIdentifierFromNames(Cluedo.casefile, card2)])
        kb.append([Cluedo.getIdentifierFromNames(Cluedo.casefile, card3)])
    

    if not correct:

        curkb=[]
        curkb.append(-Cluedo.getIdentifierFromNames(accuser, card1))
        curkb.append(-Cluedo.getIdentifierFromNames(accuser, card2))
        curkb.append(-Cluedo.getIdentifierFromNames(accuser, card3))
        kb.append(curkb)
        
        kb.append([-Cluedo.getIdentifierFromNames(Cluedo.casefile, card1)])
        kb.append([-Cluedo.getIdentifierFromNames(Cluedo.casefile, card2)])
        kb.append([-Cluedo.getIdentifierFromNames(Cluedo.casefile, card3)])
        # kb.append(curkb)
    return kb

