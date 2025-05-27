import random

class RideTheBus:
    
    def __init__(self):
        self.deck = ['A', 'A', 'A', 'A', '2', '2', '2', '2',
        '3', '3', '3', '3', '4', '4', '4', '4',
        '5', '5', '5', '5', '6', '6', '6', '6',
        '7', '7', '7', '7', '8', '8', '8', '8',
        '9', '9', '9', '9', '10', '10', '10', '10',
        'J', 'J', 'J', 'J', 'Q', 'Q', 'Q', 'Q',
        'K', 'K', 'K', 'K']

        self.cardvals = {'A':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 
                         '10':10, 'J':11, 'Q':12, 'K':13}

        self.board = [[], [], [], [],
                      [], [], [], [],
                      [], [], [], [],
                      [], [], [], []]

        self.score = 0
        
        self.shuffle()
        self.fill()
        

    def __repr__(self):
        rtnString = ""
        counter = 0
        for i in self.board:
            rtnString += i[0] + ",  "
            if counter == 3:
                rtnString += "\n"
                counter = 0
            else:
                counter += 1
            
        return rtnString

    def __str__(self):
        rtnString = ""
        counter = 0
        for i in self.board:
            rtnString += i[0] + ",  "
            if counter == 3:
                rtnString += "\n"
                counter = 0
            else:
                counter += 1
            
        return rtnString

    def shuffle(self):
        newDeck = []
        while len(self.deck) > 0:
            w = random.randint(0, len(self.deck)-1)
            x = self.deck.pop(w)
            newDeck.append(x)
        self.deck = newDeck

    def fill(self):
        for i in self.board:
            card = self.deck.pop()
            i.append(card)

    def recommend(self):
        ev = 0
        total = 0
        for i in self.deck:
            total += self.getval(i)
        ev = total / len(self.deck)
        return ev
    
    def getval(self, card):
        return self.cardvals[card]
    
    def replace(self, card, newCard):
        idx = self.board.index([card])
        self.board[idx] = [newCard]

    def guess(self):
        card = str(input("Please pick a card from the board "))
        if [card] not in self.board:
            print("selection not in board")
            return
        highlow = str(input("Please pick if the card is higher or lower ")).lower()
        
        if highlow not in ("h", "higher", "l", "lower"):
            print("please guess 'h' or 'higher' to guess higher, or 'l' or 'lower' for lower")
            return
        
        newCard = self.deck.pop()

        if highlow in ("h", "higher"):
            if self.getval(newCard) > self.getval(card):
                print("Correct! the card was " + newCard)
                self.score += 1
            else:
                print("Wrong! the card was " + newCard)
                self.score = 0

        elif highlow in ("l", "lower"):
            if self.getval(newCard) < self.getval(card):
                print("Correct! the card was " + newCard)
                self.score += 1
            else:
                print("Wrong! the card was " + newCard)
                self.score = 0
        
        self.replace(card, newCard)

    def input_guess(self, card, highlow):
        if [card] not in self.board:
            print("selection not in board")
            return
        
        newCard = self.deck.pop()
        correct = 0

        if self.getval(newCard) == self.getval(card):
            self.score = 0

        elif highlow in ("h", "higher"):
            if self.getval(newCard) > self.getval(card):
                self.score += 1
                correct = 1
            else:
                self.score = 0

        elif highlow in ("l", "lower"):
            if self.getval(newCard) < self.getval(card):
                self.score += 1
                correct = 1
            else:
                self.score = 0
        
        self.replace(card, newCard)
        return correct

    def play(self):
        while self.score < 5:
            print("\nScore: " + str(self.score))
            print(self)
            print("Expected value: " + str(self.recommend()))
            self.guess()
        print("You win!")
            
            

if __name__ == "__main__":
    x = RideTheBus()
    x.play()
    


            
