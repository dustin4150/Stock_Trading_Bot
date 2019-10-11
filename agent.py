from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque
from market_env import Market
import pandas as pd

class Agent:
    def __init__(self, state_size, is_eval = False, model_name = ""):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []
        self.book = []
        self.dates = []
        self.initial_investment = None
        
        self.state_size = state_size #normalized previous days
        self.action_size = 3 #hold, buy, sell
        self.memory = deque(maxlen = 1000)
        self.model_name = model_name
        self.is_eval = is_eval
        self.balance = 10**6
        self.stock = 0
        
        #discount factor
        #how much importance we give for future rewards
        #If Gamma is closer to zero, the agent will tend to consider only immediate rewards.
        self.gamma = 0.9
        
        #hyper parameter tuning
        #epsilon greedy strategy
        #will start out exploring cause epsilon = 1
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = 0.995
        
        self.model = load_model("models/" + model_name) if is_eval else self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(Dense(units = 128, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(units = 32, activation = "relu"))
        model.add(Dense(units = 8, activation = "relu"))
        model.add(Dense(units = self.action_size, activation = "softmax"))
        model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 0.001))
        return model
    
    def reset(self):
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []
        self.balance = 10**6
        self.stock = 0
    
    def act(self, state, price_data, date=None):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions[0])
        
        bought_price = None
        if action == 0: #do nothing
            print(".", end = '', flush = True)
            self.action_history.append(action)
        elif action == 1 and self.balance > 0 and self.stock == 0: #buy
            self.buy(price_data)
            self.action_history.append(action)
        elif action == 2 and self.has_inventory(): #sell
            bought_price = self.sell(price_data, date)
            self.action_history.append(action)
        else:
            self.action_history.append(0)
        
        return action, bought_price
    
    def buy(self, price_data):
        self.stock = self.balance // price_data
        self.balance -= self.stock * price_data
        self.__inventory.append(price_data)
        print("Buy: {0}".format(self.format_price(price_data)))
        
    def sell(self, price_data, date=None):
        bought_price = self.__inventory.pop(0)
        profit = (price_data - bought_price) * self.stock
        self.balance += self.stock * price_data
        self.stock = 0
        self.__total_profit += profit
        print("Sell: {0} | Profit: {1}".format(self.format_price(price_data), self.format_price(profit)))
        self.book.append(profit)
        self.dates.append(date)
        
        return bought_price
     
    def has_inventory(self):
        return len(self.__inventory) > 0
    
    def format_price(self,n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))
    
    def get_total_profit(self):
        return self.format_price(self.__total_profit)
    
    def experience_replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward 
            else: 
                next_q_values = self.model.predict(next_state)[0]
                target = reward + self.gamma * np.amax(next_q_values)
            predicted_target = self.model.predict(state)
            predicted_target[0][action] = target
            self.model.fit(state, predicted_target, epochs = 1, verbose = 0)
         
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
           
                
    
    