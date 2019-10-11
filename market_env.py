import numpy as np
import pandas as pd
from agent import *

class Market:
    def __init__(self, windows_size, stock_name):
        self.stock_name = stock_name
        self.data = self.get_stock_data(stock_name)
        self.date_data = self.get_time(stock_name)
        self.states = self.get_all_window_prices_diff(self.data, windows_size)
        self.index = -1
        self.last_data_index = len(self.data) - 1
         
    def get_stock_data(self, stock_name):
        return list(pd.read_csv("data/" + stock_name + ".csv").Close)

    def get_time(self, stock_name):
        return list(pd.read_csv("data/" + stock_name + ".csv").Date)
      
    def get_all_window_prices_diff(self, data, window_size):
        processed_data = []
        for t in range(len(data)):
            state = self.get_window(data, t, window_size + 1)
            processed_data.append(state)    
        return processed_data 
    
    def get_window(self, data, t, n):
        d = t - n + 1
        block = data[d: t+1] if d >= 0 else -d * [data[0]] + data[0: t+1]
        res = []
        for i in range(n-1):
            res.append(block[i+1] - block[i])
        return np.array([res])
   
    def reset(self):
        self.index = -1
        return self.states[0], self.data[0], self.date_data[0]
    
    def get_next_state_reward(self, action, bought_price = None):
        self.index +=1
        if self.index > self.last_data_index:
            self.index = 0
        next_state = self.states[self.index + 1]
        next_price_data = self.data[self.index + 1]
        next_date_data = self.date_data[self.index + 1]
       
        price_data = self.data[self.index]
        
        reward = 0
        if action == 2 and bought_price is not None:
            reward = max(price_data - bought_price, 0)
        
        done = True if self.index == self.last_data_index - 1 else False
        return next_state, next_price_data, next_date_data, reward, done
                       
                       
        
        
        