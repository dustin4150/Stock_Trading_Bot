from keras.models import load_model

from agent import Agent
from market_env import Market

import matplotlib.pyplot as plt
import pandas as pd

def main_eval():
    stock_name = "BABA"
    model_name = "model_ep0"
    
    model = load_model("models/" + model_name)
    window_size = model.layers[0].input.shape.as_list()[1]
    
    agent = Agent(window_size, True, model_name)
    market = Market(window_size, stock_name)
    
    state, price_data, date_data = market.reset()
    date = []
    
    for t in range(market.last_data_index):
            
            
            action, bought_price = agent.act(state, price_data, date_data)
            
            next_state, next_price_data, next_date_data, reward, done = market.get_next_state_reward(action, bought_price)
                
            state = next_state
            price_data = next_price_data
            date_data = next_date_data
            
            if done:
                print("--------------------")
                print("{0} Total profit: {1}".format(stock_name, agent.get_total_profit()))
             
                print("--------------------")
    plot_action_profit(market.data, agent.action_history, agent.get_total_profit())
    return agent.book, agent.initial_investment, agent.dates

def plot_action_profit(data, action_data, profit):
    plt.plot(range(len(data)), data)
    plt.xlabel("data")
    plt.ylabel("price")
    
    buy, sell = False, False
    for d in range(len(data) - 1):
        if action_data[d] == 1: #buy
            buy, = plt.plot(d, data[d], 'g*')
        elif action_data[d] == 2: #sell
            sell, = plt.plot(d, data[d], 'r+')
    if buy and sell:
        plt.legend([buy,sell], ["Buy", "Sell"])
    plt.title("Total Profit: {}".format(profit))
    plt.savefig("buy_sell.png")
    plt.show()
    
if __name__== "__main_eval__":
    main_eval()
 
