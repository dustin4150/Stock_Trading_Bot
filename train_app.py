from agent import Agent
from market_env import Market

import os
import time

def main_train():
    #last 10 changes of stock price
    windows_size = 10
    #how many epochs
    episode_count = 100
    stock_name = "BABA"
    batch_size = 32
    
    agent = Agent(windows_size)
    market = Market(windows_size=windows_size, stock_name=stock_name)
    
    start_time = time.time()
    for e in range(episode_count + 1):
        print("Episode {0}/{1}.".format(e, episode_count))
        agent.reset()
        state, price_data = market.reset()
        
        for t in range(market.last_data_index):
            action, bought_price = agent.act(state, price_data)
            
            next_state, next_price_data, reward, done = market.get_next_state_reward(action, bought_price)
            agent.memory.append((state, action, reward, next_state, done))
            
            if len(agent.memory) > batch_size:
                agent.experience_replay(batch_size)
                
            state = next_state
            price_data = next_price_data
            
            if done:
                print("--------------------")
                print("Total profit: {0}".format(agent.get_total_profit()))
         
                print("--------------------")
                
        if e % 10 == 0:
            if not os.path.exists("models"):
                os.mkdir("models")
            agent.model.save("models/model_ep" + str(e))
            
    end_time = time.time()
    training_time = end_time - start_time
    print("Training time took {0} seconds.".format(training_time))
    
    
            

if __name__ == "__main_train__":
    main_train()