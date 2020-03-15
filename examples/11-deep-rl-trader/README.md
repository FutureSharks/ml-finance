# Deep RL Trader (Duel DQN) Implemented using Keras-RL

1. Trading environment(OpenAI Gym) for trading crypto currency  
2. Duel Deep Q Network  
Agent is implemented using `keras-rl`(https://github.com/keras-rl/keras-rl)     

Agent is expected to learn useful action sequences to maximize profit in a given environment.  
Environment limits agent to either buy, sell, hold stock(coin) at each step.  
If an agent decides to take a   
* LONG position it will initiate sequence of action such as `buy- hold- hold- sell`    
* for a SHORT position vice versa (e.g.) `sell - hold -hold -buy`.    

Only a single position can be opened per trade.
* Thus invalid action sequence like `buy - buy` will be considered `buy- hold`.   
* Default transaction fee is : 0.0005  

Reward is given
* when the position is closed or
* an episode is finished.  

## Run

Open the [Jupyter Notebook](run.ipynb)

## Dependencies

`pip3 install -r requirements.txt`

## Sources

- https://github.com/miroblog/deep_rl_trader
