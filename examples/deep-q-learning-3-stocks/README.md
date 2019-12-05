# Deep Q Learning with 3 stocks

- `agent.py`: a Deep Q learning agent
- `envs.py`: a simple 3-stock trading environment
- `model.py`: a multi-layer perceptron as the function approximator
- `utils.py`: some utility functions
- `requirement.txt`: all dependencies
- `data/`: Pricing data for IBM, MSFT, and QCOM from 01-03-2000 to 27-12-2017

## Run

Open the [Jupyter Notebook](deep-q-learning-agent.ipynb)

## Dependencies

`pip3 install -r requirements.txt`

## Issues

- The environment is too simplistic with buying a single stock and selling all stocks
- Add short selling option
- Can't use current bar close price for buying. A price from the next bar must be used.

## Sources

https://github.com/ShuaiW/teach-machine-to-trade

https://github.com/llSourcell/Q-Learning-for-Tradi
