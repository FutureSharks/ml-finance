# Deep Q Learning with simulated price data

This example uses predictable prices generated from a sine wave. This is to ensure the model actually works. The RSI was added as a feature to help the model predict when to buy and sell.

- `agent.py`: a Deep Q learning agent
- `envs.py`: a simple trading environment
- `model.py`: a multi-layer perceptron as the function approximator
- `utils.py`: some utility functions
- `requirement.txt`: all dependencies

## Run

Open the [Jupyter Notebook](deep-q-learning-agent-simulated-prices.ipynb)

## Dependencies

`pip3 install -r requirements.txt`

## Sources

[Example 0: Deep Q Learning with 3 stocks](../0-deep-q-learning-3-stocks)
