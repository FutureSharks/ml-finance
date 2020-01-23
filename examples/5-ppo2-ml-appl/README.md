# PPO2 and MLP trading AAPL stock

- `StockTradingEnv.py`: A trading environment
- `data/AAPL.csv`: AAPL stock price data for 1998-2018

## Run

Open the [Jupyter Notebook](run.ipynb)

## Dependencies

You need to install [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/guide/install.html
):

```
brew install cmake openmpi
pip3 install stable-baselines[mpi]
```

## Issues

- Is the model even tested in this example?
- Surely the stock price split breaks something here

## Sources

- https://medium.com/@adamjking3/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
- https://github.com/notadamking/Stock-Trading-Environment
