import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class SimpleTradingEnvironment(gym.Env):
    """
    A simple trading environment:
        - No complexity about position sizing based on account balance
        - Just takes short or long positions
        - Reward is simply enter and exit price difference

    Arguments:
        price_data: A pandas DataFrame of pricing data.

        price_column_name: The name of the column containing the prices in the
            pandas DataFrame

        environment_columns: Names of columns to include in the
            environment, for example technical indicators and the price column.

    Action:
        0: Short
        1: Hold
        2: Long
    """
    def __init__(self, price_data, environment_columns, price_column_name='price'):
        self.price_data = price_data
        self.price_column_name = price_column_name
        self.environment_columns = environment_columns
        self.n_step = len(self.price_data)

        # instance attributes
        self.current_position = 1
        self.current_price = 0
        self.enter_price = 0

        ## Trading statistics
        # number of completed trades
        self.trade_count = 0
        # Number of profitable completed trades
        self.trades_profitable = 0
        # Balance from completed trades
        self.account_balance = 0
        # Balance from completed and open trades
        self.account_balance_unrealised = 0

        # action space
        self.action_space = spaces.Discrete(3)

        # observation space: give estimates in order to sample and build scaler
        data_max = self.price_data[self.environment_columns].max().tolist()
        data_range = [[0, i] for i in data_max]
        position_range = [[0, 2]]
        self.observation_space = spaces.MultiDiscrete([data_max] + data_range + position_range)

        # seed and start
        self._seed()
        self._reset()

    def _stats(self):
        '''
        Returns a dict of trading statistics
        '''
        if self.trade_count == 0:
            win_loss_ratio = 0
        else:
            win_loss_ratio = self.trades_profitable / self.trade_count

        return {
            'trade_count': self.trade_count,
            'win_loss_ratio': win_loss_ratio,
            'account_balance': self.account_balance,
            'unrealised_pl': self._get_unrealised_pl(),
        }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.trade_count = 0
        self.trades_profitable = 0
        self.account_balance = 0
        self.account_balance_unrealised = 0
        self.current_step = 0
        self.current_position = 1
        self.current_price = 0
        self.enter_price = 0
        self.stock_price = self.price_data[self.price_column_name][self.current_step]
        self.enter_price = 0
        return self._get_observations()

    def _step(self, action):
        assert self.action_space.contains(action)
        previous_pl = self._get_unrealised_pl()
        self.current_step += 1
        self.current_price = self.price_data[self.price_column_name][self.current_step]
        self._trade(action)
        self.account_balance_unrealised = self._get_unrealised_pl()
        reward = self.account_balance_unrealised - previous_pl
        done = self.current_step == len(self.price_data) - 1
        return self._get_observations(), reward, done

    def _get_observations(self):
        observations = []
        # Current position
        observations.append(self.current_position)
        # Account balance unrealised
        observations.append(self.account_balance_unrealised)
        # Price and environment columns
        observations.extend(self.price_data[self.environment_columns].iloc[self.current_step].values)
        return observations

    def _get_unrealised_pl(self):
        '''
        Calculates the current unrealised profit and loss.
        This is the unrealised P&L from an open position + the current account balance
        '''
        return ((self.current_price - self.enter_price) * (self.current_position - 1)) + self.account_balance

    def _trade(self, action):
        '''
        Performs trades based on action:
            0: Short
            1: Hold
            2: Long
        '''
        # Nothing to do if action is the same as current position
        if self.current_position == action:
            pass

        # Opening a new trade from hold
        elif self.current_position == 1:
            self.enter_price = self.current_price
            self.current_position = action

        # Closing or switching position
        elif self.current_position in [0, 2]:
            assert self.enter_price is not 0
            profit = (self.current_price - self.enter_price) * (self.current_position - 1)
            self.account_balance += profit
            if profit > 0:
                self.trades_profitable += 1

            # Closing a position
            if action == 1:
                self.enter_price = 0
                self.trade_count += 1
            # Open new position
            else:
                self.enter_price = self.current_price

            self.current_position = action
        else:
            raise Exception('Unknown trade situation')

        return
