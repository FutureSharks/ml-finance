import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv(gym.Env):
    """
    A 3-stock (MSFT, IBM, QCOM) trading environment.

    State: [# of stock owned, current stock prices, cash in hand]
        - array of length n_stock * 2 + 1
        - price is discretized (to integer) to reduce state space
        - use close price for each stock
        - cash in hand is evaluated at each step based on action performed

    Action: sell (0), hold (1), and buy (2)
        - when selling, sell all the shares
        - when buying, buy as many as cash in hand allows
        - if buying multiple stock, equally distribute cash in hand and then utilize the balance
    """
    def __init__(self, train_data, init_invest=20000):
        # data
        self.stock_price_history = np.around(train_data) # round up to integer to reduce state space
        self.n_stock, self.n_step = self.stock_price_history.shape

        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # Trading statistics
        self.trade_count = 0
        self.trades_profitable = 0
        self.current_value = 0
        self.book_cost = [0] * self.n_stock

        # action space
        self.action_space = spaces.Discrete(3**self.n_stock)

        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
        price_range = [[0, mx] for mx in stock_max_price]
        cash_in_hand_range = [[0, init_invest * 2]]
        self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)

        # seed and start
        self._seed()
        self._reset()

    def _stats(self):
        '''
        Returns a dict of trading statistics
        '''
        return {
            'trade_count': self.trade_count,
            'win_loss_ratio': self.trades_profitable / self.trade_count,
            'value': self.current_value,
        }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.cur_step = 0
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.cash_in_hand = self.init_invest
        self.trade_count = 0
        self.trades_profitable = 0
        self.current_value = 0
        self.book_cost = [0] * self.n_stock
        return self._get_obs()

    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1
        self.stock_price = self.stock_price_history[:, self.cur_step] # update price
        self._trade(action)
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        self.current_value = cur_val
        return self._get_obs(), reward, done

    def _get_obs(self):
        obs = []
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.append(self.cash_in_hand)
        return obs


    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        action_vec = action_combo[action]

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
            for i in sell_index:
                if self.stock_owned[i] > 0:
                    self.trade_count += 1
                    sell_amount = self.stock_price[i] * self.stock_owned[i]
                    if sell_amount > self.book_cost[i]:
                        self.trades_profitable += 1
                    self.cash_in_hand += sell_amount
                    self.stock_owned[i] = 0
                    self.book_cost[i] = 0

        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                        self.book_cost[i] += self.stock_price[i] * 1
                    else:
                        can_buy = False
