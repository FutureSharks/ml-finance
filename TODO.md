# To do

- Consider changes here: https://github.com/llSourcell/Q-Learning-for-Trading/issues/2
- Create issues in repos pointing to this one
- Test example 5 with simulated prices
- Don't use scaler in function for example 2 and 3
- Use virtualenvs
- Find nice example of reinforcement using pytorch
- https://github.com/tomgrek/RL-stocktrading/blob/master/Finance%20final.ipynb
- Test different batch sizes
- "We want to incentivize profit that is sustained over long periods of time. At each step, we will set the reward to the account balance multiplied by some fraction of the number of time steps so far."
- # Set the current step to a random point within the data frame,   self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
