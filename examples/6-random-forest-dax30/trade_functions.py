# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


def create_trades_from_positions(positions, spread=0):
    '''
    Takes a dataframe of OHLC pricing data with a position column and returns a
    dataframe of trades.

    position:
      -1: Short
       0: Hold
       1: Long
    '''
    # Close last position
    positions.iloc[-1, positions.columns.get_loc('position')] = 0

    # Create position groups
    positions['position_group'] = (positions['position'].diff(1) != 0).astype('int').cumsum()
    positions['position_group_shifted'] = positions['position_group'].shift(1)

    # Create trades DataFrame
    trades = pd.DataFrame({
            'enter_date': positions.reset_index().groupby('position_group').date.first(),
            'enter_price': positions.reset_index().groupby('position_group')['price'].first(),
            'exit_date': positions.reset_index().groupby('position_group_shifted').date.last(),
            'exit_price': positions.reset_index().groupby('position_group_shifted')['price'].last(),
            'position_length': positions.groupby('position_group').size(),
            'position': positions.groupby('position_group')['position'].first(),
        }).reset_index(drop=True)

    # Remove trades with neutral position
    trades = trades[trades.position != 0].reset_index(drop=True)

    if len(trades) == 0:
        print("No trades generated")
        return

    # Create some columns to calculate stats
    trades['profit'] = ((trades['exit_price'] - trades['enter_price']) * trades['position']) - spread
    trades['profitable'] = trades['profit'] > 0

    return trades


def get_trade_statistics(trades):
    '''
    Takes a dataframe as output from create_trades_from_positions and prints
    statistics about the trades
    '''

    stats = {
        'profitable': len(trades[trades['profitable'] == True]) / len(trades),
        'ratio_long_short': len(trades[trades['position'] == 1]) / len(trades[trades['position'] == -1]),
        'median_profit': trades['profit'].median(),
        'total_profit': trades['profit'].sum(),
        'median_position_length': trades['position_length'].median(),
        'trades': len(trades),
    }

    print('Profitable: {0}%'.format(round(stats['profitable'] * 100, 2)))
    print('Ratio of long to short positions: {0}'.format(round(stats['ratio_long_short'], 2)))
    print('Median profit: {0}'.format(round(stats['median_profit'], 2)))
    print('Total profit: {0}'.format(round(stats['total_profit'], 2)))
    print('Median position length: {0}'.format(stats['median_position_length']))
    print('Number of trades: {0}'.format(stats['trades']))

def show_positions_on_price_plot(positions, extra_y_series=None, columns_to_keep=[], figsize=(16,6)):
    '''
    Creates a plot showing positions as coloured bands with price

    Red: short
    Green: long
    '''
    # Create position groups
    positions['position_group'] = (positions['position'].diff(1) != 0).astype('int').cumsum()
    positions['position_group_day_after'] = positions['position_group'].shift(1)

    positions_to_plot = positions.copy()

    for column in positions_to_plot.columns:
        if column != 'price' and column not in columns_to_keep:
            positions_to_plot.drop(column, 1, inplace=True)

    ax = positions_to_plot.plot(figsize=figsize)
    ymax = positions_to_plot['price'].max()
    ymin = positions_to_plot['price'].min()

    non_zero_positions = positions[positions.position != 0]
    position_groups = list(set(non_zero_positions['position_group'].tolist()))

    print('{0} positions to plot...'.format(len(position_groups)))

    for group in position_groups:
        enter_loc = non_zero_positions.loc[non_zero_positions['position_group'] == group].index[0]
        exit_loc = non_zero_positions.loc[non_zero_positions['position_group'] == group].index[-1]
        position = non_zero_positions.loc[non_zero_positions['position_group'] == group]['position'].values[0]

        if position == 1:
            color = '#72a8ff'
        else:
            color = '#ff9e9e'

        ax.fill_between([enter_loc, exit_loc], ymin, ymax, color=color)
        ax.annotate('LONG', (0,0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top', size="large", color='#72a8ff')
        ax.annotate('SHORT', (0,0), (80, -60), xycoords='axes fraction', textcoords='offset points', va='top', size="large", color='#ff9e9e')

    plt.title('Positions', size='x-large')

    return ax
