import pickle
from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import pandas_datareader.data as web
import pandas_datareader.iex as wb
import pandas as pd
import time
import seaborn as sns
import numpy as np
import os
import math
import labelling as lb
style.use('ggplot')


def get_data_from_yahoo(reload_data=False):
    """Function to get the stock data from Yahoo relating to the following big tech companies:
    AAPL, Facebook, Google, Microsoft, Netflix, Cisco, Intel, IBM, Microsoft, Hewlett-Packward
    Alibaba, Ebay, TripAdvisor
    """
    tickers = ['GOOG', 'AAPL', 'AMZN', 'FB', 'NFLX', 'CSCO', 'INTC', 'IBM', 'MSFT',
               'HPQ', 'BABA', 'EBAY', 'TRIP']

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2020, 12, 31)

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
            time.sleep(2)
        else:
            if reload_data:
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            else:
                print('Already have {}'.format(ticker))


def add_extra_features(df):
    """Function to add the following features to the DataFrame: IntraDay as High Value - Low value
    PreMarket as Open (next day) - Close current days
    Daychange as Percentage change in the day
    5,10,15 and 20 moving average"""
    df['IntraDay'] = np.around(df['High'] - df['Low'], 3)
    df['PreMarket'] = np.around((df['Open'].shift(-1) - df['Close']/df['Close']), 3)
    df['Daychange'] = np.around((df['Close'] - df['Open'])/df['Open'], 3)
    df['MoneyTraded'] = np.around(df['Volume'] * df['Adj Close'], 3)
    ma_day = [5, 10, 15, 20]
    for ma in ma_day:
        column_name = "MA for {} days".format(ma)
        df[column_name] = df['Adj Close'].rolling(ma).mean()
    return df


def compile_data():
    """Compile data from Tickers file into one dataframe."""

    tickers = ['GOOG', 'AAPL', 'AMZN', 'MSFT']
    company_names = ['GOOGLE', 'APPLE', 'AMAZON', 'MICROSOFT']

    Dataframes = []
    for ticker, comp_name in zip(tickers, company_names):
        comp_name_df = comp_name + '_df'
        comp_name_df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), index_col=0)
        comp_name_df.index = pd.to_datetime(comp_name_df.index)
        comp_name_df['company'] = comp_name
        comp_name_df = add_extra_features(comp_name_df)
        Dataframes.append(comp_name_df)

    df = pd.concat(Dataframes, axis=0)
    return df, Dataframes


def plot_adj_close():

    df = compile_data()
    n_companies = df.company.unique()
    plt.figure(figsize=(10, 10))

    for i, comp_name in enumerate(n_companies, 1):
        plt.subplot(2, 2, i)
        df[df['company'] == comp_name]['Adj Close'].plot()
        plt.ylabel("Adj Close")
        plt.xlabel(None)
        plt.title("{}".format(comp_name))
    plt.show()


def plot_volume():

    df, _ = compile_data()
    n_companies = df.company.unique()
    plt.figure(figsize=(10, 10))

    for i, comp_name in enumerate(n_companies, 1):
        plt.subplot(2, 2, i)
        df[df['company'] == comp_name]['Volume'].plot()
        plt.ylabel("Volume Traded")
        plt.xlabel(None)
        plt.title("{}".format(comp_name))
    plt.show()


def plot_moving_average():

    df, _ = compile_data()
    n_companies = df.company.unique()
    plt.figure(figsize=(10, 10))
    fig, axes = plt.subplots(nrows=2, ncols=2)

    df[df['company'] == 'GOOGLE'][['Adj Close', 'MA for 5 days',
                                  'MA for 10 days', 'MA for 15 days',
                                   'MA for 20 days']].plot(ax=axes[0, 0])
    axes[0, 0].set_xlabel(None)
    axes[0, 0].set_title('GOOGLE')

    df[df['company'] == 'APPLE'][['Adj Close', 'MA for 5 days',
                                  'MA for 10 days', 'MA for 15 days',
                                  'MA for 20 days']].plot(ax=axes[0, 1])
    axes[0, 1].set_title('AAPLE')
    axes[0, 1].set_xlabel(None)

    df[df['company'] == 'AMAZON'][['Adj Close', 'MA for 5 days',
                                  'MA for 10 days', 'MA for 15 days',
                                   'MA for 20 days']].plot(ax=axes[1, 0])
    axes[1, 0].set_title('AMAZON')
    axes[1, 0].set_xlabel(None)

    df[df['company'] == 'MICROSOFT'][['Adj Close', 'MA for 5 days',
                                     'MA for 10 days', 'MA for 15 days',
                                      'MA for 20 days']].plot(ax=axes[1, 1])
    axes[1, 1].set_title('MICROSOFT')
    axes[1, 1].set_xlabel(None)

    plt.show()


def plot_dailyreturns():

    df, _ = compile_data()
    n_companies = df.company.unique()
    plt.figure(figsize=(10, 10))

    for i, comp_name in enumerate(n_companies, 1):
        plt.subplot(2, 2, i)
        df[df['company'] == comp_name]['Daychange'].plot()
        plt.ylabel("Day Change")
        plt.xlabel(None)
        plt.title("{}".format(comp_name))
    plt.show()


def plot_dailyreturns_distribution():

    df, _ = compile_data()
    n_companies = df.company.unique()
    plt.figure(figsize=(10, 10))

    for i, comp_name in enumerate(n_companies, 1):
        plt.subplot(2, 2, i)
        sns.distplot(df[df['company'] == comp_name]['Daychange'].dropna(),
                     bins=200, color='red')
        plt.ylabel("Day Change")
        plt.title("{}".format(comp_name))
    plt.show()


def correlate_stocks():

    df, _ = compile_data()
    company_dict = dict()

    for comp_name in df.company.unique():
        company_dict[comp_name] = {'date': df[(df['company'] == comp_name)
                                    & (df.index.year >= 2005)].index,
                                   'price': np.array(df[(df['company'] == comp_name)
                                    & (df.index.year >= 2005)]['Adj Close'])}

    tech_corr = pd.DataFrame({'APPLE': company_dict['APPLE']['price'],
                             'GOOGLE': company_dict['GOOGLE']['price'],
                              'MICROSOFT': company_dict['MICROSOFT']['price'],
                              'AMAZON': company_dict['AMAZON']['price']}).set_index(company_dict['APPLE']['date'])
    return tech_corr


def plot_correlation(timeframe=''):
    """Plot correlation between stocks. Interesting to note that the correlation between stocks
    change considerably depending on the timeframe considered. This results from the fact
    many stocks have changed/grew quite significantly/differently from each other and over the years"""
    tech_corr = correlate_stocks()
    if (timeframe != ''):
        tech_corr = tech_corr[tech_corr.index.year >= timeframe]

    tech_fig = sns.PairGrid(tech_corr.dropna())
    tech_fig.map_upper(plt.scatter, color='red')
    tech_fig.map_lower(sns.kdeplot, cmap='cool_d')
    tech_fig.map_diag(plt.hist, bins=50)
    #plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.heatmap(tech_corr.corr(), ax=ax[0], annot=True)
    ax[0].set_title("Correlation Matrix  Adj Close")

    sns.heatmap(tech_corr.pct_change().corr(), ax=ax[1], annot=True)
    ax[1].set_title("Correlation Matrix Percentage change")
    plt.show()
