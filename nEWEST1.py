import datetime
import traceback


from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from catboost import cv
from scipy.stats import randint
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt

import scipy
import yfinance as yf
from scipy.stats.mstats import winsorize
import pandas as pd
from scipy import stats
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pickle
import datetime
import concurrent.futures
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import *
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import *
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
import yfinance as yf
import warnings
import math

from alpaca.trading.models import *
import time
import robin_stocks.robinhood as r
from datetime import datetime
import traceback
import pyotp
import catboost as cb
from catboost import CatBoostClassifier
from sklearn import metrics 
from sklearn.model_selection import TimeSeriesSplit
import asyncio
import warnings
warnings.filterwarnings('ignore')
import numba
from numba import jit

from scipy.signal import savgol_filter
from scipy.signal import *

pd.set_option("display.precision", 3)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)

print(datetime.now())



symbol = "MSFT"
BASE_URL = "https://paper-api.alpaca.markets"
API_KEY = "PKBX5XZQ1JG2CEODIOKD"
SECRET_KEY = "laKd5n4c7pnjRT9nC6WJztVEWruDz2b1VDJab5Hg"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

midpoint_AAPL = 0
midpoint_MSFT = 0
midpoint_TSLA = 0
midpoint = 0

def get_orderbook(symbol):

    global midpoint_AAPL
    global midpoint_MSFT
    global midpoint_TSLA

    try:
        symbol = symbol
        totp  = pyotp.TOTP("HOPRBD4K5QWBMKCW").now()
        #print("Current OTP:", totp)

        username = "torndoff@icloud.com"
        password = "qu2t3f8Ew9BxM"

        login = r.login(username,password, mfa_code=totp)


        data = r.stocks.get_pricebook_by_symbol(symbol=symbol)

        
        #print(pd.DataFrame.from_dict(data['asks']))
        all_prices = [dct["price"] for dct in data["asks"]]
        all_amounts = [dct["amount"] for dct in all_prices]
        
        x = pd.DataFrame.from_dict(data['asks'])
        x['price'] = all_amounts
        all_prices = [dct["price"] for dct in data["bids"]]
        all_amounts = [dct["amount"] for dct in all_prices]
        

        y = pd.DataFrame.from_dict(data['bids'])

        y['price'] = all_amounts
        best_ask = float(x['price'][0])
        best_bid = float(y['price'][0])

        midpoint = (best_ask + best_bid)/2
        x['midpoint'] = float(midpoint)
        x['price'].apply(lambda x: float(x))

        x['delta'] = pd.to_numeric(x['price']) - x['midpoint']
        x['delta'] = abs(x['delta'])
        x['delta'] = pow(x['delta'], -1)
        x['delta_volume'] = x['delta'] * x['quantity']
        x = x[:10]
        ask_sum_delta_vol = x['delta_volume'].sum() * len(x)
        #print(ask_sum_delta_vol)
        y['midpoint'] = float(midpoint)
        y['price'].apply(lambda y: float(y))
        y['delta'] = pd.to_numeric(y['price']) - y['midpoint']
        y['delta'] = abs(y['delta'])
        y['delta'] = pow(y['delta'], -1)
        y['delta_volume'] = y['delta'] * y['quantity']
        y = y[:10]
        bid_sum_delta_vol = y['delta_volume'].sum() * len(y)
        #print(bid_sum_delta_vol)

        alpha = abs((ask_sum_delta_vol - bid_sum_delta_vol) / ((bid_sum_delta_vol + ask_sum_delta_vol)/2))
        ask_alpha = (ask_sum_delta_vol - bid_sum_delta_vol) / ((bid_sum_delta_vol + ask_sum_delta_vol)/2)
        if ask_alpha <= 0:
            ask_alpha = 0.001
        bid_alpha = (bid_sum_delta_vol - ask_sum_delta_vol) / ((bid_sum_delta_vol + ask_sum_delta_vol)/2)
        if bid_alpha <= 0:
            bid_alpha = 0.001
        #print("alpha:", alpha)
        

    except:
        ask_alpha = 0.01
        bid_alpha = 0.01
        print(traceback.format_exc())
        
    inventory_qty = 1
    try:
        symbol = symbol
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    
        position = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(position)
        inventory_qty = int(ORDERS[1][6])

        avg_entry_price = float(ORDERS[1][5])
        side = str(ORDERS[1][7])

    except:

        print("No inventory position.")
        inventory_qty = 1
        #print(traceback.format_exc())
        ask_alpha = 0.01
        bid_alpha = 0.01
        
        
        


        

    symbol = symbol
    
    df = pd.DataFrame(yf.download(symbol, period="1d", interval="1m"))
    """
    df['highest'] = df['High'].cummax() #take the cumulative max
    df['lowest'] = df['Low'].cummax() #take the cumulative max
    df['buy_trailingstop'] = df['highest']*0.9995 #subtract 1% of the max
    df['sell_trailingstop'] = df['lowest']*1.0005 #add 1% of the max

    if side == 'PositionSide.LONG':
        if avg_entry_price > df['buy_trailingstop'][-1]:
            cancel_orders_for_symbol(symbol=symbol)
            trading_client.close_position(symbol)
            print("\n buy trailing stop sleep \n")
            #time.sleep(1)

    if side == 'PositionSide.SHORT':
        if avg_entry_price < df['sell_trailingstop'][-1]:
            cancel_orders_for_symbol(symbol=symbol)
            trading_client.close_position(symbol)
            print("\n sell trailing stop sleep \n")
            #time.sleep(1)
    """
    df['inventory'] = inventory_qty
    #df = df.iloc[::-1]
    df["mu"] = abs((np.log(df["Open"].ewm(span=5).mean()).pct_change()/2) * 10000)
    #print('mu: ', df["mu"][:-1])

    df['gamma'] = get_inventory_risk(symbol = symbol)
    #print('gamma: ', df["gamma"][:-1])

    df['sigma'] = ((np.log(df["Open"]).ewm(span=5).std() * np.sqrt(5)).ewm(span=5).mean()) * 100
    #print('sigma: ', df["sigma"][:-1])

    df['Volume'] = df['Volume'] + 1


    df['k'] = (0.5*(df['sigma'])*np.sqrt(df['Volume']/df['Volume'].ewm(span=5).mean()))*1
    #print('k: ', df["k"][:-1])
    #(df['k'] / 2 * df['sigma'] * df['gamma']**2) * 100000
    df['bid_alpha'] = bid_alpha
    df['ask_alpha'] = ask_alpha

    df['bid_sum_delta_vol'] = bid_sum_delta_vol
    df['ask_sum_delta_vol'] = ask_sum_delta_vol
    

    df['bid_spread_aysm'] = ((1 / df['gamma'] * np.log(1 + df['gamma'] / df['k']) + (2 * df['inventory'] + 1) / 2 * np.sqrt((df['sigma']**2 * df['gamma']) / (2 * df['k'] * df['bid_alpha']) * (1 + df['gamma'] / df['k'])**(1 + df['k'] / df['gamma']))) / 100000)

    df['ask_spread_aysm'] = ((1 / df['gamma'] * np.log(1 + df['gamma'] / df['k']) - (2 * df['inventory'] - 1) / 2 * np.sqrt((df['sigma']**2 * df['gamma']) / (2 * df['k'] * df['ask_alpha']) * (1 + df['gamma'] / df['k'])**(1 + df['k'] / df['gamma']))) / 100000)
    


    
    df['bid_spread_aysm2'] = ((1 / df['gamma'] * np.log(1 + df['gamma'] / df['k']) + (- df["mu"] / (df['gamma'] * df['sigma']**2) + (2 * df['inventory'] + 1) / 2) * np.sqrt((df['sigma']**2 * df['k']) / (2 * df['k'] * df['bid_alpha']) * (1 + df['gamma'] / df['k'])**(1 + df['k'] / df['gamma']))) / 19999998)

    df['ask_spread_aysm2'] = ((1 / df['gamma'] * np.log(1 + df['gamma'] / df['k']) + (  df["mu"] / (df['gamma'] * df['sigma']**2) - (2 * df['inventory'] - 1) / 2) * np.sqrt((df['sigma']**2 * df['k']) / (2 * df['k'] * df['ask_alpha']) * (1 + df['gamma'] / df['k'])**(1 + df['k'] / df['gamma']))) / 19999998)

    #print(df['bid_spread_aysm'][-1])
    #print(df['ask_spread_aysm'][-1])
    print("\n bid: \n", df['bid_spread_aysm2'][-1])
    print("\n ask: \n", df['ask_spread_aysm2'][-1])

    best_ask = df['ask_spread_aysm2'][-1]
    best_bid = df['bid_spread_aysm2'][-1]

    if symbol == 'MSFT':
        midpoint_MSFT = midpoint
        df['midpoint_MSFT'] = midpoint

    if symbol == 'AAPL':
        midpoint_AAPL = midpoint
        df['midpoint_AAPL'] = midpoint

    if symbol == 'TSLA':
        midpoint_TSLA = midpoint
        df['midpoint_TSLA'] = midpoint

    return best_bid, best_ask, midpoint, df, inventory_qty






def get_time_til_close(symbol):

    try:
        symbol = symbol
        now = datetime.now()
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        

        if int(now.hour) > 21:
            if int(now.minute) > 57:
                cancel_orders_for_symbol(symbol=symbol)
                trading_client.close_position(symbol)
                time.sleep(1)
    except:
        print("get_time_til_close exception. It's not trading time...")
        #print(traceback.format_exc())
        
    
def get_inventory_risk(symbol):

    try:
        symbol = symbol

        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    
        position = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(position)
        inventory_qty = int(ORDERS[1][6])

        if symbol == "MSFT":
            inventory_risk = 0.00002 * abs(inventory_qty)

        if symbol == 'AAPL':
            inventory_risk = 0.00002 * abs(inventory_qty)

    except:

        print("No inventory position.")
        inventory_qty = 1
        inventory_risk = 0.00002
        #print(traceback.format_exc())
        
        

    

    return inventory_risk


def get_open_position(symbol):

    try:
        symbol = symbol
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    
        position = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(position)
    
        inventory_qty = int(ORDERS[1][6])

        return inventory_qty

    except:

        print("No inventory position.")
        inventory_qty = 1
        #print(traceback.format_exc())
        return inventory_qty



def take_profit_method(symbol):
    try:
            
        symbol = symbol
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        position = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(position)

        if float(ORDERS[1][10]) / abs(float(ORDERS[1][6])) >=  0.05:
            cancel_orders_for_symbol(symbol=symbol)
            time.sleep(1)
            trading_client.close_position(symbol)

        if float(ORDERS[1][10]) / abs(float(ORDERS[1][6])) <=  -0.04:
            cancel_orders_for_symbol(symbol=symbol)
            time.sleep(1)
            trading_client.close_position(symbol)
        
        if float(ORDERS[1][12]) >=  4:
            
            cancel_orders_for_symbol(symbol=symbol)
            time.sleep(1)
            trading_client.close_position(symbol)
            

        elif float(ORDERS[1][12]) <=  -5:

            cancel_orders_for_symbol(symbol=symbol)
            time.sleep(1)
            trading_client.close_position(symbol)
            

        elif str(ORDERS[1][7]) ==  "PositionSide.LONG":

            if float(ORDERS[1][5]) - float(ORDERS[1][14]) <= -0.1:
                cancel_orders_for_symbol(symbol=symbol)
                time.sleep(1)
                trading_client.close_position(symbol)
                

            if float(ORDERS[1][5]) - float(ORDERS[1][14]) >= 0.1:
                cancel_orders_for_symbol(symbol=symbol)
                time.sleep(1)
                trading_client.close_position(symbol)
                


        elif str(ORDERS[1][7]) ==  "PositionSide.SHORT":

            if float(ORDERS[1][5]) - float(ORDERS[1][14]) >= 0.1:
                cancel_orders_for_symbol(symbol=symbol)
                time.sleep(1)
                trading_client.close_position(symbol)
                

            if float(ORDERS[1][5]) - float(ORDERS[1][14]) <= -0.1:
                cancel_orders_for_symbol(symbol=symbol)
                time.sleep(1)
                trading_client.close_position(symbol)
                

    except:
        print ("take_profit_method error.")
        #print(traceback.format_exc())
        






@jit
def limit_order(symbol, spread, side, take_profit_multiplier, loss_stop_multiplier, loss_limit_multiplier, qty, inventory_risk):
    
    symbol = str(symbol)
    best_bid, best_ask, midpoint, df, inventory_qty = get_orderbook(symbol = symbol)

    dataset = df


    dataset['spread'] = abs(np.log(dataset['Open']).ewm(span=5).mean() - ((np.log(dataset['Low']).ewm(span=5).mean() + np.log(dataset['High']).ewm(span=5).mean())/2))
    dataset['variance'] = (np.log(dataset['Open']).ewm(span=5).std() * np.sqrt(5)).ewm(span=5).mean()
    current_variance = (dataset["variance"][-1])
    current_spread = (dataset["spread"][-1])
    current_variance = float(current_variance) * 1000
    current_spread = float(current_spread) * 1000


    
    now = datetime.now()

    end_of_day = datetime(now.year, now.month, now.day, hour=22)




    steps_in_day = end_of_day - now
    steps_in_day = float(round(steps_in_day.total_seconds()/60))
    #steps_in_day = 100
    #mid_price = float(current_price)

    if symbol == 'MSFT':
        midpoint = midpoint_MSFT

    if symbol == 'AAPL':
        midpoint = midpoint_AAPL

    if symbol == 'TSLA':
        midpoint = midpoint_TSLA

    mid_price = float(midpoint)

    inventory = float(inventory_qty)
    inventory_risk = float(inventory_risk)
    variance = float(current_variance)
    total_steps_in_day = float(420)
    #print(steps_in_day)
    print("\n inventory: \n", inventory)

    

    res = np.array(mid_price) - ((np.array(inventory) * np.array(inventory_risk) * (np.array(variance)) * (1 - (np.array(steps_in_day, dtype='float64')/np.array(total_steps_in_day))))/4)
    #np.array(current_spread
    print("\n reservation price: \n", res)
    print("\n reservation price delta: \n", res-mid_price)

    


    


    
    best_spread = 0.0001

    

    print("\n midpoint: \n", midpoint)



    if side == 'OrderSide.BUY':
        cancel_orders_for_side(symbol=symbol, side='sell')
        best_spread = best_bid

    if side == 'OrderSide.SELL':
        cancel_orders_for_side(symbol=symbol, side='buy')
        best_spread = best_ask

    spread = best_spread + spread
    current_price = res
    limit_price = round(current_price + spread, 2)


    #current_price = yf.download(symbol, period="1d", interval="1m")
    #current_price = float(current_price["Open"][-1:])

    market_order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=int(qty),
                    side=side,
                    type='limit',
                    time_in_force=TimeInForce.GTC,
                    limit_price = limit_price,
                    #take_profit={'limit_price': round((limit_price+ (spread*take_profit_multiplier)), 2)},
                    #stop_loss={'stop_price': round((limit_price+ (spread*loss_stop_multiplier)), 2),
                    #'limit_price':  round((limit_price+ (spread*loss_limit_multiplier)), 2)},
                    
                )
    limit_order_data = trading_client.submit_order(market_order_data)

    #print("spread, limit_price: ", spread, limit_price)
    #print(limit_order_data)


            
    
    
    

    

def cancel_orders_for_symbol(symbol):

    try:
        
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        get_order = GetOrdersRequest(symbols=[symbol], limit=500)
        ORDERS = trading_client.get_orders(get_order)
        ORDERS = pd.DataFrame.from_records(ORDERS)
        #print("\n ORDERS for cancel_for_symbol: \n", ORDERS)
        ORDERS = ORDERS[0]
        ORDERS1 = []
        for i in range(0,len(ORDERS)):
            ORDERS2 = ORDERS[i]
            ORDERS2 = ORDERS2[1]
            
            ORDERS1.append(ORDERS2)
        #print(ORDERS1)

        for i in ORDERS1:
            trading_client.cancel_order_by_id(i)

    except:
        print("cancel_orders_for_symbol exception")
        #print(traceback.format_exc())
        

def cancel_orders_for_side(symbol, side):

    try:

        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        get_order = GetOrdersRequest(symbols=[symbol], limit=500, side=side)
        ORDERS = trading_client.get_orders(get_order)
        ORDERS = pd.DataFrame.from_records(ORDERS)
        #print("order for side", ORDERS)
        ORDERS = ORDERS[0]
        ORDERS1 = []
        for i in range(0,len(ORDERS)):
            ORDERS2 = ORDERS[i]
            ORDERS2 = ORDERS2[1]
            
            ORDERS1.append(ORDERS2)
        #print(ORDERS1)

        for i in ORDERS1:
            trading_client.cancel_order_by_id(i)

    except:
        print("cancel_orders_for_side exception")
        #print(traceback.format_exc())
        pass


def match_orders_for_symbol(symbol):

    try:
        symbol = symbol
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        ORDERS = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(ORDERS)
        #print('\n ORDERS: \n',ORDERS)
        side = str(ORDERS[1][7])
        qty = float(ORDERS[1][20])
        best_bid, best_ask, midpoint, df, inventory_qty = get_orderbook(symbol = symbol)
        best_bid = best_bid / 2.0
        best_ask = best_ask / 2.0

        if str(side) == 'PositionSide.SHORT':

            cancel_orders_for_side(symbol=symbol, side='buy')
            limit_order(symbol=symbol, 
                        spread=-0.021 + (best_bid),
                        side=OrderSide.BUY, 
                        take_profit_multiplier = 2,
                        loss_stop_multiplier = 2,
                        loss_limit_multiplier = 2.1,
                        qty = abs(float(ORDERS[1][20])),
                        inventory_risk = get_inventory_risk(symbol = symbol)
                        )
        

        if str(side) == 'PositionSide.LONG':
            
            cancel_orders_for_side(symbol=symbol, side='sell')
            limit_order(symbol=symbol, 
                        spread=0.021 + best_ask, 
                        side=OrderSide.SELL, 
                        take_profit_multiplier = 2,
                        loss_stop_multiplier = 2,
                        loss_limit_multiplier = 2.1,
                        qty = abs(float(ORDERS[1][20])),
                        inventory_risk = get_inventory_risk(symbol = symbol)
                        )
        
        

    except:    
        print("match_orders_for_symbol() exception, probably no inventory present.")  
        #print(traceback.format_exc())   
        pass



def metric(y_true, y_pred):

    Accuracy = metrics.accuracy_score(y_true, y_pred)
    F1_score = metrics.f1_score(y_true, y_pred, average="macro")
    print("\n Accuracy:  \n", Accuracy)
    print("\n F1_score:  \n", F1_score)

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    """cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1])
    cm_display.plot()
    plt.show()"""


@jit
def std_normalized(vals):
    return np.std(vals) / np.mean(vals)

# Ratio of diff between last price and mean value to last price
@jit
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]

# z-score for volumes and price
@jit
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)

@jit
def z_score(vals):
    vals = np.log(vals)
    vals = ((vals - vals.expanding().mean())/vals.expanding().std()).pct_change()
    return vals


def z_score_df(df):
    df = df.apply(lambda x : z_score(x))
    return df

@jit
def make_model(dataset, symbol, side):
        
        t0 = time.time()

        symbol = str(symbol)
        get_time_til_close(symbol=symbol)
        take_profit_method(symbol=symbol)
        match_orders_for_symbol(symbol=symbol)


        column_price = 'open'
        column_high = 'high'
        column_low = 'low'
        column_volume = 'volume'

        


        # Feature params
        future_period = 1


        future_period1 = 10
        std_period = 15
        ma_period = 15
        price_deviation_period = 15
        volume_deviation_period = 15
        #dataset['volume'] = dataset['volume']/1000
        #print('dataset: ', dataset)
        #dataset = dataset[-400:]
        dataset_AAPL = dataset

        
        dataset['future_return'] = dataset['open'].pct_change(future_period).shift(-future_period)
        #print(dataset['future_return'])
        y = np.sign(dataset['future_return'])
        y_sell = np.sign(dataset['future_return'])
        y = y.replace({-1:0})
        y_sell = y_sell.replace({1:0})
        y_sell = y_sell.replace({-1:1})
        y = y.dropna()
        y_sell = y_sell.dropna(how='any')
        dataset = dataset.drop(['future_return'], axis=1)

        

        dataset['future_return_AAPL'] = dataset['open_AAPL'].pct_change(future_period).shift(-future_period)
        y_AAPL = np.sign(dataset['future_return_AAPL'])
        y_AAPL = y_AAPL.replace({-1:0})
        y_AAPL = y_AAPL.dropna()

        y_AAPL_sell = np.sign(dataset['future_return_AAPL'])

        y_AAPL_sell = y_AAPL_sell.replace({1:0})
        y_AAPL_sell = y_AAPL_sell.replace({-1:1})

        y_AAPL_sell = y_AAPL_sell.dropna(how='any')

        dataset = dataset.drop(['future_return_AAPL'], axis=1)

        dataset['future_return_TSLA'] = dataset['open_TSLA'].pct_change(future_period).shift(-future_period)
        y_TSLA = np.sign(dataset['future_return_TSLA'])
        y_TSLA = y_TSLA.replace({-1:0})
        y_TSLA = y_TSLA.dropna()

        y_TSLA_sell = np.sign(dataset['future_return_TSLA'])

        y_TSLA_sell = y_TSLA_sell.replace({1:0})
        y_TSLA_sell = y_TSLA_sell.replace({-1:1})

        y_TSLA_sell = y_TSLA_sell.dropna(how='any')

        dataset = dataset.drop(['future_return_TSLA'], axis=1)

        if str(side) == 'OrderSide.BUY':
            side = OrderSide.BUY
            if symbol == 'MSFT':
                y = y
            if symbol == 'AAPL':
                y = y_AAPL
            if symbol == 'TSLA':
                y = y_TSLA

        if str(side) == 'OrderSide.SELL':
            side = OrderSide.SELL
            if symbol == 'MSFT':
                y = y_sell
            if symbol == 'AAPL':
                y = y_AAPL_sell
            if symbol == 'TSLA':
                y = y_TSLA_sell    

        for symbol in ['AAPL', 'TSLA']:

            dataset['spread' + '_' + str(symbol)] = dataset['open' + '_' + str(symbol)] - ((dataset['low' + '_' + str(symbol)] + dataset['high' + '_' + str(symbol)])/2)
            dataset['spread2' + '_' + str(symbol)] = dataset['high' + '_' + str(symbol)] - dataset['low' + '_' + str(symbol)]
            dataset['Volatility' + '_' + str(symbol)] = (np.log(dataset['open' + '_' + str(symbol)]).ewm(span=5).std() * np.sqrt(5))
            dataset['Volatility2' + '_' + str(symbol)] = (np.log(dataset['Volatility' + '_' + str(symbol)]).ewm(span=5).std() * np.sqrt(5))
            dataset['last_return1'] = np.log(dataset['open' + '_' + str(symbol)]).pct_change()
            dataset['std_normalized1'] = np.log(dataset['open' + '_' + str(symbol)]).rolling(std_period).apply(std_normalized)
            dataset['ma_ratio1'] = np.log(dataset['open' + '_' + str(symbol)]).rolling(ma_period).apply(ma_ratio)
            dataset['price_deviation1'] = np.log(dataset['open' + '_' + str(symbol)]).rolling(price_deviation_period).apply(values_deviation)
            #dataset['volume_deviation1'] = np.log(dataset['volume1']).rolling(volume_deviation_period).apply(values_deviation)
            dataset['OBV1'] = stats.zscore((np.sign(dataset['open' + '_' + str(symbol)].diff()) * dataset['volume' + '_' + str(symbol)]).fillna(0).cumsum())
            dataset['ratio'] = (dataset["open" + '_' + str(symbol)]) / (dataset["open"])
            dataset['ratio_reversed'] = (dataset["open"]) / (dataset["open" + '_' + str(symbol)])
            dataset['ratio_volu'] = dataset["open"].pct_change() / dataset["volume"]
            dataset['difference'] = (dataset["open" + '_' + str(symbol)]) - (dataset["open"])
            dataset['difference_reversed'] = (dataset["open"]) - (dataset["open" + '_' + str(symbol)])

            #dataset['ratio_v'] = (dataset["Volatility"]) / (dataset["Volatility" + '_' + str(symbol)])
            #dataset['ratio_reversed_v'] = (dataset["Volatility" + '_' + str(symbol)]) / (dataset["Volatility"])

            #dataset['difference_v'] = (dataset["Volatility"]) - (dataset["Volatility" + '_' + str(symbol)])
            #dataset['difference_reversed_v'] = (dataset["Volatility" + '_' + str(symbol)]) - (dataset["Volatility"])


        dataset['spread'] = dataset['open'] - ((dataset['low'] + dataset['high'])/2)
        dataset['spread2'] = dataset['high'] - dataset['low']
        dataset['Volatility'] = (np.log(dataset['open']).ewm(span=5).std() * np.sqrt(5))
        dataset['Volatility2'] = (np.log(dataset['Volatility']).ewm(span=5).std() * np.sqrt(5))
        #

        
        dataset['last_return'] = np.log(dataset["open"]).pct_change()
        dataset['std_normalized'] = np.log(dataset[column_price]).rolling(std_period).apply(std_normalized)
        dataset['ma_ratio'] = np.log(dataset[column_price]).rolling(ma_period).apply(ma_ratio)
        dataset['price_deviation'] = np.log(dataset[column_price]).rolling(price_deviation_period).apply(values_deviation)
        #dataset['volume_deviation'] = np.log(dataset[column_volume]).rolling(volume_deviation_period).apply(values_deviation)
        dataset['OBV'] = stats.zscore((np.sign(dataset["open"].diff()) * dataset['volume']).fillna(0).cumsum())

        

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0)
        sos = butter(4, 0.125, output='sos')

        for i in dataset.columns.tolist():
            detrend(dataset[i], overwrite_data=True)

        for i in dataset.columns.tolist():
            #dataset[str(i)+'_sosfiltfilt'] = sosfiltfilt(sos, dataset[i])
            dataset[str(i)+'_savgol'] = savgol_filter(dataset[i], 5, 3)
            #dataset[str(i)+'_smooth_5'] = dataset[i].ewm(span=5).mean()
            #dataset[str(i)+'_smooth_10'] = dataset[i].ewm(span=10).mean()
            #dataset[str(i)+'_smooth_20'] = dataset[i].ewm(span=20).mean()

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0)
        

        #print('dataset: \n', dataset)
            
        dataset = z_score_df(dataset)
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        
        dataset = dataset.fillna(0)
        #print('\n after z_score percent dataset: \n', dataset)
        #print('\n after z_score percent dataset: \n', dataset.describe())
        index1 = dataset.index
        columns_list = dataset.columns.tolist()
        dataset = winsorize(dataset.values, limits=[0.02, 0.02], inplace=True, nan_policy='propagate')
        dataset = pd.DataFrame(dataset, columns=columns_list, index=index1)
        #print('\n after winsorize dataset: \n', dataset)
        print('\n after winsorize dataset: \n', dataset.describe())
        #dataset = dataset.dropna(how="all", axis=1)
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        #dataset = dataset.dropna(how='any')
        dataset = dataset.fillna(0)


        last_input = dataset[-2:]
        dataset_AAPL = dataset
        last_input_AAPL = dataset_AAPL[-2:]
        
        #print('\n last input: \n', last_input)


        



        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0)

        

        y = y.dropna()

        #dataset = dataset.apply(pd.to_numeric, downcast='float')
        #dataset = dataset.apply(pd.to_numeric, downcast='integer')



            

        
        
        dataset = dataset[dataset.index.isin(y.index)]
        #dataset_AAPL = dataset_AAPL[dataset_AAPL.index.isin(y_AAPL.index)]


        
        #print('\n dataset: \n', dataset)
        #print('\n y: \n', y)

        #print('\n last dataset input: \n', dataset[-1:])
        #print('\n last y input: \n', y[-1:])

        


        X_train, X_test, y_train, y_test = train_test_split(dataset[-(len(y)):], y, test_size = 0.5, random_state = 42, shuffle=False)
        X_valid, X_test2, y_valid, y_test2 = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42, shuffle=False)


        
        
        train_dataset = cb.Pool(X_train, y_train)
        test_dataset = cb.Pool(X_test, y_test)
        valid_dataset = cb.Pool(X_valid, y_valid)
        
        catboost_class = CatBoostClassifier(iterations=300, early_stopping_rounds=10, silent=True, thread_count=-1)
        """
        my_file = Path(f'model_{symbol}_{side}') # file path for persistant model
        if my_file.exists():
            catboost_class = CatBoostClassifier()      # parameters not required.
            catboost_class.load_model(f'model_{symbol}_{side}')
            """
        selected_features = catboost_class.select_features(train_dataset, eval_set=test_dataset, features_for_select=list(dataset.columns), num_features_to_select=10, steps=5, algorithm='RecursiveByShapValues', train_final_model=True, verbose=False)
        print('\n selected_features: \n', selected_features['selected_features_names'])
        #catboost_class.select_features(train_dataset, eval_set=test_dataset, num_features_to_select=50, steps=10, algorithm='RecursiveByShapValues', train_final_model=True,)




        grid = {

            'max_depth': randint(2,7),
            'learning_rate': np.linspace(0.001, 1, 50),
            #'iterations': np.arange(100, 1000, 100),
            'l2_leaf_reg': np.linspace(0.1, 20, 50),
            'random_strength': np.linspace(0.1, 20, 50),
            'subsample': np.linspace(0.75, 1, 5),
            'bagging_temperature': np.linspace(0.1, 20, 50),
            'early_stopping_rounds': np.linspace(1, 20, 20),
            'diffusion_temperature':np.linspace(1, 20000, 200),
            'fold_len_multiplier':np.linspace(2, 10, 50),
            #'boosting_type': ['Ordered','Plain'],
            #'thread_count':[-1,-1],
            'loss_function': ['Logloss','CrossEntropy'],
            'eval_metric': ['AUC','Accuracy', 'Precision', 'Recall', 'F1', 'BalancedAccuracy', 'TotalF1', 'BalancedErrorRate', 'PRAUC','LogLikelihoodOfPrediction'  ],

            
        }
        tscv = TimeSeriesSplit(n_splits=5, gap=1)
        rscv = HalvingRandomSearchCV(catboost_class, grid, resource='iterations', n_candidates='exhaust', aggressive_elimination=True, factor=4, min_resources=1, max_resources=20, cv=tscv, verbose=1, scoring='f1_weighted')

        rscv.fit(dataset[-(len(y)):], y)

        best_params = rscv.best_params_
        print("\n best params: \n", rscv.best_params_, rscv.best_score_)

        catboost_class = rscv.best_estimator_
        metric(y_test, catboost_class.predict(X_test))


        model = catboost_class

        #model.save_model(f'model_{symbol}_{side}')


        
        y_pred_test = catboost_class.predict(last_input)

        CatBoost_pred = int(y_pred_test[-1:])
        #previous_CatBoost = int(y_pred_test[-2:-1])
        print("last CatBoost_buy output: ", CatBoost_pred)

        




        if int(CatBoost_pred) == 1:



                limit_order(symbol=symbol, 
                            spread=0, 
                            side=side, 
                            take_profit_multiplier = 2,
                            loss_stop_multiplier = 2,
                            loss_limit_multiplier = 2.1,
                            qty = 100,
                            inventory_risk = get_inventory_risk(symbol=symbol)
                            )
                

        match_orders_for_symbol(symbol=symbol)
        t1 = time.time()
        total = t1-t0
        print('\n Total time to order: \n', total)









from alpaca.data.live import StockDataStream, CryptoDataStream

API_KEY = "PKBX5XZQ1JG2CEODIOKD"
SECRET_KEY = "laKd5n4c7pnjRT9nC6WJztVEWruDz2b1VDJab5Hg"
wss_client = StockDataStream(API_KEY, SECRET_KEY)

now = datetime.now()
ask_price_list = pd.DataFrame()
ask_price_list5 = pd.DataFrame()
ask_price_list_TSLA5 = pd.DataFrame()
# async handler
async def trade_data_handler(data):
    # quote data will arrive here
    
    #print('\n Raw Data: \n', data)
    df = pd.DataFrame(data)

    symbol = df[1][0]


    if symbol == "MSFT":
        timestamp = df[1][1]
        ask_price = df[1][3]
        volume = df[1][4]
        global ask_price_list
        global ask_price_list3
        best_bid_MSFT, best_ask_MSFT, midpoint_MSFT, df1, inventory_qty_MSFT = get_orderbook("MSFT")

        d = {'close':[ask_price],'volume':[volume], 'Open':float(df1['Open'][-1]), 'High':[df1['High'][-1]], 'Low':[df1['Low'][-1]], 'Close':[df1['Close'][-1]], 
                'Volume':[df1['Volume'][-1]],
                'mu_MSFT':[df1['mu'][-1]], 'gamma_MSFT':[df1['gamma'][-1]], 'sigma_MSFT':[df1['sigma'][-1]], 'k_MSFT':[df1['k'][-1]],
                'bid_alpha_MSFT':[df1['bid_alpha'][-1]], 'ask_alpha_MSFT':[df1['ask_alpha'][-1]], 'ask_sum_delta_vol_MSFT':[df1['ask_sum_delta_vol'][-1]], 
                'bid_sum_delta_vol_MSFT':[df1['bid_sum_delta_vol'][-1]], 
                'bid_spread_aysm_MSFT':[df1['bid_spread_aysm'][-1]], 
                'ask_spread_aysm_MSFT':[df1['ask_spread_aysm'][-1]], 
                'inventory_MSFT':[df1['inventory'][-1]],
                'bid_spread_aysm2_MSFT':[df1['bid_spread_aysm2'][-1]], 
                'ask_spread_aysm2_MSFT':[df1['ask_spread_aysm2'][-1]],
                'midpoint_MSFT':[midpoint_MSFT],
                'inventory_qty_MSFT':[inventory_qty_MSFT], 
                'best_bid_MSFT':[best_bid_MSFT], 'best_ask_MSFT':[best_ask_MSFT],
                }
        
        row = pd.DataFrame(d, index = [timestamp])
        #print('\n row: \n', row)
        
        ask_price_list = pd.concat([ask_price_list, row])
        volume = ask_price_list['volume'].resample('10S').sum()

        ask_price_list3 = ask_price_list['close'].resample('10S').ohlc()
        ask_price_list3 = pd.merge(left=ask_price_list3, right=volume, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
        #ask_price_list3.drop(ask_price_list3.filter(regex='_y$').columns, axis=1, inplace=True)
        for i in ['Open', 'High','Low','Close', 'Volume', 'best_bid_MSFT', 'best_ask_MSFT', 'midpoint_MSFT', 'inventory_qty_MSFT' ,'mu_MSFT','gamma_MSFT','sigma_MSFT','k_MSFT', 'bid_alpha_MSFT', 'ask_alpha_MSFT', 'ask_sum_delta_vol_MSFT', 'bid_sum_delta_vol_MSFT', 'bid_spread_aysm_MSFT', 'ask_spread_aysm_MSFT', 'bid_spread_aysm2_MSFT', 'ask_spread_aysm2_MSFT',]:
            ask_price_list_temp = ask_price_list[i].resample('10S').mean()
            ask_price_list3 = pd.merge(left=ask_price_list3, right=ask_price_list_temp, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
            #ask_price_list3.drop(ask_price_list3.filter(regex='_y$').columns, axis=1, inplace=True)

        ask_price_list3 = ask_price_list3.ffill()


        #print('\n ask_price_list_MSFT: \n', ask_price_list3)
        return ask_price_list3


    


    if symbol == 'AAPL':
            
        timestamp2 = df[1][1]
        ask_price2 = df[1][3]
        volume2 = df[1][4]
        global ask_price_list5
        global ask_price_list2

        best_bid_AAPL, best_ask_AAPL, midpoint_AAPL, df2, inventory_qty_AAPL = get_orderbook("AAPL")

        d2 = {'close':[ask_price2],'volume':[volume2], 'Open_AAPL':[df2['Open'][-1]], 'High_AAPL':[df2['High'][-1]], 'Low_AAPL':[df2['Low'][-1]],
                'Close_AAPL':[df2['Close'][-1]],'midpoint_AAPL':[midpoint_AAPL], 'inventory_qty_AAPL':[inventory_qty_AAPL], 
                'best_bid_AAPL':[best_bid_AAPL], 'best_ask_AAPL':[best_ask_AAPL],
                'mu_AAPL':[df2['mu'][-1]], 'gamma_AAPL':[df2['gamma'][-1]], 'sigma_AAPL':[df2['sigma'][-1]], 'k_AAPL':[df2['k'][-1]],
                'bid_alpha_AAPL':[df2['bid_alpha'][-1]], 'ask_alpha_AAPL':[df2['ask_alpha'][-1]], 'ask_sum_delta_vol_AAPL':[df2['ask_sum_delta_vol'][-1]], 
                'bid_sum_delta_vol_AAPL':[df2['bid_sum_delta_vol'][-1]], 
                'bid_spread_aysm_AAPL':[df2['bid_spread_aysm'][-1]], 
                'ask_spread_aysm_AAPL':[df2['ask_spread_aysm'][-1]], 
                'inventory_AAPL':[df2['inventory'][-1]],
                'bid_spread_aysm2_AAPL':[df2['bid_spread_aysm2'][-1]], 
                'ask_spread_aysm2_AAPL':[df2['ask_spread_aysm2'][-1]], }
        
        row2 = pd.DataFrame(d2, index = [timestamp2])
                
        ask_price_list5 = pd.concat([ask_price_list5, row2])
        volume2 = ask_price_list5['volume'].resample('10S').sum()


        ask_price_list2 = ask_price_list5['close'].resample('10S').ohlc()
        ask_price_list2 = pd.merge(left=ask_price_list2, right=volume2, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
        #ask_price_list2.drop(ask_price_list2.filter(regex='_y$').columns, axis=1, inplace=True)

        for i in ['Open_AAPL', 'High_AAPL','Low_AAPL','Close_AAPL', 'midpoint_AAPL', 'inventory_qty_AAPL', 'best_bid_AAPL', 'best_ask_AAPL','mu_AAPL','gamma_AAPL','sigma_AAPL','k_AAPL', 'bid_alpha_AAPL', 'ask_alpha_AAPL', 'ask_sum_delta_vol_AAPL', 'bid_sum_delta_vol_AAPL', 'bid_spread_aysm_AAPL', 'ask_spread_aysm_AAPL', 'bid_spread_aysm2_AAPL', 'ask_spread_aysm2_AAPL',]:
            ask_price_list_temp = ask_price_list5[i].resample('10S').mean()
            ask_price_list2 = pd.merge(left=ask_price_list2, right=ask_price_list_temp, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
            #ask_price_list2.drop(ask_price_list2.filter(regex='_y$').columns, axis=1, inplace=True)
        
        ask_price_list2 = ask_price_list2.ffill()
        ask_price_list2 = ask_price_list2.rename(columns={"open":"open_AAPL", "high":"high_AAPL", "low":"low_AAPL", "close":"close_AAPL", "volume":"volume_AAPL"})

        #print('\n ask_price_list_AAPL: \n', ask_price_list2)
        return ask_price_list2

    
    if symbol == 'TSLA':
            
        timestamp3 = df[1][1]
        ask_price3 = df[1][3]
        volume3 = df[1][4]
        global ask_price_list_TSLA5
        global ask_price_list_TSLA2

        best_bid_TSLA, best_ask_TSLA, midpoint_TSLA, df3, inventory_qty_TSLA = get_orderbook("TSLA")

        d2_TSLA = {'close':[ask_price3],'volume':[volume3], 'Open_TSLA':[df3['Open'][-1]], 'High_TSLA':[df3['High'][-1]], 'Low_TSLA':[df3['Low'][-1]],
                        'Close_TSLA':[df3['Close'][-1]],'midpoint_TSLA':[midpoint_TSLA], 'inventory_qty_TSLA':[inventory_qty_TSLA], 'best_bid_TSLA':[best_bid_TSLA], 
                        'best_ask_TSLA':[best_ask_TSLA],
                        'mu_TSLA':[df3['mu'][-1]], 'gamma_TSLA':[df3['gamma'][-1]], 'sigma_TSLA':[df3['sigma'][-1]], 'k_TSLA':[df3['k'][-1]],
                        'bid_alpha_TSLA':[df3['bid_alpha'][-1]], 'ask_alpha_TSLA':[df3['ask_alpha'][-1]], 'ask_sum_delta_vol_TSLA':[df3['ask_sum_delta_vol'][-1]], 
                        'bid_sum_delta_vol_TSLA':[df3['bid_sum_delta_vol'][-1]], 
                        'bid_spread_aysm_TSLA':[df3['bid_spread_aysm'][-1]], 
                        'ask_spread_aysm_TSLA':[df3['ask_spread_aysm'][-1]], 
                        'bid_spread_aysm2_TSLA':[df3['bid_spread_aysm2'][-1]], 
                        'ask_spread_aysm2_TSLA':[df3['ask_spread_aysm2'][-1]], }
        
        row2_TSLA = pd.DataFrame(d2_TSLA, index = [timestamp3])
                
        ask_price_list_TSLA5 = pd.concat([ask_price_list_TSLA5, row2_TSLA])
        volume_TSLA2 = ask_price_list_TSLA5['volume'].resample('10S').sum()


        ask_price_list_TSLA2 = ask_price_list_TSLA5['close'].resample('10S').ohlc()
        ask_price_list_TSLA2 = pd.merge(left=ask_price_list_TSLA2, right=volume_TSLA2, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
        #ask_price_list_TSLA2.drop(ask_price_list_TSLA2.filter(regex='_y$').columns, axis=1, inplace=True)

        for i in ['Open_TSLA', 'High_TSLA','Low_TSLA','Close_TSLA','best_bid_TSLA', 'best_ask_TSLA', 'midpoint_TSLA', 'inventory_qty_TSLA', 'best_bid_TSLA', 'best_ask_TSLA','mu_TSLA','gamma_TSLA','sigma_TSLA','k_TSLA', 'bid_alpha_TSLA', 'ask_alpha_TSLA', 'ask_sum_delta_vol_TSLA', 'bid_sum_delta_vol_TSLA', 'bid_spread_aysm_TSLA', 'ask_spread_aysm_TSLA', 'bid_spread_aysm2_TSLA', 'ask_spread_aysm2_TSLA',]:
            ask_price_list_temp_TSLA = ask_price_list_TSLA5[i].resample('10S').mean()
            ask_price_list_TSLA2 = pd.merge(left=ask_price_list_TSLA2, right=ask_price_list_temp_TSLA, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
            #ask_price_list_TSLA2.drop(ask_price_list_TSLA2.filter(regex='_y$').columns, axis=1, inplace=True)
        
        ask_price_list_TSLA2 = ask_price_list_TSLA2.ffill()
        ask_price_list_TSLA2 = ask_price_list_TSLA2.rename(columns={"open":"open_TSLA", "high":"high_TSLA", "low":"low_TSLA", "close":"close_TSLA", "volume":"volume_TSLA"})

        #print('\n ask_price_list_TSLA2: \n', ask_price_list_TSLA2)
        return ask_price_list_TSLA2
    
    

        
        

    """ask_price_list4 = pd.merge(left=ask_price_list3, right=ask_price_list2, left_index=True, right_index=True)
    ask_price_list4 = pd.merge(left=ask_price_list4, right=ask_price_list_TSLA2, left_index=True, right_index=True)
    
    ask_price_list4 = ask_price_list4.ffill()
    print('\n latest merged df: \n', ask_price_list4)
    print('\n latest data recieved: \n', ask_price_list4[-1:])

    return ask_price_list4"""








            
data_out = pd.DataFrame()

async def create_model(data):

    t = time.process_time()

    global data_out

    data_in = await trade_data_handler(data)

    #data_out = data_in.join(data_out)
    #data_out = data_out.append(data_in[~data_in.index.isin(data_out.index)])
    #print('\n data_in: \n', data_in)
    #data_out = pd.concat([data_out, data_in], axis=1)
    #print('\n data_out concat: \n', data_out)
    data_out = pd.merge(left=data_in, right=data_out, left_index=True, right_index=True, how='outer', suffixes=('', '_y'))
    #print('\n data_out merge: \n', data_out)
    data_out.drop(data_out.filter(regex='_y$').columns, axis=1, inplace=True)

    #if data_in[-1:].index > data_out[-1:].index:
        #data_out = pd.concat([data_out, data_in[-1:]])

    #data_out = data_out.drop_duplicates()
    

    

    #data_out2 = pd.merge(left=data_in, right=data_out, left_index=True, right_index=True,how='left')

    
    data_out = data_out.ffill()
    data_out = data_out.fillna(0)
    #print('\n latest merged df: \n', data_out)
    #print('\n latest data recieved: \n', data_out[-1:])

    elapsed_time = time.process_time() - t
    print('\n Time to fetch data: \n', elapsed_time)

    dataset = data_out
    symbol_list = ['MSFT', 'MSFT', 'AAPL', 'AAPL', 'TSLA', 'TSLA']
    side_list = ['OrderSide.BUY', 'OrderSide.SELL', 'OrderSide.BUY', 'OrderSide.SELL', 'OrderSide.BUY', 'OrderSide.SELL',]
    x_list = [dataset, dataset, dataset, dataset, dataset, dataset]
    #y_list = [y, y_sell, y, y_sell]



    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(make_model, x_list, symbol_list, side_list)

    now1 = datetime.now()
    print('\n ------- Current Local Machine Time ------- \n', now1)
    if int(now1.hour) == 21:
            if int(now1.minute) == 56:
                dataset.to_csv(f'/home/vboxuser/Documents/dataset_{now1}.csv')
                dataset.to_csv(f'/home/vboxuser/Documents/dataset_AAPL_{now1}.csv')
                y = pd.DataFrame(y)
                y.to_csv(f'/home/vboxuser/Documents/y_{now1}.csv')
                y_sell = pd.DataFrame(y_sell)
                y_sell.to_csv(f'/home/vboxuser/Documents/y_sell_{now1}.csv')
                y_AAPL = pd.DataFrame(y_AAPL)
                y_AAPL.to_csv(f'/home/vboxuser/Documents/y_AAPL_{now1}.csv')
                y_AAPL_sell = pd.DataFrame(y_AAPL_sell)
                y_AAPL_sell.to_csv(f'/home/vboxuser/Documents/y_AAPL_sell_{now1}.csv')


wss_client.subscribe_trades(create_model, "MSFT", "AAPL", "TSLA")

wss_client.run()





"""
async def g():
    # Pause here and come back to g() when f() is ready
    r = await f()
    return r

"""
