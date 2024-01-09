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



symbol = "NVDA"
BASE_URL = "https://paper-api.alpaca.markets"
API_KEY = "PKBX5XZQ1JG2CEODIOKD"
SECRET_KEY = "laKd5n4c7pnjRT9nC6WJztVEWruDz2b1VDJab5Hg"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

midpoint_SPY = 0
midpoint_NVDA = 0
midpoint_AMD = 0
midpoint = 0
ask_alpha = 0.01
bid_alpha = 0.01
inventory_qty = 1
best_bid = 0.02
best_ask = 0.02
bid_sum_delta_vol = 10000
ask_sum_delta_vol = 10000
previous_df = 0
df_orderbook = 0


def get_orderbook(symbol):

    t = time.process_time()

    
    take_profit_method(symbol)
    
    
    global midpoint_SPY
    global midpoint_NVDA
    global midpoint_AMD
    global previous_df
    global df_orderbook

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
        
        
        


        

    symbol = str(symbol)
    previous_df = df_orderbook
    df_orderbook = previous_df
    try:
        df_orderbook = pd.DataFrame(yf.download(symbol, period="1d", interval="1m"))
    except:
        df_orderbook = previous_df

    df_orderbook['inventory'] = inventory_qty
    df_orderbook["mu"] = abs((np.log(df_orderbook["Open"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})).pct_change()/2) * 10000)

    df_orderbook['gamma'] = get_inventory_risk(symbol = symbol)


    df_orderbook['sigma'] = ((np.log(df_orderbook["Open"]).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5)).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})) * 100


    df_orderbook['Volume'] = df_orderbook['Volume'] + 1


    df_orderbook['k'] = (0.5*(df_orderbook['sigma'])*np.sqrt(df_orderbook['Volume']/df_orderbook['Volume'].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})))*1

    df_orderbook['bid_alpha'] = bid_alpha
    df_orderbook['ask_alpha'] = ask_alpha

    df_orderbook['bid_sum_delta_vol'] = bid_sum_delta_vol
    df_orderbook['ask_sum_delta_vol'] = ask_sum_delta_vol
    

    df_orderbook['bid_spread_aysm'] = ((1 / df_orderbook['gamma'] * np.log(1 + df_orderbook['gamma'] / df_orderbook['k']) + (2 * df_orderbook['inventory'] + 1) / 2 * np.sqrt((df_orderbook['sigma']**2 * df_orderbook['gamma']) / (2 * df_orderbook['k'] * df_orderbook['bid_alpha']) * (1 + df_orderbook['gamma'] / df_orderbook['k'])**(1 + df_orderbook['k'] / df_orderbook['gamma']))) / 100000)

    df_orderbook['ask_spread_aysm'] = ((1 / df_orderbook['gamma'] * np.log(1 + df_orderbook['gamma'] / df_orderbook['k']) - (2 * df_orderbook['inventory'] - 1) / 2 * np.sqrt((df_orderbook['sigma']**2 * df_orderbook['gamma']) / (2 * df_orderbook['k'] * df_orderbook['ask_alpha']) * (1 + df_orderbook['gamma'] / df_orderbook['k'])**(1 + df_orderbook['k'] / df_orderbook['gamma']))) / 100000)
    


    
    df_orderbook['bid_spread_aysm2'] = ((1 / df_orderbook['gamma'] * np.log(1 + df_orderbook['gamma'] / df_orderbook['k']) + (- df_orderbook["mu"] / (df_orderbook['gamma'] * df_orderbook['sigma']**2) + (2 * df_orderbook['inventory'] + 1) / 2) * np.sqrt((df_orderbook['sigma']**2 * df_orderbook['k']) / (2 * df_orderbook['k'] * df_orderbook['bid_alpha']) * (1 + df_orderbook['gamma'] / df_orderbook['k'])**(1 + df_orderbook['k'] / df_orderbook['gamma']))) / 19999998)

    df_orderbook['ask_spread_aysm2'] = ((1 / df_orderbook['gamma'] * np.log(1 + df_orderbook['gamma'] / df_orderbook['k']) + (  df_orderbook["mu"] / (df_orderbook['gamma'] * df_orderbook['sigma']**2) - (2 * df_orderbook['inventory'] - 1) / 2) * np.sqrt((df_orderbook['sigma']**2 * df_orderbook['k']) / (2 * df_orderbook['k'] * df_orderbook['ask_alpha']) * (1 + df_orderbook['gamma'] / df_orderbook['k'])**(1 + df_orderbook['k'] / df_orderbook['gamma']))) / 19999998)


    print("\n bid: \n", symbol, df_orderbook['bid_spread_aysm2'][-1])
    print("\n ask: \n", symbol, df_orderbook['ask_spread_aysm2'][-1])

    best_ask = df_orderbook['ask_spread_aysm2'][-1]
    best_bid = df_orderbook['bid_spread_aysm2'][-1]

    if symbol == 'NVDA':
        midpoint_NVDA = midpoint
        df_orderbook['midpoint_NVDA'] = midpoint

    if symbol == 'SPY':
        midpoint_SPY = midpoint
        df_orderbook['midpoint_SPY'] = midpoint

    if symbol == 'AMD':
        midpoint_AMD = midpoint
        df_orderbook['midpoint_AMD'] = midpoint

    elapsed_time = time.process_time() - t
    print('\n Time to ordebook method: \n', elapsed_time)

    return best_bid, best_ask, midpoint, df_orderbook, inventory_qty






def get_time_til_close(symbol):

    try:
        symbol = symbol
        now = datetime.now()
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        

        if int(now.hour) > 21:
            if int(now.minute) > 57:
                cancel_orders_for_symbol(symbol=symbol)
                trading_client.close_position(symbol)
                
    except:
        print("get_time_til_close exception. It's not trading time...")
        #print(traceback.format_exc())
        
    
def get_inventory_risk(symbol):
    inventory_risk = 0.00002
    try:
        symbol = str(symbol)

        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    
        position = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(position)
        inventory_qty = int(ORDERS[1][6])

        if symbol == "NVDA":
            inventory_risk = 0.000002 * abs(inventory_qty)

        if symbol == 'SPY':
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

        if float(ORDERS[1][10]) / abs(float(ORDERS[1][6])) >=  0.06:
            cancel_orders_for_symbol(symbol=symbol)
            
            trading_client.close_position(symbol)

        if float(ORDERS[1][10]) / abs(float(ORDERS[1][6])) <=  -0.3:
            cancel_orders_for_symbol(symbol=symbol)
            
            trading_client.close_position(symbol)
        
        if float(ORDERS[1][12]) >=  6:
            
            cancel_orders_for_symbol(symbol=symbol)
            
            trading_client.close_position(symbol)
            

        elif float(ORDERS[1][12]) <=  -2:

            cancel_orders_for_symbol(symbol=symbol)
            
            trading_client.close_position(symbol)
            

        elif str(ORDERS[1][7]) ==  "PositionSide.LONG":

            if float(ORDERS[1][5]) - float(ORDERS[1][14]) <= -0.1:
                cancel_orders_for_symbol(symbol=symbol)
                
                trading_client.close_position(symbol)
                

            if float(ORDERS[1][5]) - float(ORDERS[1][14]) >= 0.1:
                cancel_orders_for_symbol(symbol=symbol)
                
                trading_client.close_position(symbol)
                


        elif str(ORDERS[1][7]) ==  "PositionSide.SHORT":

            if float(ORDERS[1][5]) - float(ORDERS[1][14]) >= 0.1:
                cancel_orders_for_symbol(symbol=symbol)
                
                trading_client.close_position(symbol)
                

            if float(ORDERS[1][5]) - float(ORDERS[1][14]) <= -0.1:
                cancel_orders_for_symbol(symbol=symbol)
                
                trading_client.close_position(symbol)
                

    except:
        print ("take_profit_method error.")
        #print(traceback.format_exc())
        






@jit
def limit_order(symbol, spread, side, take_profit_multiplier, loss_stop_multiplier, loss_limit_multiplier, qty, inventory_risk):
    
    symbol = str(symbol)
    best_bid, best_ask, midpoint, df, inventory_qty = get_orderbook(symbol = symbol)

    dataset = df


    dataset['spread'] = abs(np.log(dataset['Open']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) - ((np.log(dataset['Low']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) + np.log(dataset['High']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}))/2))
    dataset['variance'] = (np.log(dataset['Open']).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5)).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
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

    if symbol == 'NVDA':
        midpoint = midpoint_NVDA

    if symbol == 'SPY':
        midpoint = midpoint_SPY

    if symbol == 'AMD':
        midpoint = midpoint_AMD

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


@jit(cache=True, nopython=True)
def std_normalized(vals):
    return np.std(vals) / np.mean(vals)

# Ratio of diff between last price and mean value to last price
@jit(cache=True, nopython=True)
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]

# z-score for volumes and price
@jit(cache=True, nopython=True)
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)

@jit(cache=True)
def z_score(vals):
    vals = np.log(vals)
    vals = ((vals - vals.expanding().mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}))/vals.expanding().std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})).pct_change()
    return vals


def z_score_df(df):
    df = df.apply(lambda x : z_score(x))
    return df


def make_model(dataset, symbol, side):
    try: 
        t0 = time.time()

        symbol = str(symbol)
        get_time_til_close(symbol=symbol)
        take_profit_method(symbol=symbol)



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

        """

        dataset['future_return_SPY'] = dataset['open_SPY'].pct_change(future_period).shift(-future_period)
        y_SPY = np.sign(dataset['future_return_SPY'])
        y_SPY = y_SPY.replace({-1:0})
        y_SPY = y_SPY.dropna()

        y_SPY_sell = np.sign(dataset['future_return_SPY'])

        y_SPY_sell = y_SPY_sell.replace({1:0})
        y_SPY_sell = y_SPY_sell.replace({-1:1})

        y_SPY_sell = y_SPY_sell.dropna(how='any')

        dataset = dataset.drop(['future_return_SPY'], axis=1)
        """
 

        if str(side) == 'OrderSide.BUY':
            side = OrderSide.BUY
            if symbol == 'NVDA':
                y = y
            #if symbol == 'SPY':
                #y = y_SPY


        if str(side) == 'OrderSide.SELL':
            side = OrderSide.SELL
            if symbol == 'NVDA':
                y = y_sell
            #if symbol == 'SPY':
                #y = y_SPY_sell


        """for symbol in ['SPY','NVDA']:

            dataset['spread' + '_' + str(symbol)] = dataset['Open' + '_' + str(symbol)] - ((dataset['Low' + '_' + str(symbol)] + dataset['High' + '_' + str(symbol)])/2)
            dataset['spread2' + '_' + str(symbol)] = dataset['High' + '_' + str(symbol)] - dataset['Low' + '_' + str(symbol)]
            dataset['Volatility' + '_' + str(symbol)] = (np.log(dataset['Open' + '_' + str(symbol)]).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5))
            dataset['Volatility2' + '_' + str(symbol)] = (np.log(dataset['Volatility' + '_' + str(symbol)]).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5))
            dataset['last_return1'+ '_' + str(symbol)] = np.log(dataset['Open' + '_' + str(symbol)]).pct_change()
            dataset['std_normalized1'+ '_' + str(symbol)] = np.log(dataset['Open' + '_' + str(symbol)]).rolling(std_period).apply(std_normalized, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
            dataset['ma_ratio1'+ '_' + str(symbol)] = np.log(dataset['Open' + '_' + str(symbol)]).rolling(ma_period).apply(ma_ratio, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
            #dataset['price_deviation1'+ '_' + str(symbol)] = np.log(dataset['open' + '_' + str(symbol)]).rolling(price_deviation_period).apply(values_deviation, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
            #dataset['volume_deviation1'] = np.log(dataset['volume1']).rolling(volume_deviation_period).apply(values_deviation)
            dataset['OBV1'+ '_' + str(symbol)] = stats.zscore((np.sign(dataset['Open' + '_' + str(symbol)].diff()) * dataset['Volume' + '_' + str(symbol)]).fillna(0.0000001).cumsum())
            dataset['ratio'+ '_' + str(symbol)] = (dataset["Open" + '_' + str(symbol)]) / (dataset["open"])
            dataset['ratio_reversed'+ '_' + str(symbol)] = (dataset["open"]) / (dataset["Open" + '_' + str(symbol)])
            dataset['ratio_volu'+ '_' + str(symbol)] = dataset["Open"+ '_' + str(symbol)].pct_change() / dataset["Volume"+ '_' + str(symbol)]
            dataset['difference'+ '_' + str(symbol)] = (dataset["Open" + '_' + str(symbol)]) - (dataset["open"])
            dataset['difference_reversed'+ '_' + str(symbol)] = (dataset["open"]) - (dataset["Open" + '_' + str(symbol)])

            #dataset['ratio_v'] = (dataset["Volatility"]) / (dataset["Volatility" + '_' + str(symbol)])
            #dataset['ratio_reversed_v'] = (dataset["Volatility" + '_' + str(symbol)]) / (dataset["Volatility"])

            #dataset['difference_v'] = (dataset["Volatility"]) - (dataset["Volatility" + '_' + str(symbol)])
            #dataset['difference_reversed_v'] = (dataset["Volatility" + '_' + str(symbol)]) - (dataset["Volatility"])
        """

        dataset['spread'] = dataset['open'] - ((dataset['low'] + dataset['high'])/2)
        dataset['spread2'] = dataset['high'] - dataset['low']
        dataset['Volatility'] = (np.log(dataset['open']).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5))
        dataset['Volatility2'] = (np.log(dataset['Volatility']).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5))
        #

        
        dataset['last_return'] = np.log(dataset["open"]).pct_change()
        dataset['std_normalized'] = np.log(dataset[column_price]).rolling(std_period).apply(std_normalized, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
        dataset['ma_ratio'] = np.log(dataset[column_price]).rolling(ma_period).apply(ma_ratio, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
        #dataset['price_deviation'] = np.log(dataset[column_price]).rolling(price_deviation_period).apply(values_deviation, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
        #dataset['volume_deviation'] = np.log(dataset[column_volume]).rolling(volume_deviation_period).apply(values_deviation)
        dataset['OBV'] = stats.zscore((np.sign(dataset["open"].diff()) * dataset['volume']).fillna(0.0000001).cumsum())

        

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)
        #sos = butter(4, 0.125, output='sos')

        for i in dataset.columns.tolist():
            detrend(dataset[i], overwrite_data=True)
        
        for i in dataset.columns.tolist():
            #dataset[str(i)+'_sosfiltfilt'] = sosfiltfilt(sos, dataset[i])
            #dataset[str(i)+'_savgol'] = savgol_filter(dataset[i], 5, 3)
            dataset[str(i)+'_smooth_5'] = dataset[i].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
            dataset[str(i)+'_smooth_10'] = dataset[i].rolling(10).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
            dataset[str(i)+'_smooth_20'] = dataset[i].rolling(20).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
        

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)
        

        #print('dataset: \n', dataset)
            
        dataset = z_score_df(dataset)
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        
        dataset = dataset.fillna(0.0000001)
        #print('\n after z_score percent dataset: \n', dataset)
        #print('\n after z_score percent dataset: \n', dataset.describe())
        index1 = dataset.index
        columns_list = dataset.columns.tolist()
        dataset = winsorize(dataset.values, limits=[0.05, 0.05], inplace=True, nan_policy='propagate')
        dataset = pd.DataFrame(dataset, columns=columns_list, index=index1)
        #print('\n after winsorize dataset: \n', dataset)
        print('\n after winsorize dataset: \n', dataset.describe())
        #dataset = dataset.dropna(how="all", axis=1)
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        #dataset = dataset.dropna(how='any')
        dataset = dataset.fillna(0.0000001)


        last_input = dataset[-2:]
        dataset_SPY = dataset
        last_input_SPY = dataset_SPY[-2:]
        
        #print('\n last input: \n', last_input)


        



        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)

        

        y = y.dropna()

        dataset = dataset.apply(pd.to_numeric, downcast='float')
        dataset = dataset.apply(pd.to_numeric, downcast='integer')



            

        
        
        dataset = dataset[dataset.index.isin(y.index)]
        #dataset_SPY = dataset_SPY[dataset_SPY.index.isin(y_SPY.index)]


        
        #print('\n dataset: \n', dataset)
        #print('\n y: \n', y)

        #print('\n last dataset input: \n', dataset[-1:])
        #print('\n last y input: \n', y[-1:])

        


        X_train, X_test, y_train, y_test = train_test_split(dataset[-(len(y)):], y, test_size = 0.5, random_state = 42, shuffle=False)
        X_valid, X_test2, y_valid, y_test2 = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42, shuffle=False)


        
        
        train_dataset = cb.Pool(X_train, y_train)
        test_dataset = cb.Pool(X_test2, y_test2)
        valid_dataset = cb.Pool(X_valid, y_valid)
        
        catboost_class = CatBoostClassifier(iterations=300, early_stopping_rounds=5, silent=True, thread_count=-1)
        """
        my_file = Path(f'model_{symbol}_{side}') # file path for persistant model
        if my_file.exists():
            catboost_class = CatBoostClassifier()      # parameters not required.
            catboost_class.load_model(f'model_{symbol}_{side}')
            """
        selected_features = catboost_class.select_features(train_dataset, eval_set=valid_dataset, features_for_select=list(dataset.columns), num_features_to_select=15, steps=10, algorithm='RecursiveByShapValues', shap_calc_type='Approximate', train_final_model=True, logging_level='Silent')
        print('\n selected_features: \n', selected_features['selected_features_names'])
        #catboost_class.select_features(train_dataset, eval_set=test_dataset, num_features_to_select=50, steps=10, algorithm='RecursiveByShapValues', train_final_model=True,)

        take_profit_method(symbol)


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
        rscv = HalvingRandomSearchCV(catboost_class, grid, resource='iterations', n_candidates='exhaust', aggressive_elimination=True, factor=4, min_resources=1, max_resources=30, cv=tscv, verbose=1, scoring='f1_weighted')

        rscv.fit(X_test2, y_test2)

        best_params = rscv.best_params_
        print("\n best params: \n", rscv.best_params_, rscv.best_score_)

        catboost_class = rscv.best_estimator_
        metric(y_test, catboost_class.predict(X_test))


        model = catboost_class

        #model.save_model(f'model_{symbol}_{side}')


        
        y_pred_test = catboost_class.predict(last_input)

        CatBoost_pred = int(y_pred_test[-1:])
        #previous_CatBoost = int(y_pred_test[-2:-1])
        print("last", str(symbol), str(side), "output: ", CatBoost_pred)

        




        if int(CatBoost_pred) == 1:

            if str(side) == 'OrderSide.BUY':
                spread = -0.02

                limit_order(symbol=symbol, 
                            spread=spread, 
                            side=side, 
                            take_profit_multiplier = 2,
                            loss_stop_multiplier = 2,
                            loss_limit_multiplier = 2.1,
                            qty = 100,
                            inventory_risk = get_inventory_risk(symbol=symbol)
                            )


            if str(side) == 'OrderSide.SELL':
                spread = 0.02

                limit_order(symbol=symbol, 
                            spread=spread, 
                            side=side, 
                            take_profit_multiplier = 2,
                            loss_stop_multiplier = 2,
                            loss_limit_multiplier = 2.1,
                            qty = 100,
                            inventory_risk = get_inventory_risk(symbol=symbol)
                            )
                
        take_profit_method(symbol)
        #match_orders_for_symbol(symbol=symbol)
        t1 = time.time()
        total = t1-t0
        print('\n Total time to order: \n', total)
    except:
        print("model error.")  
        print(traceback.format_exc())








from alpaca.data.live import StockDataStream, CryptoDataStream

API_KEY = "PKBX5XZQ1JG2CEODIOKD"
SECRET_KEY = "laKd5n4c7pnjRT9nC6WJztVEWruDz2b1VDJab5Hg"
wss_client = StockDataStream(API_KEY, SECRET_KEY)

now = datetime.now()
ask_price_list = pd.DataFrame()
ask_price_list5 = pd.DataFrame()
ask_price_list_AMD5 = pd.DataFrame()
# async handler
async def trade_data_handler(data):
    # quote data will arrive here
    t = time.process_time()

    
    take_profit_method(symbol='NVDA')
    #take_profit_method(symbol='SPY')
    #print('\n Raw Data: \n', data)
    df = pd.DataFrame(data)

    symbol = df[1][0]


    if symbol == "NVDA":
        timestamp = df[1][1]
        ask_price = df[1][3]
        volume = df[1][4]
        global ask_price_list
        global ask_price_list3
        best_bid_NVDA, best_ask_NVDA, midpoint_NVDA, df1, inventory_qty_NVDA = get_orderbook("NVDA")

        d = {'close':[ask_price],'volume':[volume], 'Open_NVDA':float(df1['Open'][-1]), 'High_NVDA':[df1['High'][-1]],
                'Low_NVDA':[df1['Low'][-1]], 'Close_NVDA':[df1['Close'][-1]], 
                'Volume_NVDA':[df1['Volume'][-1]],
                'mu_NVDA':[df1['mu'][-1]], 'gamma_NVDA':[df1['gamma'][-1]], 'sigma_NVDA':[df1['sigma'][-1]], 'k_NVDA':[df1['k'][-1]],
                'bid_alpha_NVDA':[df1['bid_alpha'][-1]], 'ask_alpha_NVDA':[df1['ask_alpha'][-1]],
                'ask_sum_delta_vol_NVDA':[df1['ask_sum_delta_vol'][-1]], 
                'bid_sum_delta_vol_NVDA':[df1['bid_sum_delta_vol'][-1]], 
                'inventory_NVDA':[df1['inventory'][-1]],
                'bid_spread_aysm2_NVDA':[df1['bid_spread_aysm2'][-1]], 
                'ask_spread_aysm2_NVDA':[df1['ask_spread_aysm2'][-1]],
                'midpoint_NVDA':[midpoint_NVDA],
                'best_bid_NVDA':[best_bid_NVDA], 'best_ask_NVDA':[best_ask_NVDA],
                }
        
        row = pd.DataFrame(d, index = [timestamp])
        #print('\n row: \n', row)
        
        ask_price_list = pd.concat([ask_price_list, row])
        volume = ask_price_list['volume'].resample('10S').sum()

        ask_price_list3 = ask_price_list['close'].resample('10S').ohlc()
        ask_price_list3 = pd.merge(left=ask_price_list3, right=volume, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
        #ask_price_list3.drop(ask_price_list3.filter(regex='_y$').columns, axis=1, inplace=True)
        for i in ['Open_NVDA', 'High_NVDA','Low_NVDA','Close_NVDA', 'Volume_NVDA', 'best_bid_NVDA', 'inventory_NVDA', 'best_ask_NVDA', 'midpoint_NVDA', 'mu_NVDA','gamma_NVDA','sigma_NVDA','k_NVDA', 'bid_alpha_NVDA', 'ask_alpha_NVDA', 'ask_sum_delta_vol_NVDA', 'bid_sum_delta_vol_NVDA', 'bid_spread_aysm2_NVDA', 'ask_spread_aysm2_NVDA',]:
            ask_price_list_temp = ask_price_list[i].resample('10S').mean()
            ask_price_list3 = pd.merge(left=ask_price_list3, right=ask_price_list_temp, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
            #ask_price_list3.drop(ask_price_list3.filter(regex='_y$').columns, axis=1, inplace=True)

        ask_price_list3 = ask_price_list3.ffill()


        #print('\n ask_price_list_NVDA: \n', ask_price_list3)
        elapsed_time = time.process_time() - t
        print('\n Time to fetch data: \n', elapsed_time)
        return ask_price_list3


    
    """

    if symbol == 'SPY':
            
        timestamp2 = df[1][1]
        ask_price2 = df[1][3]
        volume2 = df[1][4]
        global ask_price_list5
        global ask_price_list2

        best_bid_SPY, best_ask_SPY, midpoint_SPY, df2, inventory_qty_SPY = get_orderbook("SPY")

        d2 = {'close':[ask_price2],'volume':[volume2], 'Open_SPY':[df2['Open'][-1]], 'High_SPY':[df2['High'][-1]], 'Low_SPY':[df2['Low'][-1]],
                'Close_SPY':[df2['Close'][-1]],'midpoint_SPY':[midpoint_SPY],'Volume_SPY':[df2['Volume'][-1]],
                'best_bid_SPY':[best_bid_SPY], 'best_ask_SPY':[best_ask_SPY],
                'mu_SPY':[df2['mu'][-1]], 'gamma_SPY':[df2['gamma'][-1]], 'sigma_SPY':[df2['sigma'][-1]], 'k_SPY':[df2['k'][-1]],
                'bid_alpha_SPY':[df2['bid_alpha'][-1]], 'ask_alpha_SPY':[df2['ask_alpha'][-1]], 'ask_sum_delta_vol_SPY':[df2['ask_sum_delta_vol'][-1]], 
                'bid_sum_delta_vol_SPY':[df2['bid_sum_delta_vol'][-1]], 
                'inventory_SPY':[df2['inventory'][-1]],
                'bid_spread_aysm2_SPY':[df2['bid_spread_aysm2'][-1]], 
                'ask_spread_aysm2_SPY':[df2['ask_spread_aysm2'][-1]], }
        
        row2 = pd.DataFrame(d2, index = [timestamp2])
                
        ask_price_list5 = pd.concat([ask_price_list5, row2])
        volume2 = ask_price_list5['volume'].resample('10S').sum()


        ask_price_list2 = ask_price_list5['close'].resample('10S').ohlc()
        ask_price_list2 = pd.merge(left=ask_price_list2, right=volume2, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
        #ask_price_list2.drop(ask_price_list2.filter(regex='_y$').columns, axis=1, inplace=True)

        for i in ['Open_SPY', 'High_SPY','Low_SPY','Close_SPY', 'Volume_SPY', 'midpoint_SPY', 'best_bid_SPY','inventory_SPY', 'best_ask_SPY','mu_SPY','gamma_SPY','sigma_SPY','k_SPY', 'bid_alpha_SPY', 'ask_alpha_SPY', 'ask_sum_delta_vol_SPY', 'bid_sum_delta_vol_SPY', 'bid_spread_aysm2_SPY', 'ask_spread_aysm2_SPY',]:
            ask_price_list_temp = ask_price_list5[i].resample('10S').mean()
            ask_price_list2 = pd.merge(left=ask_price_list2, right=ask_price_list_temp, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
            #ask_price_list2.drop(ask_price_list2.filter(regex='_y$').columns, axis=1, inplace=True)
        
        ask_price_list2 = ask_price_list2.ffill()
        ask_price_list2 = ask_price_list2.rename(columns={"open":"open_SPY", "high":"high_SPY", "low":"low_SPY", "close":"close_SPY", "volume":"volume_SPY"})

        #print('\n ask_price_list_SPY: \n', ask_price_list2)
        elapsed_time = time.process_time() - t
        print('\n Time to fetch data: \n', elapsed_time)
        return ask_price_list2

    
    if symbol == 'AMD':
            
        timestamp3 = df[1][1]
        ask_price3 = df[1][3]
        volume3 = df[1][4]
        global ask_price_list_AMD5
        global ask_price_list_AMD2

        best_bid_AMD, best_ask_AMD, midpoint_AMD, df3, inventory_qty_AMD = get_orderbook("AMD")

        d2_AMD = {'close':[ask_price3],'volume':[volume3], 'Open_AMD':[df3['Open'][-1]], 'High_AMD':[df3['High'][-1]], 'Low_AMD':[df3['Low'][-1]],
                        'Close_AMD':[df3['Close'][-1]],'midpoint_AMD':[midpoint_AMD], 'inventory_qty_AMD':[inventory_qty_AMD], 'best_bid_AMD':[best_bid_AMD], 
                        'best_ask_AMD':[best_ask_AMD],
                        'mu_AMD':[df3['mu'][-1]], 'gamma_AMD':[df3['gamma'][-1]], 'sigma_AMD':[df3['sigma'][-1]], 'k_AMD':[df3['k'][-1]],
                        'bid_alpha_AMD':[df3['bid_alpha'][-1]], 'ask_alpha_AMD':[df3['ask_alpha'][-1]], 'ask_sum_delta_vol_AMD':[df3['ask_sum_delta_vol'][-1]], 
                        'bid_sum_delta_vol_AMD':[df3['bid_sum_delta_vol'][-1]], 
                        'bid_spread_aysm_AMD':[df3['bid_spread_aysm'][-1]], 
                        'ask_spread_aysm_AMD':[df3['ask_spread_aysm'][-1]], 
                        'bid_spread_aysm2_AMD':[df3['bid_spread_aysm2'][-1]], 
                        'ask_spread_aysm2_AMD':[df3['ask_spread_aysm2'][-1]], }
        
        row2_AMD = pd.DataFrame(d2_AMD, index = [timestamp3])
                
        ask_price_list_AMD5 = pd.concat([ask_price_list_AMD5, row2_AMD])
        volume_AMD2 = ask_price_list_AMD5['volume'].resample('10S').sum()


        ask_price_list_AMD2 = ask_price_list_AMD5['close'].resample('10S').ohlc()
        ask_price_list_AMD2 = pd.merge(left=ask_price_list_AMD2, right=volume_AMD2, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
        #ask_price_list_AMD2.drop(ask_price_list_AMD2.filter(regex='_y$').columns, axis=1, inplace=True)

        for i in ['Open_AMD', 'High_AMD','Low_AMD','Close_AMD','best_bid_AMD', 'best_ask_AMD', 'midpoint_AMD', 'inventory_qty_AMD', 'best_bid_AMD', 'best_ask_AMD','mu_AMD','gamma_AMD','sigma_AMD','k_AMD', 'bid_alpha_AMD', 'ask_alpha_AMD', 'ask_sum_delta_vol_AMD', 'bid_sum_delta_vol_AMD', 'bid_spread_aysm_AMD', 'ask_spread_aysm_AMD', 'bid_spread_aysm2_AMD', 'ask_spread_aysm2_AMD',]:
            ask_price_list_temp_AMD = ask_price_list_AMD5[i].resample('10S').mean()
            ask_price_list_AMD2 = pd.merge(left=ask_price_list_AMD2, right=ask_price_list_temp_AMD, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
            #ask_price_list_AMD2.drop(ask_price_list_AMD2.filter(regex='_y$').columns, axis=1, inplace=True)
        
        ask_price_list_AMD2 = ask_price_list_AMD2.ffill()
        ask_price_list_AMD2 = ask_price_list_AMD2.rename(columns={"open":"open_AMD", "high":"high_AMD", "low":"low_AMD", "close":"close_AMD", "volume":"volume_AMD"})

        #print('\n ask_price_list_AMD2: \n', ask_price_list_AMD2)
        return ask_price_list_AMD2
    """
    

        
        

    """ask_price_list4 = pd.merge(left=ask_price_list3, right=ask_price_list2, left_index=True, right_index=True)
    ask_price_list4 = pd.merge(left=ask_price_list4, right=ask_price_list_AMD2, left_index=True, right_index=True)
    
    ask_price_list4 = ask_price_list4.ffill()
    print('\n latest merged df: \n', ask_price_list4)
    print('\n latest data recieved: \n', ask_price_list4[-1:])

    return ask_price_list4"""








            
data_out = pd.DataFrame()

async def create_model(data):

    t = time.process_time()
    
    take_profit_method(symbol='NVDA')
    #take_profit_method(symbol='SPY')

    global data_out

    data_in = await trade_data_handler(data)


    data_out = pd.merge(left=data_in, right=data_out, left_index=True, right_index=True, how='outer', suffixes=('', '_y'))

    data_out.drop(data_out.filter(regex='_y$').columns, axis=1, inplace=True)


    
    data_out = data_out.ffill()
    data_out = data_out.fillna(0.0000001)
    #print('\n latest merged df: \n', data_out)
    #print('\n latest data recieved: \n', data_out[-1:])

    elapsed_time = time.process_time() - t
    print('\n Time to fetch data: \n', elapsed_time)

    dataset = data_out
    symbol_list = ['NVDA', 'NVDA', ]
    side_list = ['OrderSide.BUY', 'OrderSide.SELL' ]
    x_list = [dataset, dataset]



    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(make_model, x_list, symbol_list, side_list)

    now1 = datetime.now()
    print('\n ------- Current Local Machine Time ------- \n', now1)
    take_profit_method(symbol='NVDA')
    #take_profit_method(symbol='SPY')
    get_time_til_close(symbol='NVDA')
    #get_time_til_close(symbol='SPY')



wss_client.subscribe_trades(create_model, "NVDA")

wss_client.run()





"""
async def g():
    # Pause here and come back to g() when f() is ready
    r = await f()
    return r

"""
