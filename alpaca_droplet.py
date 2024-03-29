# Tristan Orndoff
# 
# Here is my semi-automated market making bot based off of "High-frequency trading in a limit order book" by Marco Avellaneda and Sasha Stoikov (2008) and,
#  "Dealing with the Inventory Risk. A solution to the market making problem" by Olivier Guéant, Charles-Albert Lehalle, and Joaquin Fernandez Tapia (2012).
# It utilizes the reference price concept put forth by Avellaneda and Stoikov and the solutions to the inventory control problem from Guéant, Lehalle, and Tapia. 
# I did this because this project has been, and will always be a work in progress that I do in my free time. 
# It's an evolution of my learning and a product of my preference for agile development. 
# Therefore, I have created a Frankenstien, an amalgamation of my code looked over many nights. 
#
# This program has been designed to be automatically uploaded from my Github repo. 
# to a DigitalOcean droplet through a bash script ran on my local computer before work in the morning.
# The bash script uses the DigitalOCean API to instantly create a unique doplet with my credentials, 
# then utilizes ssh commands to connect to my Github repo. and dowloand the current version of this program.
# It downloads the require packages and initializes this python script. The script runs in a loop until 9am EST, 
# It starts the background async methods (calibrate_params and take_profit_method.)
# After that it connects to the Alpaca data stream for the given symbols, currently only "IWM", and runs the trading loop until close.
#
#
#
#
# The trading loop logic is: 
#                           Raw Data Input -> Resampled Data -> Model Creation, Training, and Prediction ------------------------> Order Placement and Matching ------------------------v
#                                                                     ^                                      |                                                                          |       
#                                                                     |                                      |-If there's no predicted change, the loop goes back to model creation     |      
#                                                                     |                                      |                                                                          |   
#                                                                     |<-----------------------------------< v <----------------------------------------------------------------------< v
#
#
# To Do: 
# Fix A and k background calibration
# Set up the bash script on it's own droplet so that I don't have to do anything in the mornings
#
#
#
#
#
#
# Necessary packages to import
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
from scipy.optimize import curve_fit
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
from alpaca.data.live import StockDataStream, CryptoDataStream
warnings.filterwarnings('ignore')
import numba as nb
from numba import jit
from scipy.signal import savgol_filter
from scipy.signal import *
pd.set_option("display.precision", 3)
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)
print(datetime.now())

# Required initial values
res = 0
column_price = 'open'
column_high = 'high'
column_low = 'low'
column_volume = 'volume'
current_variance = 0.3
midpoint_SPY = 0
midpoint_IWM = 0
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
a = 0.5
k = 0.3
dataset = pd.DataFrame()
df = pd.DataFrame()
data_in = pd.DataFrame()
data_out = pd.DataFrame()
future_period = 1
future_period1 = 10
std_period = 15
ma_period = 15
price_deviation_period = 15
volume_deviation_period = 15
yf_download_previous = 0



now = datetime.now()
ask_price_list = pd.DataFrame()
ask_price_list5 = pd.DataFrame()
ask_price_list_AMD5 = pd.DataFrame()
current_vwap = 200
symbol = "IWM"
BASE_URL = "https://paper-api.alpaca.markets"
A_KY = "alpaca paper trading account key"
S_KY = "alpaca secret key"
wss_client = StockDataStream(A_KY, S_KY)
trading_client = TradingClient(A_KY, S_KY, paper=True)
symbol = symbol
rh_key = "robinhood key"
totp  = pyotp.TOTP(rh_key).now()
un = "your robinhood login username"
pw = "robinhood password"
login = r.login(un,pw, mfa_code=totp)





# Fast RSI Calculator
@nb.jit(fastmath=True, nopython=True, cache=True)   
def calc_rsi( array, deltas, avg_gain, avg_loss, n ):

    # Use Wilder smoothing method
    up   = lambda x:  x if x > 0 else 0
    down = lambda x: -x if x < 0 else 0
    i = n+1
    for d in deltas[n+1:]:
        avg_gain = ((avg_gain * (n-1)) + up(d)) / n
        avg_loss = ((avg_loss * (n-1)) + down(d)) / n
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            array[i] = 100 - (100 / (1 + rs))
        else:
            array[i] = 100
        i += 1

    return array

# Fast RSI Calculator
@jit(cache=True) 
def get_rsi( array, n = 14 ):   

    deltas = np.append([0],np.diff(array))

    avg_gain =  np.sum(deltas[1:n+1].clip(min=0)) / n
    avg_loss = -np.sum(deltas[1:n+1].clip(max=0)) / n

    array = np.empty(deltas.shape[0])
    array.fill(np.nan)

    array = calc_rsi( array, deltas, avg_gain, avg_loss, n )
    return array




# Fast VWAP for OHLC data
@jit(cache=True)
def np_vwap(h,l,v):
    return np.cumsum(v*(h+l)/2) / np.cumsum(v)

# Fast VWAP for 1D prices
@jit(cache=True)
def d_vwap(c,v):
    return np.cumsum(v*c) / np.cumsum(v)

# Fast exponential decay used for calibrating A and k
@jit(cache=True, nopython=True)
def exp_decay(k,A,delta,c):
    return A * np.exp(-k * delta) + c

# A looped async function that will place n-orders away from the midpoint, log the time until hit, and fit the data to an exponential decay curve.
# This will give us the variables A and k for the optimal bid and ask spread
# Currently not working as intended
async def calibrate_params(symbol):
    while True:

        # Set variables
        order_id_list_buy = []
        order_id_list_sell = []
        order_i_list_buy = []
        order_i_list_sell = []
        duration_buy = []
        duration_sell = []
        
        # Allow the local variable to be set as the global variable to be used later on
        global a
        global k


        try:    
            #ask_alpha, bid_alpha, bid_sum_delta_vol, ask_sum_delta_vol, midpoint = get_pricebook(symbol)
            midpoint = current_vwap
            for i in range(1,5):

                market_order_data = LimitOrderRequest(
                                symbol=symbol,
                                qty=1,
                                side=OrderSide.BUY,
                                type='limit',
                                time_in_force=TimeInForce.GTC,
                                limit_price = round((float(midpoint) - float(i)/100.0), 2),
                                
                                
                            )
                limit_order_data = trading_client.submit_order(market_order_data)
                
                time.sleep(4)

            time.sleep(60)
            limit_order_data = pd.DataFrame(limit_order_data)
            #print("\n limit_order_data: \n", limit_order_data)
            order_id = limit_order_data[1][1]

            order_id_list_buy.append(order_id)
            order_i_list_buy.append(i)

            for i in order_id_list_buy:
                order = pd.DataFrame(trading_client.get_order_by_client_id(i))
                print("\n order: \n", order)
                order_begin = order[1][2]
                order_end = order[1][5]

                if order_end == None:
                    duration = 100
                else:
                    duration = order_end - order_begin

                if duration == None:
                    duration = 100
                duration_buy.append(str(duration))
                
                # The sell side of the calibration problem that is depreciated until I can get the buy side working as intended.
                """
                market_order_data = LimitOrderRequest(
                                symbol=symbol,
                                qty=1,
                                side=OrderSide.SELL,
                                type='limit',
                                time_in_force=TimeInForce.GTC,
                                limit_price = round((float(midpoint) + float(i)/100.0), 2),
                                #take_profit={'limit_price': round((limit_price+ (spread*take_profit_multiplier)), 2)},
                                #stop_loss={'stop_price': round((limit_price+ (spread*loss_stop_multiplier)), 2),
                                #'limit_price':  round((limit_price+ (spread*loss_limit_multiplier)), 2)},
                                
                            )
                limit_order_data = trading_client.submit_order(market_order_data)
                time.sleep(1)
                limit_order_data = pd.DataFrame(limit_order_data)
                order_id = limit_order_data[1][1]
                

                order_id_list_sell.append(order_id)
                order_i_list_sell.append(i)

            

            for i in order_id_list_sell:
                order = pd.DataFrame(trading_client.get_order_by_client_id(i))
                order_begin = order[1][2]
                order_end = order[1][5]

                duration = order_end - order_begin
                duration_sell.append(duration)



            X_sell = order_i_list_sell.reverse()
            y_sell = duration_sell.reverse()
            popt_sell, pcov_sell = curve_fit(exp_decay, X_sell, y_sell, p0=(1,1,1))
            print("\n Calculated parameters for ", symbol, " selling side: ", popt_sell)
            """

            
            
            X = order_i_list_buy.reverse()
            y = duration_buy.reverse()

            print(X)
            print(y)

            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(100)
            y = y.replace([np.inf, -np.inf], np.nan)
            y = y.fillna(100)
            popt, pcov = curve_fit(exp_decay, X, y, p0=(1,1,1))

            

            print("\n Calculated parameters for ", symbol, " buying side: ", popt)
            

            calibrate_params_previous = popt

        except:
            print("\n Calibrating k, A error... \n")
            print(traceback.format_exc()) 

        await asyncio.sleep(300)

# Downloads price data from yahoo finance
def yf_download(symbol):
    global yf_download_previous

    try:
        
        df_orderbook = pd.DataFrame(yf.download(symbol, period="1d", interval="1m"))
        yf_download_previous = df_orderbook # Creates a variable that stores the previous version of df_orderbook for when Yahoo denies the request
        return df_orderbook
    
    except:
        return yf_download_previous

# Connects to Robinhood's API and extracts Lvl 2 orderbook data
def get_pricebook(symbol):


    
    try:
        


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
            ask_alpha = 0.002
        bid_alpha = (bid_sum_delta_vol - ask_sum_delta_vol) / ((bid_sum_delta_vol + ask_sum_delta_vol)/2)
        if bid_alpha <= 0:
            bid_alpha = 0.002
        #print("alpha:", alpha)
        

    except:
        ask_alpha = 0.5
        bid_alpha = 0.5
        bid_sum_delta_vol = 10000
        ask_sum_delta_vol = 10000
        midpoint = 999
        print(traceback.format_exc())
        return ask_alpha, bid_alpha, bid_sum_delta_vol, ask_sum_delta_vol, midpoint

    finally:

        get_pricebook_previous = [ask_alpha, bid_alpha, bid_sum_delta_vol, ask_sum_delta_vol, midpoint]
        return ask_alpha, bid_alpha, bid_sum_delta_vol, ask_sum_delta_vol, midpoint
    


    

# Retrieves current inventory position for a given symbol from Alpaca
# Preferred return inventory method
def get_inventory(symbol):

    t = time.process_time()

    
    

    try:
        #df_orderbook = yf_download()
        #ask_alpha, bid_alpha, bid_sum_delta_vol, ask_sum_delta_vol, midpoint = get_pricebook(symbol)  
        symbol = symbol
        trading_client = TradingClient(A_KY, S_KY, paper=True)
        position = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(position)
        inventory_qty = int(ORDERS[1][6])

    except:
        #df_orderbook = yf_download_previous
        #ask_alpha, bid_alpha, bid_sum_delta_vol, ask_sum_delta_vol, midpoint = get_pricebook.previous

        print("No inventory position.")
        inventory_qty = 1
        #print(traceback.format_exc())

    finally:

        
        elapsed_time = time.process_time() - t
        print('\n Time to ordebook method: \n', elapsed_time)

    return inventory_qty





# Calculates the current remaining time until the trading session closes
def get_time_til_close(symbol):

    try:
        symbol = symbol
        now = datetime.now()
        trading_client = TradingClient(A_KY, S_KY, paper=True)
        

        if int(now.hour) > 21:
            if int(now.minute) > 57:
                cancel_orders_for_symbol(symbol=symbol)
                trading_client.close_position(symbol)
                
    except:
        print("get_time_til_close exception. It's not trading time...")
        #print(traceback.format_exc())
        
# Calculates the inventory risk (aka gamma or ) for any given symbol
def get_inventory_risk(symbol):
    inventory_risk = 0.02
    try:
        inventory_qty = 1
        symbol = str(symbol)

        trading_client = TradingClient(A_KY, S_KY, paper=True)
    
        position = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(position)




        inventory_qty = int(ORDERS[1][6])

        if symbol == "IWM":
            inventory_risk = 0.1 * (abs(inventory_qty)/10) * (current_variance - current_variance.min() / current_variance.max() - current_variance.min())

        if symbol == 'SPY':
            inventory_risk = 0.01 * abs(inventory_qty)

    except:

        print("No inventory position.")
        inventory_qty = 1
        inventory_risk = 0.002
        #print(traceback.format_exc())
        
        
        
    finally:
            print("\n Current", inventory_qty, inventory_risk, "inventory and risk. \n")
            
    

    return inventory_risk

# Retrieves current inventory position for a given symbol from Alpaca
# Old and depreciated.
def get_open_position(symbol):

    try:
        symbol = symbol
        trading_client = TradingClient(A_KY, S_KY, paper=True)
    
        position = trading_client.get_open_position(symbol)
        ORDERS = pd.DataFrame(position)
    
        inventory_qty = int(ORDERS[1][6])

        side = str(ORDERS[1][7])
        qty = float(ORDERS[1][20])

        
    except:

        print("No inventory position.")
        inventory_qty = 1
        #print(traceback.format_exc())

    finally:
            print("\n Current", qty, side, "position. \n")
            return inventory_qty
        
    
    



# A looped, background async method that checks the total, current profit for a given symbol every three seconds
# and closes it with market orders if above a certain threshold.
async def take_profit_method(symbol):
    while True:
        try:
            
            
            symbol = symbol
            trading_client = TradingClient(A_KY, S_KY, paper=True)
            position = trading_client.get_open_position(symbol)
            ORDERS = pd.DataFrame(position)

            cancel_orders_for_symbol(symbol)
            

            if float(ORDERS[1][10]) / abs(float(ORDERS[1][6])) >=  0.06:
                cancel_orders_for_symbol(symbol=symbol)
                
                trading_client.close_position(symbol)

            
            
            if float(ORDERS[1][12]) >=  6:
                
                cancel_orders_for_symbol(symbol=symbol)
                
                trading_client.close_position(symbol)
                

            
            # Depreciated stop-loss closing conditions
            """
            if float(ORDERS[1][10]) / abs(float(ORDERS[1][6])) <=  -0.12:
                cancel_orders_for_symbol(symbol=symbol)
                
                trading_client.close_position(symbol)

            if float(ORDERS[1][12]) <=  -13:

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
                    
                    trading_client.close_position(symbol)"""
                    

        except:
            print ("\n take_profit_method error. \n")
            #print(traceback.format_exc())

        finally:
            print("\n Current positions have been closed. \n")
            await asyncio.sleep(3)





order_list = []

# Sends a bracket, limit order including a loose stop-loss and a tight take-profit through Alpaca
def limit_order(symbol, limit_price, side, take_profit, stop_loss, qty, inventory_risk):
    global order_list
    symbol = str(symbol)
    
    market_order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=int(qty),
                    side=side,
                    type='limit',
                    order_class = OrderClass.BRACKET,
                    time_in_force=TimeInForce.GTC,
                    limit_price = round(limit_price, 2),
                    take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                    stop_loss=StopLossRequest(stop_price = round(stop_loss, 2))
                )
    limit_order_data = trading_client.submit_order(market_order_data)

    limit_order_data = pd.DataFrame(limit_order_data)

    order_id = limit_order_data[1]
                

    order_list.append(order_id)

    #print("spread, limit_price: ", spread, limit_price)
    #print(limit_order_data)




            
    
    
    

    
# Cancels all orders for both BUY and SELL sides for a given symbol
def cancel_orders_for_symbol(symbol):

    try:
        
        trading_client = TradingClient(A_KY, S_KY, paper=True)
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
        

# Cancels all orders for one BUY or SELL side for a given symbol
def cancel_orders_for_side(symbol, side):

    try:

        trading_client = TradingClient(A_KY, S_KY, paper=True)
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

# Matches current, available positions with limit orders
def match_orders_for_symbol(symbol):

    qty = 1

    

    try:
        symbol = symbol
        trading_client = TradingClient(A_KY, S_KY, paper=True)
        ORDERS = trading_client.get_open_position(symbol)

    except:    
        print("\n match_orders_for_symbol() exception, probably no inventory present. \n")  
        print(traceback.format_exc())   
        pass

    else:
        ORDERS = pd.DataFrame(ORDERS)
        #print('\n ORDERS: \n',ORDERS)
        side = str(ORDERS[1][7])
        qty = float(ORDERS[1][20])
        
        cancel_orders_for_symbol(symbol)

        if str(side) == 'PositionSide.SHORT':

            spread = -0.02
            #cancel_orders_for_side(symbol=symbol, side='sell')
            best_spread = best_bid
            stop_loss = round((res + (best_spread * 10)), 2)
            stop_loss_limit = round((stop_loss - 0.01), 2)
            take_profit = round((res - (best_spread * 3)), 2)
            if float(best_spread) > -0.01:
                best_spread = round((best_spread - 0.05), 2)

            spread = round(best_spread, 2)
            current_price = round(res, 2)
            limit_price = round((current_price + spread), 2)

            cancel_orders_for_side(symbol=symbol, side='buy')
            
            for i in np.linspace(1, qty, num=10):
                limit_order(symbol=symbol, 
                            limit_price= round((limit_price - float(i)), 2),
                            side=OrderSide.BUY, 
                            take_profit = round((limit_price - float(i)), 2),
                            stop_loss = round((limit_price + float(i)), 2),
                            qty = abs(i),
                            inventory_risk = get_inventory_risk(symbol = symbol)
                            )
            print("\n Current", qty, side, "positions have been matched. \n")

            
        

        if str(side) == 'PositionSide.LONG':
            
            cancel_orders_for_side(symbol=symbol, side='sell')

            spread = -0.02
            #cancel_orders_for_side(symbol=symbol, side='sell')
            best_spread = best_ask
            stop_loss = round((res - (best_spread * 10)), 2)
            stop_loss_limit = round((stop_loss + 0.01), 2)
            take_profit = round((res + (best_spread * 3)), 2)
            if float(best_spread) < -0.01:
                best_spread = round((best_spread + 0.05), 2)

            spread = round(best_spread, 2)
            current_price = round(res, 2)
            limit_price = round((current_price + spread), 2)

            for i in np.linspace(1, qty, num=10):
                limit_order(symbol=symbol, 
                            limit_price= round((limit_price + float(i)), 2),
                            side=OrderSide.SELL, 
                            take_profit = round((limit_price + float(i)), 2),
                            stop_loss = round((limit_price - float(i)), 2),
                            qty = abs(i),
                            inventory_risk = get_inventory_risk(symbol = symbol)
                            )
            
            print("\n Current", qty, side, "positions have been matched. \n")
        

    finally:
        print(f'\n match_orders_for_symbol: {symbol} (a finally block thats always executed)')


# Draws up a confusion matrix, accuracy, and F1 scores for validation data
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

# Fast std calculator
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

# Fast zscore calculator
@jit(cache=True)
def z_score(vals):
    vals = np.log(vals)
    vals = ((vals - vals.expanding().mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}))/vals.expanding().std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})).pct_change()
    return vals

# A method for applying the z_score(vals) method to all columns in a dataframe
def z_score_df(df):
    df = df.apply(lambda x : z_score(x))
    return df

# Semi fast feature generator for features that won't be globally transmitted
@jit(cache=True)
def create_features(dataset):

        
        #print(dataset)
        
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)

        dataset['spread3'] = dataset['open'] - ((dataset['low'] + dataset['high'])/2)
        dataset['spread2'] = dataset['high'] - dataset['low']
        dataset['Volatility'] = (np.log(dataset['open']).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5))
        dataset['Volatility2'] = (np.log(dataset['Volatility']).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5))
        dataset['Volatility3'] = (np.log(dataset['open']).rolling(25).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(25))
        dataset['Volatility4'] = (np.log(dataset['Volatility3']).rolling(25).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(25))
        dataset['Volatility_ratio'] = dataset['Volatility'] / dataset['volume'].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})

        dataset['trade-able spread'] = dataset['spread2'] - np.sqrt(8 * np.log(100) * dataset['Volatility'])
        
        dataset['last_return'] = np.log(dataset["open"]).pct_change()
        dataset['std_normalized'] = np.log(dataset[column_price]).rolling(std_period).apply(std_normalized, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
        dataset['ma_ratio'] = np.log(dataset[column_price]).rolling(ma_period).apply(ma_ratio, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
        #dataset['price_deviation'] = np.log(dataset[column_price]).rolling(price_deviation_period).apply(values_deviation, engine='numba', raw=True, engine_kwargs={"nogil":True, "nopython": True,})
        #dataset['volume_deviation'] = np.log(dataset[column_volume]).rolling(volume_deviation_period).apply(values_deviation)
        dataset['OBV'] = stats.zscore((np.sign(dataset["open"].diff()) * dataset['volume']).fillna(0.0000001).cumsum())
        dataset['OBV1'] = (np.sign(dataset["open"].diff()) * dataset['volume']).fillna(0.0000001).cumsum()
        dataset['OBV2'] = (np.sign(dataset["open"].rolling(10).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}).diff()) * dataset['volume']).fillna(0.0000001).cumsum()
        dataset['OBV3'] = (np.sign((dataset["open"].rolling(10).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) / dataset["volume"].rolling(10).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})).diff()) * dataset['volume']).fillna(0.0000001).cumsum()

        dataset['vwap'] = np_vwap(h= dataset['high'],l= dataset['low'],v= dataset['volume'])
        dataset['D_vwap'] = d_vwap(c= dataset['open'],v= dataset['volume'])

        dataset['rsi_open'] = get_rsi( dataset["open"], 14 )
        dataset['rsi_high'] = get_rsi( dataset["high"], 14 )
        dataset['rsi_low'] = get_rsi( dataset["low"], 14 )
        dataset['rsi_close'] = get_rsi( dataset["close"], 14 )
        dataset['rsi_volume'] = get_rsi( dataset["volume"], 14 )

        dataset['rsi_vwap'] = get_rsi( dataset["vwap"], 14 )
        dataset['rsi_D_vwap'] = get_rsi( dataset["D_vwap"], 14 )

        dataset['spread'] = abs(np.log(dataset['open']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) - ((np.log(dataset['low']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) + np.log(dataset['high']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}))/2))
        dataset['variance'] = (np.log(dataset['open']).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5)).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
        dataset['open+var'] = np.log(dataset['open']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) + (np.log(dataset['open']).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5)).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
        dataset['open-var'] = np.log(dataset['open']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) - (np.log(dataset['open']).rolling(5).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(5)).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
        dataset['open+var_diff'] = np.cumsum(np.log(dataset['open']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}).pct_change() - dataset['open+var'].shift(1))
        dataset['open+var_diff'] = np.cumsum(np.log(dataset['open']).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}).pct_change() - dataset['open-var'].shift(1))
        dataset['inventory'] = get_inventory(symbol)
        dataset["mu"] = abs((np.log(dataset["open"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})).pct_change()/2) * 10000)

        dataset['gamma'] = get_inventory_risk(symbol = symbol)


        dataset['sigma'] = ((np.log(dataset["open"]).rolling(15).std(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) * np.sqrt(15)).rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})) * 100


        dataset['Volume'] = dataset['volume'] + 1


        #dataset['bid_sum_delta_vol'] = bid_sum_delta_vol
        #dataset['ask_sum_delta_vol'] = ask_sum_delta_vol
        dataset['market_impact'] = dataset['sigma']*np.sqrt(dataset['inventory']/dataset['Volume'].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}))
        

        dataset['bid_spread_aysm'] = ((1 / dataset['gamma'] * np.log(1 + dataset['gamma'] / dataset['k']) + (2 * dataset['inventory'] + 1) / 2 * np.sqrt((dataset['sigma']**2 * dataset['gamma']) / (2 * dataset['k'] * dataset['bid_alpha']) * (1 + dataset['gamma'] / dataset['k'])**(1 + dataset['k'] / dataset['gamma']))) / 100000)

        dataset['ask_spread_aysm'] = ((1 / dataset['gamma'] * np.log(1 + dataset['gamma'] / dataset['k']) - (2 * dataset['inventory'] - 1) / 2 * np.sqrt((dataset['sigma']**2 * dataset['gamma']) / (2 * dataset['k'] * dataset['ask_alpha']) * (1 + dataset['gamma'] / dataset['k'])**(1 + dataset['k'] / dataset['gamma']))) / 100000)
        

        # ((1 / g * log(1 + g / k) + (  mu / (g * s **2) - (2 * i - 1) / 2) * sqrt((s**2 * k) / (2 *k * a) * (1 + g / k)**(1 + k / g))) 
        # ((1 / gamma * log(1 + gamma / k) + (  mu/ (gamma * sigma**2) - (2 * i - 1) / 2) * sqrt((sigma**2 * k) / (2 *k * ask_alpha) * (1 + gamma / k)**(1 + k / gamma))) / 9999999) 
        dataset['bid_spread_aysm2'] = ((1 / dataset['gamma'] * np.log(1 + dataset['gamma'] / dataset['k']) + (- dataset["mu"] / (dataset['gamma'] * dataset['sigma']**2) + (2 * dataset['inventory'] + 1) / 2) * np.sqrt((dataset['sigma']**2 * dataset['k']) / (2 * dataset['k'] * dataset['bid_alpha']) * (1 + dataset['gamma'] / dataset['k'])**(1 + dataset['k'] / dataset['gamma']))) / 25000)

        dataset['ask_spread_aysm2'] = ((1 / dataset['gamma'] * np.log(1 + dataset['gamma'] / dataset['k']) + (  dataset["mu"] / (dataset['gamma'] * dataset['sigma']**2) - (2 * dataset['inventory'] - 1) / 2) * np.sqrt((dataset['sigma']**2 * dataset['k']) / (2 * dataset['k'] * dataset['ask_alpha']) * (1 + dataset['gamma'] / dataset['k'])**(1 + dataset['k'] / dataset['gamma']))) / 25000)

        s = dataset['sigma']
        g = dataset['gamma']
        k = dataset['k']
        m = dataset['mu']
        q = dataset['inventory']
        a = dataset['bid_alpha']

        dataset['inventory_risk_roc'] = (s**2 * (g/k + 1)**(k/g + 1)*((k/g + 1)/(k (g/k + 1)) - (k * np.log(g/k + 1))/g**2)*(m/(g * s**2) + 1/2 (1 - 2 * q)))/(2 * np.sqrt(2) * a * np.sqrt((s**2 (g/k + 1)**(k/g + 1))/a)) - (m * np.sqrt((s^2 (g/k + 1)**(k/g + 1))/a))/(np.sqrt(2) * g**2 * s**2) - np.log(g/k + 1)/g**2 + 1/(g * k (g/k + 1))
        dataset['inventory derivative'] = -(np.sqrt(((1 + g/k)**(1 + k/g) * s**2)/a)/np.sqrt(2))
        #dataset['bid_spread_aysm3'] = 1 / dataset['gamma'] * np.log( 1 + dataset['gamma']/dataset['k'] ) + dataset['market_impact']/2 + (2 * dataset['inventory'] + 1)/2 * np.exp((dataset['k']/4) * dataset['market_impact']) * np.sqrt( ((dataset['sigma'] * 2 * dataset['gamma']) / (2 * dataset['k'] * dataset['bid_alpha'])) ( 1 + dataset['gamma'] * dataset['k'] )**(1+ dataset['k'] * dataset['gamma']) )

        #dataset['ask_spread_aysm3'] = 1 / dataset['gamma'] * np.log( 1 + dataset['gamma']/dataset['k'] ) + dataset['market_impact']/2 - (2 * dataset['inventory'] - 1)/2 * np.exp((dataset['k']/4) * dataset['market_impact']) * np.sqrt( ((dataset['sigma'] * 2 * dataset['gamma']) / (2 * dataset['k'] * dataset['ask_alpha'])) ( 1 + dataset['gamma'] * dataset['k'] )**(1+ dataset['k'] * dataset['gamma']) )
        
        
        """
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)
        #sos = butter(4, 0.125, output='sos')
        
        for i in dataset.columns.tolist():
            dataset[str(i)+'_volu_ratio'] = dataset[i] / dataset["volume"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)
        
        for i in dataset.columns.tolist():
            dataset[str(i)+'_volu_ratio'] = dataset[i] / dataset["Volatility"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
            """

        from sklearn import linear_model
        lr = linear_model.LinearRegression()

        for i in dataset.columns.tolist():
            y = dataset[i][-10:].values
            X = range(len(y))
            lr.fit(X,y)
            dataset[i+'+1'] = lr.predict(len(y)+1)

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)

        for i in dataset.columns.tolist():
            detrend(dataset[i], overwrite_data=True)

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)
        
        for i in dataset.columns.tolist():
            #dataset[str(i)+'_sosfiltfilt'] = sosfiltfilt(sos, dataset[i])
            #dataset[str(i)+'_savgol'] = savgol_filter(dataset[i], 5, 3)
            #dataset[str(i)+'_smooth_5'] = dataset[i].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
            dataset[str(i)+'_smooth_10'] = dataset[i].rolling(10).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
            #dataset[str(i)+'_smooth_60'] = dataset[i].rolling(60).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})
        

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)

        return dataset

# The core logic behind creating a Catboost classifier that predicts the change in returns 
# for a given future period (usually one time-step ahead). 
# This includes calculating the target, y;
# Collecting and normalizing all of the data inputs into a single array,
# Training BUY and SELL models, cross-validating hyper-parameters, and predicting the next time-step,
# Calculating optimal bids, asks, and reservation prices,
# Sending limit orders if the model predicts a favorable trade
# The sampling period isn't handled by this method.
def make_model(dataset, symbol, side):
    global best_ask
    global best_bid
    global res
    global current_variance

    try: 
        t0 = time.time()

        symbol = str(symbol)
        get_time_til_close(symbol=symbol)


        #best_bid, best_ask, midpoint, df, inventory_qty = get_orderbook(symbol = symbol)
        ask_alpha, bid_alpha, bid_sum_delta_vol, ask_sum_delta_vol, midpoint = get_pricebook(symbol)


        


        # Feature params
        

        mid_price = (float(dataset['d_vwap'][-1]))
        
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



        


        
        dataset['ask_alpha'] = ask_alpha
        dataset['bid_alpha'] = bid_alpha
        dataset['bid_sum_delta_vol'] = bid_sum_delta_vol
        dataset['ask_sum_delta_vol'] = ask_sum_delta_vol
        dataset['k'] = k
        dataset['midpoint'] = midpoint
        

        dataset['bid_alpha'] = (dataset["volume"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) / dataset["volume"].rolling(25).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})) / np.exp(dataset['k'] * 1)
        dataset['ask_alpha'] = (dataset["volume"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) / dataset["volume"].rolling(25).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})) / np.exp(dataset['k'] * 1)

        dataset['k'] = np.log((dataset['bid_alpha'] * 1) / (dataset["volume"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) / dataset["volume"].rolling(25).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})))

        dataset['bid_alpha'] = (dataset["volume"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) / dataset["volume"].rolling(25).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})) / np.exp(dataset['k'] * 1)
        dataset['ask_alpha'] = (dataset["volume"].rolling(5).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,}) / dataset["volume"].rolling(25).mean(engine='numba', engine_kwargs={"nogil":True, "nopython": True,})) / np.exp(dataset['k'] * 1)


        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)

        dataset = create_features(dataset)

        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.fillna(0.0000001)

        now = datetime.now()

        end_of_day = datetime(now.year, now.month, now.day, hour=22)




        steps_in_day = end_of_day - now
        steps_in_day = float(round(steps_in_day.total_seconds()/60))
        #steps_in_day = 100
        #mid_price = float(current_price)

        current_variance = (dataset["variance"][-1])
        current_spread = (dataset["spread"][-1])
        current_variance = float(current_variance) * 1000
        current_spread = float(current_spread) * 1000
        #dataset['best_bid2'] = best_bid
        #dataset['best_ask2'] = best_ask
        #dataset['inventory_qty2'] = inventory_qty
        
        

        inventory = float(inventory_qty)
        inventory_risk = float(get_inventory_risk(symbol))
        variance = float(current_variance)
        total_steps_in_day = float(420)
        #print(steps_in_day)
        print("\n inventory: \n", inventory)

        

        res = np.array(mid_price) - ((np.array(inventory) * np.array(inventory_risk) * (np.array(variance)) * (1 - (np.array(steps_in_day, dtype='float64')/np.array(total_steps_in_day))))/40)
        #np.array(current_spread
        print("\n reservation price: \n", res)
        print("\n reservation price delta: \n", res-mid_price)

        print("\n bid: \n", symbol, dataset['bid_spread_aysm2'][-1])
        print("\n ask: \n", symbol, dataset['ask_spread_aysm2'][-1])

        best_ask = dataset['ask_spread_aysm2'][-1]
        best_bid = dataset['bid_spread_aysm2'][-1]

        #print('\n before transform dataset: \n', dataset)
  
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
        y_sell = y_sell.dropna()

        dataset = dataset.apply(pd.to_numeric, downcast='float')
        dataset = dataset.apply(pd.to_numeric, downcast='integer')



            

        
        
        dataset = dataset[dataset.index.isin(y.index)]
        dataset_sell = dataset[dataset.index.isin(y_sell.index)]
        #dataset_SPY = dataset_SPY[dataset_SPY.index.isin(y_SPY.index)]

        if str(side) == 'OrderSide.BUY':
            side = OrderSide.BUY
            if symbol == 'IWM':
                y = y
            #if symbol == 'SPY':
                #y = y_SPY


        if str(side) == 'OrderSide.SELL':
            side = OrderSide.SELL
            if symbol == 'IWM':
                y = y_sell
                dataset = dataset_sell
            #if symbol == 'SPY':
                #y = y_SPY_sell

        
        #print('\n dataset: \n', dataset)
        #print('\n after winsorize dataset: \n', dataset.describe())
        #print('\n y: \n', y)
        #print('\n y_sell: \n', y_sell.describe())

        #print('\n last dataset input: \n', dataset[-1:])
        #print('\n last y input: \n', y[-1:])

        


        X_train, X_test, y_train, y_test = train_test_split(dataset[-(len(y)):], y, test_size = 0.5, random_state = 42, shuffle=False)
        X_valid, X_test2, y_valid, y_test2 = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42, shuffle=False)


        
        
        train_dataset = cb.Pool(X_train, y_train)
        test_dataset = cb.Pool(X_test2, y_test2)
        valid_dataset = cb.Pool(X_valid, y_valid)
        
        catboost_class = CatBoostClassifier(iterations=500, early_stopping_rounds=5, silent=True, thread_count=-1)
        """
        my_file = Path(f'model_{symbol}_{side}') # file path for persistant model
        if my_file.exists():
            catboost_class = CatBoostClassifier()      # parameters not required.
            catboost_class.load_model(f'model_{symbol}_{side}')
            """
        selected_features = catboost_class.select_features(train_dataset, eval_set=valid_dataset, features_for_select=list(dataset.columns), num_features_to_select=30, steps=4, algorithm='RecursiveByShapValues', shap_calc_type='Approximate', train_final_model=True, logging_level='Silent')
        print('\n selected_features: \n', selected_features['selected_features_names'])
        #catboost_class.select_features(train_dataset, eval_set=test_dataset, num_features_to_select=50, steps=10, algorithm='RecursiveByShapValues', train_final_model=True,)




        grid = {

            'max_depth': randint(2,7),
            'learning_rate': np.linspace(0.001, 1, 100),
            #'iterations': np.arange(100, 1000, 100),
            'l2_leaf_reg': np.linspace(0.1, 20, 50),
            'random_strength': np.linspace(0.1, 20, 50),
            'subsample': np.linspace(0.75, 1, 5),
            'bagging_temperature': np.linspace(0.1, 5, 50),
            'early_stopping_rounds': np.linspace(1, 20, 20),
            'diffusion_temperature':np.linspace(1, 50000, 400),
            'fold_len_multiplier':np.linspace(2, 10, 50),
            #'boosting_type': ['Ordered','Plain'],
            #'thread_count':[-1,-1],
            'loss_function': ['Logloss','CrossEntropy'],
            'eval_metric': ['AUC', 'Precision', 'Recall', 'F1', 'BalancedAccuracy', 'TotalF1', 'BalancedErrorRate', 'PRAUC' ],

            
        }
        tscv = TimeSeriesSplit(n_splits=3, gap=1)
        rscv = HalvingRandomSearchCV(catboost_class, grid, resource='iterations', n_candidates='exhaust', aggressive_elimination=True, factor=10, min_resources=25, max_resources=500, cv=tscv, verbose=1, scoring='f1')

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
                cancel_orders_for_side(symbol=symbol, side='sell')
                best_spread = best_bid
                stop_loss = res - (best_spread * 10)
                stop_loss_limit = stop_loss - 0.01
                take_profit = res + (best_spread * 3)
                if float(best_spread) > -0.01:
                    best_spread = best_spread + -0.05

                spread = best_spread
                current_price = res
                limit_price = round(current_price + spread, 2)

                

                limit_order(symbol=symbol, 
                        limit_price=round(limit_price, 2),
                        side=side, 
                        take_profit = round(take_profit, 2),
                        stop_loss = round(stop_loss , 2),
                        qty = round((100 * (math.exp(-5((float(get_inventory_risk(symbol = symbol)) - 0.019)/(1 - 0.019)))))),
                        inventory_risk = get_inventory_risk(symbol = symbol)
                        )


            if str(side) == 'OrderSide.SELL':
                
                spread = 0.02
                cancel_orders_for_side(symbol=symbol, side='buy')
                best_spread = best_ask
                stop_loss = res + (best_spread * 10)
                stop_loss_limit = stop_loss + 0.01
                take_profit = res - (best_spread * 3)
                
                if float(best_spread) < 0.01:
                    best_spread = best_spread + 0.05

                spread = best_spread
                current_price = res
                limit_price = round(current_price + spread, 2)

                limit_order(symbol=symbol, 
                        limit_price=round(limit_price, 2),
                        side=side, 
                        take_profit = round(take_profit, 2),
                        stop_loss = round(stop_loss , 2),
                        qty = round((101 * (math.exp(-5((float(get_inventory_risk(symbol = symbol)) - 0.019)/(1 - 0.019)))))),
                        inventory_risk = get_inventory_risk(symbol = symbol)
                        )
                

        t1 = time.time()
        total = t1-t0
        print('\n Total time to order: \n', total)
    except:
        print("model error.")  
        print(traceback.format_exc())










# This looped async method streams stock data from Alpaca at a trade level granularity
# Each trade is logged and resampled into a given time-period, currently 5 second intervals 
# Only three streams can be active at any given time
async def trade_data_handler(data):
    # quote data will arrive here
    t = time.process_time()

    

    #print('\n Raw Data: \n', data)
    df = pd.DataFrame(data)

    symbol = df[1][0]


    if symbol == "IWM":
        timestamp = df[1][1]
        ask_price = df[1][3]
        volume = df[1][4]
        global ask_price_list
        global ask_price_list3
        global current_vwap
        
        d = {'close':[ask_price],'volume':[volume]}
        
        row = pd.DataFrame(d, index = [timestamp])
        #print('\n row: \n', row)
        
        ask_price_list = pd.concat([ask_price_list, row])
        ask_price_list['d_vwap'] = d_vwap(ask_price_list['close'], ask_price_list['volume'])
        d_vwap1 = ask_price_list['d_vwap'].resample('5S').mean()
        volume = ask_price_list['volume'].resample('5S').sum()

        ask_price_list3 = ask_price_list['close'].resample('5S').ohlc()
        ask_price_list3 = pd.merge(left=ask_price_list3, right=volume, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))
        ask_price_list3 = pd.merge(left=ask_price_list3, right=d_vwap1, left_index=True, right_index=True,  how='left', suffixes=('', '_y'))


        ask_price_list3 = ask_price_list3.ffill()
        current_vwap = float(ask_price_list3['d_vwap'][-1:])

        #print('\n ask_price_list_IWM: \n', ask_price_list3)
        elapsed_time = time.process_time() - t
        print('\n Time to fetch data: \n', elapsed_time)
        return ask_price_list3


            

# This method ties everything together. It is called after 9am EST on trading days and gives the resampled data from the Alpaca data stream 
# to the concurrent.futures.ProcessPoolExecutor() to create parrallel models with. This has been designed with the intention of being easily scalable with more symbols
async def create_model(data):

    t = time.process_time()
    
    

    #asyncio.gather(calibrate_params("IWM"))
    #asyncio.gather(take_profit_method("IWM"))
    


    global data_out

    data_in = pd.DataFrame()

    data_in = await trade_data_handler(data)

    data_out = pd.merge(left=data_in, right=data_out, left_index=True, right_index=True, how='outer', suffixes=('', '_y'))

    data_out.drop(data_out.filter(regex='_y$').columns, axis=1, inplace=True)
    
    dataset = data_out
    symbol_list = ['IWM', 'IWM', ]
    side_list = ['OrderSide.BUY', 'OrderSide.SELL' ]
    x_list = [dataset, dataset]



    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(make_model, x_list, symbol_list, side_list)

    now1 = datetime.now()
    print('\n ------- Current Local Machine Time ------- \n', now1)
    match_orders_for_symbol(symbol='IWM')


while True:
    now = datetime.now()
    print(now.hour, now.minute, now.second)
    if int(now.hour) >= int(15):

        print(now)
        # Call your CODE() function here
        asyncio.gather(calibrate_params("IWM"))
        asyncio.gather(take_profit_method("IWM"))

        wss_client.subscribe_trades(create_model, "IWM")

        wss_client.run()


    
    #time.sleep(10)
