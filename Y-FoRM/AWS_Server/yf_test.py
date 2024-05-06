import yfinance as yf
import pandas
from datetime import datetime, timedelta

#MACD
def MACD(price, macd_short, macd_long, macd_signal, long_only=False):

        macd_short = price.rolling(macd_short).mean()
        macd_long = price.rolling(macd_long).mean()
        macd = macd_short - macd_long
        macd_signal = macd.rolling(macd_signal).mean()  
        
        long_signal = (macd >= macd_signal) * 1
        short_signal = (macd < macd_signal) * -1
        
        if long_only:
            signal = long_signal
        else:
            signal = long_signal + short_signal
        
        return macd, macd_signal, signal

tickers = 'BTC-USD'

macd_short = 12
macd_long = 26
macd_signal = 9

start = datetime.now().date() - timedelta(1)

btc_price = yf.download(tickers, start, interval="1m")['Adj Close']
macd, macd_signal, signal = MACD(btc_price, macd_short, macd_long, macd_signal, long_only=False)
print("MACD: ", macd[-2],"->",macd[-1])
print("MACD_SIGNAL: ", macd_signal[-2],"->",macd_signal[-1])
print("Signal: ", signal[-2],"->",signal[-1])
if signal[-2] != signal[-1]:
    if signal[-1] == 1:
        print("매수!!!", "PRICE: ",btc_price[-1])
    else:
        print('매도!!!', "PRICE: ",btc_price[-1])

print(btc_price[-1])