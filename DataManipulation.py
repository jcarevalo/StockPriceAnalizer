import datetime as dt
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import requests
import time
import re
from io import StringIO

style.use('seaborn-whitegrid')

class Fetcher:
    api_url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=%s&events=%s&crumb=%s"

    def __init__(self, ticker, start, end=None, interval="1d"):
        """Initializes class variables and formats api_url string"""
        self.ticker = ticker.upper()
        self.interval = interval
        self.cookie, self.crumb = self.init()
        self.start = int(time.mktime(dt.datetime(start.year,start.month,start.day).timetuple()))

        if end is not None:
            self.end = int(time.mktime(dt.datetime(end.year,end.month,end.day).timetuple()))
        else:
            self.end = int(time.time())

    def init(self):
        """Returns a tuple pair of cookie and crumb used in the request"""
        url = 'https://es.finance.yahoo.com/quote/%s/history' % (self.ticker)
        r = requests.get(url)
        txt = r.content
        cookie = r.cookies['B']
        pattern = re.compile('.*"CrumbStore":\{"crumb":"(?P<crumb>[^"]+)"\}')

        for line in txt.splitlines():
            m = pattern.match(line.decode("utf-8"))
            if m is not None:
                crumb = m.groupdict()['crumb']
                crumb = crumb.replace(u'\\u002F', '/')
        return cookie, crumb  # return a tuple of crumb and cookie

    def getData(self, events):
        """Returns a list of historical data from Yahoo Finance"""
        if self.interval not in ["1d", "1wk", "1mo"]:
            raise ValueError("Incorrect interval: valid intervals are 1d, 1wk, 1mo")

        url = self.api_url % (self.ticker, self.start, self.end, self.interval, events, self.crumb)

        data = requests.get(url, cookies={'B':self.cookie})
        content = StringIO(data.content.decode("utf-8"))
        return pd.read_csv(content, sep=',')

    def getHistorical(self, events='history'):
        """Returns a list of historical price data from Yahoo Finance"""
        return self.getData('history')

class Grabber(object):
    """Stock price grabber"""
    def __init__(self, start_date, end_date, ticker, source='yahoo'):
        self.STORE_DIR = 'stock_dfs'
        self.start = dt.datetime(start_date[0], start_date[1], start_date[2])
        self.end = dt.datetime(end_date[0], end_date[1], end_date[2])
        self.path = self.STORE_DIR + '/' + ticker + '.csv'
        self.ticker = ticker
        if not os.path.exists(self.STORE_DIR):
            os.makedirs(self.STORE_DIR)
        if not os.path.exists(self.path):
            if source == 'yahoo':
                my_fetcher = Fetcher(ticker, self.start, self.end)
                # self.df = self.get_data_from_yahoo(self.start, self.end, ticker)
                self.df = my_fetcher.getHistorical()
                self.df.set_index('Date', inplace=True)
                self.df.dropna(inplace=True)
            self.df.to_csv(self.path)
            self.df = pd.read_csv(self.path, parse_dates=True, index_col=0) # This is a workaround to the grabber issue
        else:
            self.update_data(self.start, self.end, ticker, self.path, source)


    def get_closes(self):
        return self.df['Close']

    def get_adj_closes(self):
        return self.df['Adj Close']

    def get_highs(self):
        return self.df['High']

    def get_lows(self):
        return self.df['Low']

    def get_volume(self):
        return self.df['Volume']

    def get_opens(self):
        return self.df['Open']

    def get_df(self):
        return self.df

    def update_data(self, start, end, ticker, path, source):
        """
        Updates a csv file with prices. Needs to improve last value update
        """
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        new_df = pd.DataFrame()
        if source == 'yahoo':
            my_fetcher = Fetcher(ticker, start, end)
            new_df = my_fetcher.getHistorical()

        new_df.set_index('Date', inplace=True)
        df = pd.concat([df, new_df])
        # df.set_index('Date', inplace=True)
        df.dropna(inplace=True)
        df.reset_index('Date', inplace=True)
        df.drop_duplicates('Date', keep='last', inplace=True)
        df.set_index('Date', inplace=True)
        df.to_csv(path)
        self.df = df

    @staticmethod
    def get_data_from_yahoo(start, end, ticker):
        """Downloads data from yahoo"""
        df = web.DataReader(ticker, 'yahoo', start, end)
        return df

    @staticmethod
    def get_data_from_alpha(ticker):
        import io

        url = 'https://www.alphavantage.co/query?'
        req = 'function={req_type}&symbol={ticker}&interval={interval}&outputsize={size}&apikey={key}&datatype=csv'\
            .format(req_type='TIME_SERIES_INTRADAY', interval='1min', size='full',
                    ticker=ticker, key='0F549FI0SLCP73J5')
        my_data = requests.get(url+req)
        my_df = pd.read_csv(io.StringIO(my_data.content.decode('utf-8')), parse_dates=True, index_col=0)

        my_df.reset_index(inplace=True)
        my_df.rename(columns={'timestamp': 'Date', 'close': 'Close', 'open':
                              'Open', 'volume': 'Volume', 'high': 'High', 'low': 'Low'}, inplace=True)

        dummy = my_df.copy()
        dummy.drop(['Date', 'Open', 'High', 'Low', 'Volume'], 1, inplace=True)
        dummy.rename(columns={'Close': 'Adj Close'}, inplace=True)
        my_df = my_df.join(dummy, how='outer')
        my_df.set_index('Date', inplace=True)

        my_df.to_csv('{ticker}_intraday.csv'.format(ticker=ticker))
        my_df.reset_index(inplace=True)
        return my_df


class StockMetricsCalculator(object):

    def __init__(self, data):
        if isinstance(data, Grabber):
            self.data = data.get_df()
            self.ticker = data.ticker
        elif isinstance(data, pd.DataFrame):
            self.data = data
            self.ticker = 'None'
        else:
            raise Exception('You must provide a Grabber object or a pandas.DataFrame object')

    def get_bollinger_bands(self, window=20, k=2, values=None):
        if values is None:
            values = self.data['Adj Close']
        std = pd.Series.rolling(values, window).std()
        mean = pd.Series.rolling(values, window).mean()
        bollinger_up = mean + std*k
        bollinger_down = mean - std*k
        bollinger_central = mean
        return bollinger_up, bollinger_down, bollinger_central

    def exp_moving_average(self, window, values=None):
        if values is None:
            values = self.data['Adj Close'].values
        moving_avg = [0] * len(values)
        alpha = 2 / (1.0 + window)
        moving_avg[0] = values[0]
        for idx in range(1, len(values)):
            moving_avg[idx] = alpha * (-moving_avg[idx - 1] + values[idx]) + moving_avg[idx - 1]
        return np.asarray(moving_avg)

    def moving_average(self, window, values=None):
        if values is None:
            values = self.data['Adj Close'].values
        weights = np.repeat(1.0, window)/window
        smas = np.convolve(values, weights, 'valid')
        return smas

    def rsi_indicator(self, window):
        deltas = np.diff(self.data['Adj Close'].values)
        # rsi = [0]*len(self.closes)
        seed = deltas[:window + 1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down
        rsi = np.zeros_like(self.data['Adj Close'].values)
        rsi[:window] = 100. - 100. / (1. + rs)

        for idx in range(window, len(self.data['Adj Close'].values)):
            if deltas[idx - 1] > 0:
                upval = deltas[idx - 1]
                downval = 0
            else:
                upval = 0
                downval = -deltas[idx - 1]
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            rs = up / down
            rsi[idx] = 100. - 100. / (1. + rs)

        return rsi

    def macd_indicator(self, fast=12, slow=26):
        macd = self.exp_moving_average(fast) - self.exp_moving_average(slow)
        signal_line = self.exp_moving_average(9, values=macd)
        return macd, signal_line

    def true_range(self, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        highs = data_frame['High'].values
        lows = data_frame['Low'].values
        closes = data_frame['Adj Close'].values
        true_range = [0]*(len(closes)-1)
        for idx in range(0, len(closes)-1):
                true_range[idx] = max([highs[idx+1] - lows[idx+1],
                                       abs(highs[idx+1] - closes[idx]),
                                       abs(lows[idx+1] - closes[idx])])
        return np.asarray(true_range)

    def typical_prices(self, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        return (data_frame['Adj Close'].values + data_frame['Low'].values + data_frame['High'].values)/3

    def get_av_true_range(self, window=14, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        true_range = self.true_range(data_frame=data_frame)
        atr = self.exp_moving_average(window, values=true_range)
        return atr

    def get_dmis(self, window=14, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        move_up = np.diff(data_frame['High'].values)
        move_down = -np.diff(data_frame['Low'].values)

        pdm = np.zeros_like(move_up)
        ndm = np.zeros_like(move_down)
        for idx in range(0, len(move_up)):
            if move_up[idx] > 0 and move_up[idx] > move_down[idx]:
                pdm[idx] = move_up[idx]

            if move_down[idx] > 0 and move_down[idx] > move_up[idx]:
                ndm[idx] = move_down[idx]
        divisor = self.get_av_true_range(window)
        dividend = self.exp_moving_average(window, values=pdm)
        pdmi = 100*np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor != 0)
        dividend = self.exp_moving_average(window, values=ndm)
        ndmi = 100*np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor != 0)
        return pdmi, ndmi

    def get_adx_indicator(self, window=14, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        pdmi, ndmi = self.get_dmis(window=window, data_frame=data_frame)
        dividend = np.abs(pdmi-ndmi)
        divisor = pdmi+ndmi
        dx = 100*np.divide(dividend, pdmi+ndmi, out=np.zeros_like(dividend), where=divisor != 0)
        adx = self.exp_moving_average(window, values=dx)
        # print (np.max(pdmi))
        # print (np.max(ndmi))
        return adx, pdmi, ndmi

    def get_aroon(self, time_frame=20, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        highs = data_frame['High'].values.tolist()
        lows = data_frame['Low'].values.tolist()
        aroon_down = []
        aroon_up = []
        for idx in range(time_frame, len(highs)-1):
            aroon_up.append((highs[idx-time_frame:idx].index(max(highs[idx-time_frame:idx])))/time_frame*100)
            aroon_down.append((lows[idx-time_frame:idx].index(min(lows[idx-time_frame:idx])))/time_frame*100)
        return np.asarray(aroon_up), np.asarray(aroon_down)

    def get_cog(self, time_frame=20, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        values = data_frame['Adj Close'].values
        cog = []
        for idx in range(time_frame, len(values)+1):
            segment = values[idx-time_frame:idx]
            weights = np.asarray(range(time_frame, 0, -1))
            numerator = np.dot(segment, weights)
            den = np.sum(segment)

            cog.append(-numerator/den)

        return np.asarray(cog)

    def get_cmhof(self, time_frame=20, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        closes = data_frame['Adj Close'].values
        highs = data_frame['High'].values
        lows = data_frame['Low'].values
        volume = data_frame['Volume'].values
        mfm = []
        mfv = []
        for idx in range(time_frame, len(volume+1)):
            period_vol = np.sum(volume[idx-time_frame:idx])
            mfm_now = ((closes[idx]-lows[idx])-(highs[idx]-closes[idx]))/(highs[idx]-lows[idx])
            mfm.append(mfm_now)
            mfv.append(mfm_now*period_vol)

        chmf = []

        for idx in range(time_frame, len(volume+1)):
            period_vol = np.sum(volume[idx-time_frame:idx])
            tfs_mfv = sum(mfv[idx-time_frame:idx])
            chmf.append((tfs_mfv/period_vol))

        return np.asarray(chmf)

    def get_chaikin_volatility(self, ema_window=10, pc_window=10, data_frame=None):
        if data_frame is None:
            data_frame = self.data

        high_low = data_frame['High'].values - data_frame['Low'].values
        ema_high_low = self.exp_moving_average(ema_window, values=high_low)

        window = ema_window+pc_window
        chaikin_volatility = []
        for idx in range(window, ema_high_low.size):
            chaikin_volatility.append((ema_high_low[idx] - ema_high_low[idx-pc_window]) /
                                      abs(ema_high_low[idx-pc_window]))
        return np.asarray(chaikin_volatility)

    def get_cmo(self, time_frame=9, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        closes = data_frame['Adj Close'].values
        cmo = []
        for idx in range(time_frame, len(closes)):
            prices = closes[idx-time_frame:idx]
            ups = 0
            downs = 0
            for jdx in range(1, len(prices)):
                if prices[jdx] >= prices[jdx-1]:
                    ups += prices[jdx]-prices[jdx-1]
                else:
                    downs += prices[jdx-1] - prices[jdx]
            cmo.append((ups-downs)/(ups+downs)*100)
        return np.asarray(cmo)

    def get_cci(self, time_frame=20, window=14, data_frame=None):
        """Commodity Channel Index"""
        if data_frame is None:
            data_frame = self.data

        typical_prices = self.typical_prices(data_frame=data_frame)
        sm_typical_prices = self.moving_average(window, typical_prices)
        mds = []
        for idx in range(time_frame, len(sm_typical_prices)):
            mds.append(np.sum(np.abs(typical_prices[idx-time_frame:idx] - sm_typical_prices[idx-time_frame:idx])) /
                       len(typical_prices[idx-time_frame:idx]))

        difference = (typical_prices[window-1:]-sm_typical_prices)
        cci = np.divide(difference[time_frame:], (0.015*np.asarray(mds)))

        return cci

    def get_emv(self, window=14, factor=1e6, data_frame=None):
        """Ease of movement"""
        if data_frame is None:
            data_frame = self.data

        high_low = (data_frame['High'].values + data_frame['Low'].values)/2
        inst_emv = np.zeros_like(high_low)
        boxr = np.divide(data_frame['Volume'].values/factor, data_frame['High'].values-data_frame['Low'].values)
        for idx in range(1, len(high_low)):
            inst_emv[idx] = (high_low[idx] - high_low[idx-1])/boxr[idx]

        emv = self.moving_average(window=window, values=inst_emv)

        return emv

    def get_elder_force_index(self, window=14, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        inst_efi = np.multiply(np.diff(data_frame['Adj Close'].values), data_frame['Volume'].values[1:])
        efi = self.exp_moving_average(window=window, values=inst_efi)

        return efi

    @staticmethod
    def get_highest_high(values, time_frame):
        highest_high = np.zeros_like(values)
        for idx in range(time_frame, len(values)):
            highest_high[idx] = np.max(values[idx-time_frame:idx])
        return highest_high[time_frame:]

    @staticmethod
    def get_lowest_low(values, time_frame):
        lowest_low = np.zeros_like(values)
        for idx in range(time_frame, len(values)):
            lowest_low[idx] = np.max(values[idx - time_frame:idx])
        return lowest_low[time_frame:]

    def get_gapo_index(self, time_frame, data_frame=None):
        """Gopalakrishnan Range index"""
        import math
        if data_frame is None:
            data_frame = self.data

        highgest_high = self.get_highest_high(data_frame['Adj Close'].values, time_frame)
        lowest_low = self.get_lowest_low(data_frame['Adj Close'].values, time_frame)
        gapo = np.log(highgest_high-lowest_low)/math.log(time_frame)

        return gapo

    def get_hh_ll(self, time_frame, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        highgest_high = self.get_highest_high(data_frame['Adj Close'].values, time_frame)
        lowest_low = self.get_lowest_low(data_frame['Adj Close'].values, time_frame)

        return highgest_high, lowest_low

    def get_rate_of_change(self, time_frame=30, data_frame=None):
        if data_frame is None:
            data_frame = self.data
        prices = data_frame['Adj Close'].values
        roc = np.zeros_like(prices)
        for idx in range(time_frame, len(prices)):
            roc[idx] = (prices[idx] - prices[idx-time_frame])/prices[idx-time_frame]

        return roc


class StockVisualizer(object):
    """Receives a Grabber object or a dataframe"""

    def __init__(self, data, days=90):
        if isinstance(data, Grabber):
            self.grabber = data
            self.data = data.get_df()
            self.ticker = data.ticker
        else:
            raise Exception('You must provide a Grabber object or a pandas.DataFrame object')
        self.days = days
        self.analyzer = StockMetricsCalculator(self.data)
        self.fig = plt.figure()
        self.ax1 = plt.subplot2grid((6, 6), (1, 0), rowspan=4, colspan=6)
        self.ax3 = plt.subplot2grid((6, 6), (0, 0), rowspan=1, colspan=6, sharex=self.ax1)
        self.ax2 = plt.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=6, sharex=self.ax1)
        self.ax_twin = self.ax1.twinx()
        self.make_axes()
        self.df_date = self.build_dates_df()

    def build_dates_df(self):
        df_date = self.data.copy()
        df_date.reset_index(inplace=True)

        df_date.drop(['High', 'Close', 'Low', 'Adj Close', 'Open', 'Volume'], 1, inplace=True)

        df_date['Date'] = df_date['Date'].map(mdates.date2num)
        return df_date

    def update_data(self, start, end):
        self.grabber.update_data(start, end, self.ticker, self.grabber.path, 'yahoo')
        self.data = self.grabber.get_df()
        self.df_date = self.build_dates_df()

    def make_axes(self):
        self.ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='lower'))

        self.ax1.grid(True)
        plt.suptitle(self.ticker.upper())
        plt.subplots_adjust(left=0.12, bottom=0.14, right=0.94, top=0.94, wspace=0.052, hspace=0.052)

        for label in self.ax1.xaxis.get_ticklabels():
            label.set_rotation(90)
            label.set_color(self.fig.get_facecolor())

        for label in self.ax2.xaxis.get_ticklabels():
            label.set_rotation(45)

        for label in self.ax3.xaxis.get_ticklabels():
            label.set_visible(False)

    def get_graph(self):
        return self.fig

    def make_candle_stick(self, ma_list=None, ax=None):
        df_ohlc = self.data.copy()
        df_ohlc.drop(['Volume', 'Close'], 1, inplace=True)
        df_ohlc.rename(columns={'Adj Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)
        df_ohlc = df_ohlc.reindex(reversed(df_ohlc.columns), axis=1)
        df_ohlc.reset_index(inplace=True)
        df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

        if ax is None:
            ax = self.ax1
        ax.set_ylabel('Price')
        # self.ax1.yaxis.label.set_color('w')
        candlestick_ohlc(self.ax1, df_ohlc.values[-self.days:], width=0.75, colorup='g', colordown='r')
        if ma_list is not None:
            color = 0.0
            for each_ma in ma_list:
                ma_data = self.analyzer.moving_average(each_ma)
                ax.plot(df_ohlc['Date'][-self.days:], ma_data[-self.days:],
                        color=(color, color, color, 1-color), label=str(each_ma)+' MA')
                color += 0.25
            ax.legend(loc=9).set_alpha(0.4)
        self.annotation(ax, df_ohlc['Date'].values[-1] + 3, df_ohlc['close'].values[-1])

    def make_volume_plot(self, ax=None, overlay=False):
        df_volume = self.data.copy()
        df_volume.drop(['High', 'Close', 'Low', 'Adj Close', 'Open'], 1, inplace=True)
        df_volume.reset_index(inplace=True)
        df_volume['Date'] = df_volume['Date'].map(mdates.date2num)
        if overlay:
            if ax is None:
                ax = self.ax_twin
            ax.clear()
            ax.grid(False)
            ax.set_ylim(0, 3*max(df_volume['Volume'].values[-self.days:]))
        elif ax is None:
            ax = self.ax2
            ax.grid(True)
            ax.set_ylabel('Volume')

        ax.fill_between(df_volume['Date'][-self.days:], 0, df_volume['Volume'].values[-self.days:],
                        facecolor='#007983', alpha=0.5)
        ax.axes.yaxis.set_ticklabels([])

    @staticmethod
    def disappear_y_lables(ax):
        for label in ax.yaxis.get_ticklabels():
            label.set_visible(False)

    def make_bollinger_plot(self, ax=None, overlay=True):
        bollinger_up, bollinger_down, bollinger_central = self.analyzer.get_bollinger_bands()
        if overlay:
            ax = self.ax1.twinx()
            ax.grid(False)
        elif ax is None:
            ax = self.ax2
            ax.grid(True)
            ax.set_ylabel('Bollinger')

        ax.plot(self.df_date['Date'][-self.days:], bollinger_up[-self.days:], '#007983', linewidth=0.5)  # 007983'
        ax.plot(self.df_date['Date'][-self.days:], bollinger_central[-self.days:], '0.5', linewidth=0.5)  # 007983'
        ax.plot(self.df_date['Date'][-self.days:], bollinger_down[-self.days:], '#007983', linewidth=0.5)  # 007983'
        ax.fill_between(self.df_date['Date'][-self.days:], bollinger_down[-self.days:], bollinger_up[-self.days:],
                        facecolor='#007983', alpha=0.2)
        self.disappear_y_lables(ax)

    def make_macd_plot(self, ax=None):
        macd, signal_line = self.analyzer.macd_indicator(fast=12, slow=26)
        if ax is None:
            ax = self.ax3

        ax.plot(self.df_date['Date'][-self.days:], signal_line[-self.days:], 'r', linewidth=1)
        ax.plot(self.df_date['Date'][-self.days:], macd[-self.days:], 'g', linewidth=1)
        ax.fill_between(self.df_date['Date'][-self.days:], signal_line[-self.days:], macd[-self.days:],
                        where=(macd[-self.days:] <= signal_line[-self.days:]), facecolor='r', edgecolor='r', alpha=0.5)
        ax.fill_between(self.df_date['Date'][-self.days:], signal_line[-self.days:], macd[-self.days:],
                        where=(macd[-self.days:] >= signal_line[-self.days:]), facecolor='g', edgecolor='g', alpha=0.5)
        ax.set_ylabel('MACD')
        # legend = plt.legend(ncol=2, loc=9)
        # ax.legend(ncol=2, loc=9, prop={'size': 7})

    def make_ma_plot(self, ax=None):
        fast = self.analyzer.moving_average(12)
        slow = self.analyzer.moving_average(26)
        df_date = self.data.copy()
        df_date.drop(['High', 'Close', 'Low', 'Adj Close', 'Open', 'Volume'])
        df_date.reset_index(inplace=True)
        df_date['Date'] = df_date['Date'].map(mdates.date2num)
        # print (df_date.head())
        if ax is None:
            ax = self.ax3
        ax.plot(df_date['Date'][-self.days:], slow[-self.days:], 'r', linewidth=1)
        ax.plot(df_date['Date'][-self.days:], fast[-self.days:], 'g', linewidth=1)
        ax.fill_between(df_date['Date'][-self.days:],  slow[-self.days:],  fast[-self.days:],
                        where=(fast[-self.days:] <= slow[-self.days:]), facecolor='r', edgecolor='r', alpha=0.5)
        ax.fill_between(df_date['Date'][-self.days:], slow[-self.days:], fast[-self.days:],
                        where=(fast[-self.days:] >= slow[-self.days:]), facecolor='g', edgecolor='g', alpha=0.5)
        ax.set_ylabel('MAVGS')

    def make_rsi_plot(self, window=14, ax=None):

        if ax is None:
            ax = self.ax2

        rsi = self.analyzer.rsi_indicator(window)
        ax.plot(self.df_date['Date'][-self.days:], rsi[-self.days:])
        ax.axhline(70, color='r')  # plot(self.df_date['Date'][-self.days:], top_line, 'r')
        ax.axhline(30, color='g')
        ax.fill_between(self.df_date['Date'][-self.days:], 70, rsi[-self.days:], where=(rsi[-self.days:] >= 70),
                        facecolors='r', alpha=0.4)
        ax.fill_between(self.df_date['Date'][-self.days:], 30, rsi[-self.days:], where=(rsi[-self.days:] <= 30),
                        facecolors='g', alpha=0.4)
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI')
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))

    def make_adx_plot(self, window=14, ax=None):
        adx, pdmi, ndmi = self.analyzer.get_adx_indicator(window=window)
        if ax is None:
            ax = self.ax2

        ax.plot(self.df_date['Date'][-self.days:], adx[-self.days:], 'k', linewidth=1)
        ax.plot(self.df_date['Date'][-self.days:], pdmi[-self.days:], 'g', linewidth=1)
        ax.plot(self.df_date['Date'][-self.days:], ndmi[-self.days:], 'r', linewidth=1)
        ax.fill_between(self.df_date['Date'][-self.days:], ndmi[-self.days:], pdmi[-self.days:],
                        where=(pdmi[-self.days:] <= ndmi[-self.days:]), facecolor='r', edgecolor='r', alpha=0.5)
        ax.fill_between(self.df_date['Date'][-self.days:], ndmi[-self.days:], pdmi[-self.days:],
                        where=(pdmi[-self.days:] >= ndmi[-self.days:]), facecolor='g', edgecolor='g', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('ADX')


    def make_aroon_plot(self, window=14, ax=None, oscillator=True):
        aroon_up, aroon_down = self.analyzer.get_aroon(time_frame=window)
        if ax is None:
            ax = self.ax2

        if not oscillator:
            ax.plot(self.df_date['Date'][-self.days:], aroon_up[-self.days:], 'g', linewidth=1)
            ax.plot(self.df_date['Date'][-self.days:], aroon_down[-self.days:], 'r', linewidth=1)
            ax.set_ylim(0, 100)
        else:
            aroon_oscillator = (aroon_up - aroon_down)
            self.oscillator_plot(ax, self.df_date['Date'][-self.days:], aroon_oscillator[-self.days:], 0, 0)
            ax.set_ylim(-100, 100)
        ax.set_ylabel('AROON ' + str(window))

    @staticmethod
    def oscillator_plot(ax, x, line, positive, negative):
        ax.plot(x, line, 'k', linewidth=1)
        ax.fill_between(x, negative, line, where=(line <= negative), facecolor='r', edgecolor='r', alpha=0.5)
        ax.fill_between(x, positive, line, where=(line > positive), facecolor='g', edgecolor='g', alpha=0.5)

    @staticmethod
    def annotation(ax, x, y):
        bbox = dict(boxstyle="round4", fc="w", alpha=0.5)
        ax.annotate("{0:.2f}".format(y), (x, y),
                    xytext=(x + 2, y),
                    ha="center", va="center", size=7,
                    bbox=bbox)

    def make_atr_plot(self, window=14, ax=None):
        atr = self.analyzer.get_av_true_range(window=window)
        if ax is None:
            ax = self.ax2

        ax.plot(self.df_date['Date'][-self.days:], atr[-self.days:], 'b', linewidth=1)
        ax.fill_between(self.df_date['Date'][-self.days:], 0, atr[-self.days:],
                        facecolor='b', edgecolor='b', alpha=0.5)
        self.annotation(ax, self.df_date['Date'][-self.days:].values[-1], atr[-1])
        ax.set_ylim(min(atr[-self.days:])*0.95, max(atr[-self.days:])*1.05)
        ax.set_ylabel('ATR ' + str(window))

    def make_cog_plot(self, window=10, ax=None):
        cog = self.analyzer.get_cog(time_frame=window)
        if ax is None:
            ax = self.ax2

        ax.plot(self.df_date['Date'][-self.days:], cog[-self.days:], 'b', linewidth=1)
        ax.fill_between(self.df_date['Date'][-self.days:], 0, cog[-self.days:],
                        facecolor='b', edgecolor='b', alpha=0.5)

        ax.set_ylabel('CoG ' + str(window))
        ax.set_ylim(np.min(cog[-self.days:])*1.01, np.max(cog[-self.days:])*0.99)

    def make_chmof_plot(self, window=20, ax=None):
        if ax is None:
            ax = self.ax2
        chmof = self.analyzer.get_cmhof(time_frame=window)
        ax.plot(self.df_date['Date'][-self.days:], chmof[-self.days:], 'k', linewidth=0.5)
        ax.fill_between(self.df_date['Date'][-self.days:], 0, chmof[-self.days:],
                        where= chmof[-self.days:] > 0, facecolor='g', edgecolor='g', alpha=0.5)
        ax.fill_between(self.df_date['Date'][-self.days:], 0, chmof[-self.days:],
                        where=chmof[-self.days:] < 0, facecolor='r', edgecolor='r', alpha=0.5)
        ax.set_ylabel('CHMoF ' + str(window))

    def make_chaikin_vol_plot(self, ema_window=10, pc_window=10, ax=None):
        if ax is None:
            ax = self.ax2
        chv = self.analyzer.get_chaikin_volatility(ema_window=ema_window, pc_window=pc_window)
        ax.plot(self.df_date['Date'][-self.days:], chv[-self.days:], 'k', linewidth=0.5)
        ax.fill_between(self.df_date['Date'][-self.days:], 0, chv[-self.days:],
                        where=chv[-self.days:]>0, facecolor='g', edgecolor='g', alpha=0.5)
        ax.fill_between(self.df_date['Date'][-self.days:], 0, chv[-self.days:],
                        where=chv[-self.days:] < 0, facecolor='r', edgecolor='r', alpha=0.5)
        ax.set_ylabel('Chaikin ' + str(ema_window) + '-' + str(pc_window))

    def make_cmo_plot(self, window=10, ax=None):
        """Chade Momentum Indicator"""
        if ax is None:
            ax = self.ax2
        cmo = self.analyzer.get_cmo(time_frame=window)
        ax.plot(self.df_date['Date'][-self.days:], cmo[-self.days:], 'b', linewidth=0.5)
        ax.axhline(50, color='r')
        ax.axhline(0, color='k')
        ax.axhline(-50, color='g')
        ax.fill_between(self.df_date['Date'][-self.days:], -50, cmo[-self.days:],
                        where=cmo[-self.days:] < -50, facecolor='g', edgecolor='g', alpha=0.5)
        ax.fill_between(self.df_date['Date'][-self.days:], 50, cmo[-self.days:],
                        where=cmo[-self.days:] > 50, facecolor='r', edgecolor='r', alpha=0.5)
        ax.set_ylabel('CMO ' + str(window))
        ax.set_ylim(-110.1, 110)

    def make_cci_plot(self, window=20, time_frame=14, ax=None):
        """Chade Momentum Indicator"""
        if ax is None:
            ax = self.ax2
        cci = self.analyzer.get_cci(time_frame=time_frame, window=window)
        ax.plot(self.df_date['Date'][-self.days:], cci[-self.days:], 'b', linewidth=0.5)
        ax.axhline(100, color='r')
        ax.axhline(0, color='k')
        ax.axhline(-100, color='g')
        ax.fill_between(self.df_date['Date'][-self.days:], -100, cci[-self.days:],
                        where=cci[-self.days:] < -100, facecolor='g', edgecolor='g', alpha=0.5)
        ax.fill_between(self.df_date['Date'][-self.days:], 100, cci[-self.days:],
                        where=cci[-self.days:] > 100, facecolor='r', edgecolor='r', alpha=0.5)
        ax.set_ylabel('CCI ' + str(window))
        lim = np.max(np.abs(cci[-self.days:]))
        ax.set_ylim(-lim*1.1, lim*1.1)

    def make_emv_plot(self, window=14, factor=1e6, ax=None):
        """Chade Momentum Indicator"""
        if ax is None:
            ax = self.ax2
        emv = self.analyzer.get_emv(factor=factor, window=window)
        ax.plot(self.df_date['Date'][-self.days:], emv[-self.days:], 'b', linewidth=0.5)
        ax.axhline(0, color='k')
        ax.set_ylabel('EMV ' + str(window))

    def make_efi_plot(self, window=14, ax=None):
        """elder force index"""
        if ax is None:
            ax = self.ax2
        efi = self.analyzer.get_elder_force_index(window=window)
        ax.plot(self.df_date['Date'][-self.days:], efi[-self.days:], 'b', linewidth=0.5)
        # ax.axhline(0, color='k')
        ax.set_ylabel('EFI ' + str(window))

    def make_gapo_plot(self, window=14, ax=None):
        """elder force index"""
        if ax is None:
            ax = self.ax2
        gapo = self.analyzer.get_gapo_index(time_frame=window)
        ax.plot(self.df_date['Date'][-self.days:], gapo[-self.days:], 'b', linewidth=0.5)
        # ax.axhline(0, color='k')
        ax.set_ylabel('GAPO ' + str(window))

    def make_roc_plot(self, window=30, ax=None):
        if ax is None:
            ax = self.ax2
        roc = self.analyzer.get_rate_of_change(time_frame=window)

        self.oscillator_plot(ax, self.df_date['Date'][-self.days:], roc[-self.days:], 0, 0)
        ax.set_ylabel('RoC ' + str(window))

    def show(self):
        plt.show()


if __name__ == '__main__':
    from time import sleep
    stocks_to_watch = ['TQQQ', 'NVDA', 'WB', 'TSLA', 'BPMC', 'MZOR', 'SHOP.TO']
    # my_data = get_data_from_alpha('TQQQ')
    adxs = []
    for each_stock in stocks_to_watch:
        my_grabber = Grabber([2015, 1, 1], [2018, 3, 13], each_stock)
        print(each_stock)
        my_analyzer = StockMetricsCalculator(my_grabber)
        if my_analyzer.get_rate_of_change()[-1] > 0:
            adx, pdmi, ndmi = my_analyzer.get_adx_indicator()
            if pdmi[-1] > ndmi[-1]:
                adxs.append(adx[-1])
        sleep(5)
    indexes = sorted(adxs)
    stocks_to_plot = range(0, len(stocks_to_watch)) #stocks_to_watch[adxs.index(indexes[-1]), adxs.index(indexes[-2])]
    for stock in stocks_to_plot:
        my_grabber = Grabber([2015, 1, 1], [2018, 3, 23], stocks_to_watch[stock])
        my_visualizer = StockVisualizer(my_grabber, 60)
        my_visualizer.make_candle_stick(ma_list=[50])
        my_visualizer.make_volume_plot(overlay=True)
        my_visualizer.make_macd_plot()
        my_visualizer.make_roc_plot()
        # my_visualizer.make_atr_plot()
        # my_visualizer.make_gapo_plot()
        my_visualizer.show()
        sleep(3)
        # my_visualizer = Visualizer('data')
        # end  = dt.datetime(2016,12,31)
        # df = web.DataReader('TSLA', 'yahoo', start, end)
