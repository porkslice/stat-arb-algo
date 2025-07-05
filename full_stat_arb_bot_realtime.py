import time
import csv
import os
from datetime import datetime
from ib_insync import IB, Forex, util, MarketOrder
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pykalman import KalmanFilter

USE_KALMAN = False
Z_BAND = 0.15
REFRESH_INTERVAL = 1  # seconds

CURRENCY_PAIRS = [
    ("EURUSD", "GBPUSD"),
    ("AUDUSD", "NZDUSD"),
    ("EURUSD", "USDCHF"),
    ("AUDCAD", "NZDCAD"),
    ("USDHKD", "USDJPY"),
    ("USDSEK", "USDNOK"),
    ("USDSEK", "USDDKK"),
]

ALLOCATION_MAP = {
    1: 0.10,
    2: 0.08,
    3: 0.06,
    4: 0.05,
    5: 0.04,
    6: 0.03,
    7: 0.02,
    8: 0.02,
    9: 0.02
}
STD_LEVELS = [round(1 + i * 0.15, 2) for i in range(54)]  # From 1 to 9 in 0.15 steps
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=22)
print("✅ Connected to IBKR")

LOG_FILE = 'stat_arb_trade_log.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'pair', 'direction', 'zscore_level', 'zscore_value',
            'entry_price_base', 'entry_price_hedge', 'size_base', 'size_hedge',
            'exit_price_base', 'exit_price_hedge', 'pnl_base', 'pnl_hedge', 'total_pnl'
        ])

def log_trade(**kwargs):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            kwargs['timestamp'], kwargs['pair'], kwargs['direction'], kwargs['zscore_level'], kwargs['zscore_value'],
            kwargs['entry_price_base'], kwargs['entry_price_hedge'],
            kwargs['size_base'], kwargs['size_hedge'],
            kwargs['exit_price_base'], kwargs['exit_price_hedge'],
            kwargs['pnl_base'], kwargs['pnl_hedge'], kwargs['total_pnl']
        ])

def fetch_historical_prices(symbol):
    contract = Forex(symbol)
    ib.qualifyContracts(contract)
    bars = ib.reqHistoricalData(contract, '', '2 D', '1 hour', 'MIDPOINT', useRTH=False, formatDate=1)
    df = util.df(bars)[['date', 'close']]
    df.set_index('date', inplace=True)
    return df

def run_ols(y, x):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model

def run_kalman(y, x):
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, initial_state_mean=[0, 0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=1.0,
                      transition_covariance=trans_cov)
    state_means, _ = kf.filter(y.values)
    return pd.Series(state_means[:, 0], index=y.index), pd.Series(state_means[:, 1], index=y.index)

def adf_test(spread):
    result = adfuller(spread)
    return result[0] < result[4]['5%']

def get_price(ticker):
    return ticker.last or ticker.close or ticker.bid or ticker.ask or None

open_positions = {}

try:
    while True:
        account_value = 1_000_000
        for item in ib.accountSummary():
            if item.tag == 'NetLiquidation':
                account_value = float(item.value)
                break

        for base, hedge in CURRENCY_PAIRS:
            print(f"📡 Checking {base}-{hedge}")
            df_base = fetch_historical_prices(base)
            df_hedge = fetch_historical_prices(hedge)
            df = pd.merge(df_base, df_hedge, left_index=True, right_index=True, suffixes=(f"_{base}", f"_{hedge}"))
            y = df[f'close_{base}']
            x = df[f'close_{hedge}']
            if USE_KALMAN:
                hedge_series, intercept_series = run_kalman(y, x)
                hedge_ratio = hedge_series.iloc[-1]
                intercept = intercept_series.iloc[-1]
                spread = y - (hedge_series * x + intercept_series)
            else:
                model = run_ols(y, x)
                hedge_ratio = model.params.iloc[1]
                intercept = model.params.iloc[0]
                spread = y - (hedge_ratio * x + intercept)

            if not adf_test(spread):
                continue

            spread_mean = spread.mean()
            spread_std = spread.std()

            ticker_base = ib.reqMktData(Forex(base), "", False, False)
            ticker_hedge = ib.reqMktData(Forex(hedge), "", False, False)
            ib.sleep(0.5)

            price_base = get_price(ticker_base)
            price_hedge = get_price(ticker_hedge)
            if price_base is None or price_hedge is None:
                continue

            if price_base is None or price_hedge is None or np.isnan(price_base) or np.isnan(price_hedge):
                print(f"⚠️ Skipping {base}-{hedge}: Invalid live prices (base: {price_base}, hedge: {price_hedge})")
                continue

            spread_live = price_base - (hedge_ratio * price_hedge + intercept)

            if np.isnan(spread_live):
                print(f"⚠️ Spread live calculation returned NaN for {base}-{hedge}")
                continue

            z = (spread_live - spread_mean) / spread_std

            pair_key = f"{base}-{hedge}"

            if spread_std == 0 or np.isnan(z):
             print(f"⚠️ Z-score computation failed for {pair_key} (spread_std={spread_std}, z={z})")
             continue

            print(f"🔍 {pair_key} Z-Score: {z:.4f} | Live Spread: {spread_live:.5f} | Mean: {spread_mean:.5f} | Std: {spread_std:.5f}")


            if pair_key in open_positions and abs(z) <= Z_BAND:
                pos = open_positions.pop(pair_key)
                exit_base = price_base
                exit_hedge = price_hedge
                pnl_base = (exit_base - pos['entry_price_base']) * (-1 if pos['direction'] == 'short' else 1) * pos['size_base']
                pnl_hedge = (exit_hedge - pos['entry_price_hedge']) * (1 if pos['direction'] == 'short' else -1) * pos['size_hedge']
                log_trade(
                    timestamp=datetime.now(), pair=pair_key, direction=pos['direction'],
                    zscore_level=pos['zscore_level'], zscore_value=z,
                    entry_price_base=pos['entry_price_base'], entry_price_hedge=pos['entry_price_hedge'],
                    size_base=pos['size_base'], size_hedge=pos['size_hedge'],
                    exit_price_base=exit_base, exit_price_hedge=exit_hedge,
                    pnl_base=pnl_base, pnl_hedge=pnl_hedge, total_pnl=pnl_base + pnl_hedge
                )
                print(f"📤 Closed {pair_key} | PnL: {pnl_base + pnl_hedge:.2f}")
                continue

            for level, alloc in ALLOCATION_MAP.items():
                lower = level - Z_BAND
                upper = level + Z_BAND
                if pair_key in open_positions:
                    break
                if abs(z) >= lower and abs(z) <= upper:
                    direction = 'short' if z > 0 else 'long'
                    notional = account_value * alloc
                    size_base = round(notional / price_base, 5)
                    size_hedge = round((notional * hedge_ratio) / price_hedge, 5)
                    ib.qualifyContracts(Forex(base), Forex(hedge))
                    ib.placeOrder(Forex(base), MarketOrder('SELL' if direction == 'short' else 'BUY', size_base))
                    ib.placeOrder(Forex(hedge), MarketOrder('BUY' if direction == 'short' else 'SELL', size_hedge))
                    open_positions[pair_key] = {
                        'direction': direction,
                        'zscore_level': level,
                        'entry_price_base': price_base,
                        'entry_price_hedge': price_hedge,
                        'size_base': size_base,
                        'size_hedge': size_hedge
                    }
                    print(f"🟢 Entered {direction.upper()} {pair_key} at Z={z:.2f} | Level {level}")
                    break

        time.sleep(REFRESH_INTERVAL)

except KeyboardInterrupt:
    print("🛑 Bot manually stopped.")
finally:
    ib.disconnect()
    print("🔌 Disconnected")




    spread_live = price_base - (hedge_ratio * price_hedge + intercept)
z = (spread_live - spread_mean) / spread_std
