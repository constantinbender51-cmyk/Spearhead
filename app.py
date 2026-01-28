import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random
import http.server
import socketserver
import warnings
import requests
import threading
import time
import json
import urllib.parse
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms

# --- Configuration ---
BASE_DATA_URL = "https://ohlcendpoint.up.railway.app/data"
PORT = 8080
N_LINES = 200
POPULATION_SIZE = 40
GENERATIONS = 10
RISK_FREE_RATE = 0.0

# Ranges
STOP_PCT_RANGE = (0.001, 0.02)   # 0.1% to 2%
PROFIT_PCT_RANGE = (0.0004, 0.05) # 0.04% to 5%

warnings.filterwarnings("ignore")

# Asset List Mapping
ASSETS = [
    {"symbol": "BTC", "pair": "BTCUSDT", "csv": "btc1m.csv"},
    {"symbol": "ETH", "pair": "ETHUSDT", "csv": "eth1m.csv"},
    {"symbol": "XRP", "pair": "XRPUSDT", "csv": "xrp1m.csv"},
    {"symbol": "SOL", "pair": "SOLUSDT", "csv": "sol1m.csv"},
    {"symbol": "DOGE", "pair": "DOGEUSDT", "csv": "doge1m.csv"},
    {"symbol": "ADA", "pair": "ADAUSDT", "csv": "ada1m.csv"},
    {"symbol": "BCH", "pair": "BCHUSDT", "csv": "bch1m.csv"},
    {"symbol": "LINK", "pair": "LINKUSDT", "csv": "link1m.csv"},
    {"symbol": "XLM", "pair": "XLMUSDT", "csv": "xlm1m.csv"},
    {"symbol": "SUI", "pair": "SUIUSDT", "csv": "sui1m.csv"},
    {"symbol": "AVAX", "pair": "AVAXUSDT", "csv": "avax1m.csv"},
    {"symbol": "LTC", "pair": "LTCUSDT", "csv": "ltc1m.csv"},
    {"symbol": "HBAR", "pair": "HBARUSDT", "csv": "hbar1m.csv"},
    {"symbol": "SHIB", "pair": "SHIBUSDT", "csv": "shib1m.csv"},
    {"symbol": "TON", "pair": "TONUSDT", "csv": "ton1m.csv"},
]

# --- Global State Storage ---
GLOBAL_STATE = {}
STATE_LOCK = threading.Lock()

# --- JSON Encoder for NumPy ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- 1. DEAP Initialization ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 2. Precise Data Ingestion ---
def get_data(csv_filename):
    url = f"{BASE_DATA_URL}/{csv_filename}"
    print(f"Downloading data from {url}...")
    try:
        df = pd.read_csv(url)
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'datetime' in df.columns:
            df['dt'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        else:
            raise ValueError("No valid date column found.")

        df.dropna(subset=['dt', 'open', 'high', 'low', 'close'], inplace=True)
        df.set_index('dt', inplace=True)
        df.sort_index(inplace=True)

        # Resample for Optimization Speed
        df_1h = df.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        if len(df_1h) < 100: raise ValueError("Data insufficient.")

        split_idx = int(len(df_1h) * 0.85)
        return df_1h.iloc[:split_idx], df_1h.iloc[split_idx:]

    except Exception as e:
        print(f"CRITICAL DATA ERROR for {csv_filename}: {e}")
        return None, None

# --- 3. Strategy Logic ---
def run_backtest(df, stop_pct, profit_pct, lines, detailed_log_trades=0):
    closes = df['close'].values
    times = df.index
    equity = 10000.0
    equity_curve = [equity]
    position = 0          
    entry_price = 0.0
    trades = []
    
    lines = np.sort(lines)
    
    for i in range(1, len(df)):
        current_c = closes[i]
        prev_c = closes[i-1]
        ts = str(times[i])

        if position != 0:
            sl_hit = False
            tp_hit = False
            exit_price = 0.0
            reason = ""

            if position == 1: 
                sl_price = entry_price * (1 - stop_pct)
                tp_price = entry_price * (1 + profit_pct)
                if current_c <= sl_price: sl_hit=True; exit_price=sl_price 
                elif current_c >= tp_price: tp_hit=True; exit_price=tp_price
            elif position == -1: 
                sl_price = entry_price * (1 + stop_pct)
                tp_price = entry_price * (1 - profit_pct)
                if current_c >= sl_price: sl_hit=True; exit_price=sl_price
                elif current_c <= tp_price: tp_hit=True; exit_price=tp_price
            
            if sl_hit or tp_hit:
                if position == 1: pn_l = (exit_price - entry_price) / entry_price
                else: pn_l = (entry_price - exit_price) / entry_price
                
                equity *= (1 + pn_l)
                reason = "SL" if sl_hit else "TP"
                trades.append({'time': ts, 'type': 'Exit', 'price': exit_price, 'pnl': pn_l, 'equity': equity, 'reason': reason})
                position = 0
                equity_curve.append(equity)
                continue 

        if position == 0:
            p_min, p_max = min(prev_c, current_c), max(prev_c, current_c)
            idx_start = np.searchsorted(lines, p_min, side='right')
            idx_end = np.searchsorted(lines, p_max, side='right')
            crossed_lines = lines[idx_start:idx_end]
            
            if len(crossed_lines) > 0:
                target_line = 0.0
                new_pos = 0
                if current_c > prev_c: target_line, new_pos = crossed_lines[0], -1
                elif current_c < prev_c: target_line, new_pos = crossed_lines[-1], 1
                
                if new_pos != 0:
                    position = new_pos
                    entry_price = target_line
                    trades.append({'time': ts, 'type': 'Short' if position == -1 else 'Long', 'price': entry_price, 'pnl': 0, 'equity': equity, 'reason': 'Entry'})

        equity_curve.append(equity)

    return equity_curve, trades

def calculate_sharpe(equity_curve):
    if len(equity_curve) < 2: return -999.0
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0: return -999.0
    return np.sqrt(8760) * (returns.mean() / returns.std())

# --- 4. Genetic Algorithm ---
def setup_toolbox(min_price, max_price, df_train):
    toolbox = base.Toolbox()
    toolbox.register("attr_stop", random.uniform, STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    toolbox.register("attr_profit", random.uniform, PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    toolbox.register("attr_line", random.uniform, min_price, max_price)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_stop, toolbox.attr_profit) + (toolbox.attr_line,)*N_LINES, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_genome, df_train=df_train)
    toolbox.register("mate", tools.cxTwoPoint) 
    toolbox.register("mutate", mutate_custom, indpb=0.1, min_p=min_price, max_p=max_price)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

def evaluate_genome(individual, df_train):
    stop_pct = np.clip(individual[0], STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    profit_pct = np.clip(individual[1], PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    lines = np.array(individual[2:])
    eq_curve, _ = run_backtest(df_train, stop_pct, profit_pct, lines)
    return (calculate_sharpe(eq_curve),)

def mutate_custom(individual, indpb, min_p, max_p):
    if random.random() < indpb: individual[0] += random.gauss(0, 0.005)
    if random.random() < indpb: individual[1] += random.gauss(0, 0.005)
    for i in range(2, len(individual)):
        if random.random() < (indpb / 10.0): individual[i] += random.gauss(0, (max_p - min_p) * 0.01)
    return individual,

# --- 5. Live Forward Test Logic ---
def fetch_binance_candle(symbol_pair):
    try:
        url = "https://api.binance.com/api/v3/klines"
        r = requests.get(url, params={'symbol': symbol_pair, 'interval': '1m', 'limit': 2}, timeout=5)
        r.raise_for_status()
        data = r.json()
        if len(data) >= 2:
            return pd.to_datetime(data[-2][0], unit='ms'), float(data[-2][4])
        return None, None
    except Exception as e:
        print(f"[{symbol_pair}] API Error: {e}")
        return None, None

def live_trading_daemon(symbol, pair, best_ind, initial_equity, start_price):
    stop_pct, profit_pct = best_ind[0], best_ind[1]
    lines = np.sort(np.array(best_ind[2:]))
    
    live_equity = initial_equity
    live_position = 0 
    live_entry_price = 0.0
    prev_close = start_price
    
    # Init empty lists in global state
    with STATE_LOCK:
        GLOBAL_STATE[symbol]['live_logs'] = []
        GLOBAL_STATE[symbol]['live_trades'] = []
    
    print(f"[{symbol}] Daemon Started.")
    
    while True:
        now = datetime.now()
        next_run = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
        if next_run <= now: next_run += timedelta(minutes=1)
        time.sleep((next_run - now).total_seconds())
        
        ts, current_c = fetch_binance_candle(pair)
        if current_c is None: continue
            
        # Logging
        idx = np.searchsorted(lines, current_c)
        val_below = lines[idx-1] if idx > 0 else -1
        val_above = lines[idx] if idx < len(lines) else -1
        
        act_sl = live_entry_price * (1 - stop_pct) if live_position == 1 else (live_entry_price * (1 + stop_pct) if live_position == -1 else 0)
        act_tp = live_entry_price * (1 + profit_pct) if live_position == 1 else (live_entry_price * (1 - profit_pct) if live_position == -1 else 0)
            
        log_entry = {
            "timestamp": str(ts), "price": current_c, 
            "nearest_below": val_below, "nearest_above": val_above,
            "position": "LONG" if live_position == 1 else ("SHORT" if live_position == -1 else "FLAT"),
            "active_sl": act_sl, "active_tp": act_tp, "equity": live_equity
        }
        
        with STATE_LOCK:
            if symbol in GLOBAL_STATE:
                GLOBAL_STATE[symbol]['live_logs'].insert(0, log_entry)
                GLOBAL_STATE[symbol]['live_logs'] = GLOBAL_STATE[symbol]['live_logs'][:50] # Keep last 50
        
        # Trading Logic
        if live_position != 0:
            sl_hit, tp_hit = False, False
            exit_price = 0.0
            
            if live_position == 1:
                if current_c <= act_sl: sl_hit = True; exit_price = act_sl
                elif current_c >= act_tp: tp_hit = True; exit_price = act_tp
            elif live_position == -1:
                if current_c >= act_sl: sl_hit = True; exit_price = act_sl
                elif current_c <= act_tp: tp_hit = True; exit_price = act_tp
            
            if sl_hit or tp_hit:
                pn_l = (exit_price - live_entry_price)/live_entry_price if live_position == 1 else (live_entry_price - exit_price)/live_entry_price
                live_equity *= (1 + pn_l)
                
                trade = {
                    'time': str(ts), 'type': 'Exit', 'price': exit_price, 
                    'pnl': pn_l, 'equity': live_equity, 'reason': "SL" if sl_hit else "TP"
                }
                
                with STATE_LOCK:
                     if symbol in GLOBAL_STATE:
                        GLOBAL_STATE[symbol]['live_trades'].insert(0, trade)
                
                live_position = 0
                prev_close = current_c
                continue

        if live_position == 0:
            p_min, p_max = min(prev_close, current_c), max(prev_close, current_c)
            idx_start = np.searchsorted(lines, p_min, side='right')
            idx_end = np.searchsorted(lines, p_max, side='right')
            crossed_lines = lines[idx_start:idx_end]
            
            if len(crossed_lines) > 0:
                target_line = 0.0
                new_pos = 0
                if current_c > prev_close: target_line, new_pos = crossed_lines[0], -1
                elif current_c < prev_close: target_line, new_pos = crossed_lines[-1], 1
                
                if new_pos != 0:
                    live_position = new_pos
                    live_entry_price = target_line
                    trade = {
                        'time': str(ts), 'type': 'Short' if live_position == -1 else 'Long', 
                        'price': live_entry_price, 'pnl': 0, 'equity': live_equity, 'reason': 'Entry'
                    }
                    with STATE_LOCK:
                         if symbol in GLOBAL_STATE:
                            GLOBAL_STATE[symbol]['live_trades'].insert(0, trade)

        prev_close = current_c

# --- 6. Server Handler ---
class ResultsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/api/assets':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            safe_assets = [{'symbol': a['symbol'], 'pair': a['pair']} for a in ASSETS]
            self.wfile.write(json.dumps(safe_assets).encode('utf-8'))
            
        elif path == '/api/status':
            symbol = query.get('symbol', [None])[0]
            if symbol and symbol in GLOBAL_STATE:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                with STATE_LOCK:
                    data = GLOBAL_STATE[symbol]
                    self.wfile.write(json.dumps(data, cls=NumpyEncoder).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Symbol not found or initializing"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

# --- 7. Main Execution ---
def process_asset(asset_config):
    sym = asset_config['symbol']
    
    with STATE_LOCK:
        GLOBAL_STATE[sym] = {"status": "Initializing"}

    train_df, test_df = get_data(asset_config['csv'])
    if train_df is None: return

    min_p, max_p = train_df['close'].min(), train_df['close'].max()
    toolbox = setup_toolbox(min_p, max_p, train_df)

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    
    # FIXED: Added halloffame=hof argument
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, halloffame=hof, verbose=False)
    
    # Safety check if GA failed completely
    if len(hof) == 0:
        print(f"[{sym}] GA Error: No individuals in Hall of Fame.")
        with STATE_LOCK:
            GLOBAL_STATE[sym]["status"] = "Error"
        return

    best_ind = hof[0]
    
    # Backtest for curves
    test_curve, test_trades = run_backtest(test_df, best_ind[0], best_ind[1], np.array(best_ind[2:]))

    # Save Initial State
    with STATE_LOCK:
        GLOBAL_STATE[sym] = {
            "status": "Running",
            "params": {
                "stop_pct": best_ind[0],
                "profit_pct": best_ind[1],
                "lines": list(best_ind[2:])
            },
            "equity_curve": test_curve[-500:], # Last 500 points for chart
            "test_trades": test_trades[-20:],
            "live_logs": [],
            "live_trades": []
        }

    # Start Live Thread
    t = threading.Thread(
        target=live_trading_daemon, 
        args=(sym, asset_config['pair'], best_ind, 10000.0, test_df['close'].iloc[-1]),
        daemon=True
    )
    t.start()

if __name__ == "__main__":
    for asset in ASSETS:
        process_asset(asset)
    
    with socketserver.TCPServer(("", PORT), ResultsHandler) as httpd:
        print(f"Serving JSON API at port {PORT}")
        httpd.serve_forever()
