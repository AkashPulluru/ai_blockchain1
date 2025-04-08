import requests
import pandas as pd
from xlsxwriter import Workbook

markets_api_url_template = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMARMAD&limit=1000"
trades_api_url_template = "https://api.elections.kalshi.com/trade-api/v2/markets/trades?limit=1000&ticker={}"

headers = {
    "accept": "application/json",
}

def fetch_markets():
    all_markets = []
    cursor = None

    while True:
        url = markets_api_url_template
        if cursor:
            url += f"&cursor={cursor}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        markets = data.get('markets', [])
        all_markets.extend(markets)

        cursor = data.get('cursor')
        if not cursor or len(markets) == 0:
            break

    return all_markets

def fetch_all_trades(ticker):
    all_trades = []
    cursor = None

    while True:
        url = trades_api_url_template.format(ticker)
        if cursor:
            url += f"&cursor={cursor}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        trades = data.get('trades', [])
        all_trades.extend(trades)

        cursor = data.get('cursor')
        if not cursor or len(trades) == 0:
            break

    return all_trades

def add_fees_columns(trades):
    for trade in trades:
        count = trade.get('count', 0)
        yes_price = trade.get('yes_price', 0)
        no_price = trade.get('no_price', 0)

        maker_fees = count * 0.0025
        fees = count * 0.01 * yes_price * 0.01 * no_price * 0.07 * 2
        total_fees = maker_fees + fees

        trade['maker_fees'] = maker_fees
        trade['fees'] = fees
        trade['total_fees'] = total_fees

    return trades

def save_to_excel(data, filename, sheet_size=1000000):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for i in range(0, len(data), sheet_size):
            df_chunk = pd.DataFrame(data[i:i + sheet_size])
            sheet_name = f'Sheet_{(i // sheet_size) + 1}'
            df_chunk.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Data chunk {sheet_name} written successfully.")

    print(f"All data successfully saved to {filename}")

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data successfully saved to {filename}")

def main():
    series_ticker = 'KXMARMAD'
    markets = fetch_markets()

    if not markets:
        print(f"No markets found for series ticker '{series_ticker}'.")
        return

    all_trades = []
    for market in markets:
        ticker = market.get('ticker')
        print(f"Fetching all trades for ticker: {ticker}")
        trades = fetch_all_trades(ticker)
        for trade in trades:
            trade['market_ticker'] = ticker
        all_trades.extend(trades)

    if not all_trades:
        print("No trades data found.")
        return

    all_trades = add_fees_columns(all_trades)

    excel_filename = f'{series_ticker}_trades.xlsx'
    csv_filename = f'{series_ticker}_trades.csv'

    save_to_excel(all_trades, excel_filename)
    save_to_csv(all_trades, csv_filename)

if __name__ == "__main__":
    main()
