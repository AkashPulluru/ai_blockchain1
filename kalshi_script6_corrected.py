import requests
import pandas as pd

markets_api_url_template = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMARMAD&limit=1000"  # Cursor added conditionally
trades_api_url_template = "https://api.elections.kalshi.com/trade-api/v2/markets/trades?limit=1000&ticker={}"  # Cursor added conditionally

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

def save_to_excel(data, filename, sheet_size=1000000):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for i in range(0, len(data), sheet_size):
            df_chunk = pd.DataFrame(data[i:i + sheet_size])
            sheet_name = f'Sheet_{(i // sheet_size) + 1}'
            df_chunk.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Data chunk {sheet_name} written successfully.")

    print(f"All data successfully saved to {filename}")

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

    save_to_excel(all_trades, f'{series_ticker}_trades.xlsx')

if __name__ == "__main__":
    main()