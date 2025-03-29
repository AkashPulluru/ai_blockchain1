import requests
import pandas as pd

markets_api_url = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMARMAD"
trades_api_url_template = "https://api.elections.kalshi.com/trade-api/v2/markets/trades?limit=1000&ticker={}"

headers = {
    "accept": "application/json",
}

def fetch_markets():
    response = requests.get(markets_api_url, headers=headers)
    response.raise_for_status()
    return response.json()

def fetch_trades(ticker):
    url = trades_api_url_template.format(ticker)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get('trades', [])

def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Data successfully saved to {filename}")

def main():
    series_ticker = 'KXMARMAD'
    markets_data = fetch_markets()

    markets = markets_data.get('markets', [])
    if not markets:
        print(f"No markets found for series ticker '{series_ticker}'.")
        return

    all_trades = []
    for market in markets:
        ticker = market.get('ticker')
        print(f"Fetching trades for ticker: {ticker}")
        trades = fetch_trades(ticker)
        for trade in trades:
            trade['market_ticker'] = ticker
        all_trades.extend(trades)

    if not all_trades:
        print("No trades data found.")
        return

    save_to_excel(all_trades, f'{series_ticker}_trades.xlsx')

if __name__ == "__main__":
    main()
