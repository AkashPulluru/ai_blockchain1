import requests
import pandas as pd
from xlsxwriter import Workbook

markets_api_url_template = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={}&limit=1000"
trades_api_url_template = "https://api.elections.kalshi.com/trade-api/v2/markets/trades?limit=1000&ticker={}"

headers = {"accept": "application/json"}

def fetch_markets(series_ticker):
    all_markets = []
    cursor = None

    while True:
        url = markets_api_url_template.format(series_ticker)
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
        fee_percentage = total_fees / count if count > 0 else 0

        trade['maker_fees'] = maker_fees
        trade['fees'] = fees
        trade['total_fees'] = total_fees
        trade['fee_percentage'] = fee_percentage

    return trades


def compute_trade_analytics(trades_df):
    trades_df['yes_weighted'] = trades_df['yes_price'] * trades_df['count']
    trades_df['no_weighted'] = trades_df['no_price'] * trades_df['count']

    analytics = trades_df.groupby('market_ticker').agg(
        total_trades=('count', 'sum'),
        weighted_yes_price=('yes_weighted', 'sum'),
        weighted_no_price=('no_weighted', 'sum'),
        total_count=('count', 'sum'),
        total_maker_fees=('maker_fees', 'sum'),
        total_fees=('fees', 'sum'),
        total_combined_fees=('total_fees', 'sum'),
    ).reset_index()

    # Calculate final weighted averages
    analytics['average_yes_price'] = analytics['weighted_yes_price'] / analytics['total_count']
    analytics['average_no_price'] = analytics['weighted_no_price'] / analytics['total_count']

    # Drop intermediate columns if desired
    analytics = analytics.drop(columns=['weighted_yes_price', 'weighted_no_price', 'total_count'])

    return analytics


def save_to_excel_with_analytics(data, analytics, filename, sheet_size=1000000):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Write main data
        for i in range(0, len(data), sheet_size):
            df_chunk = pd.DataFrame(data[i:i + sheet_size])
            sheet_name = f'Sheet_{(i // sheet_size) + 1}'
            df_chunk.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Data chunk {sheet_name} written successfully.")

        # Write analytics
        analytics.to_excel(writer, sheet_name='Summary_Analytics', index=False)
        print("Analytics summary sheet written successfully.")

    print(f"All data successfully saved to {filename}")

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data successfully saved to {filename}")

def main():
    series_tickers = ['KXNBAGAME', 'PRES']

    for series_ticker in series_tickers:
        markets = fetch_markets(series_ticker)

        if not markets:
            print(f"No markets found for series ticker '{series_ticker}'.")
            continue

        all_trades = []
        for market in markets:
            ticker = market.get('ticker')
            print(f"Fetching all trades for ticker: {ticker}")
            trades = fetch_all_trades(ticker)
            for trade in trades:
                trade['market_ticker'] = ticker
            all_trades.extend(trades)

        if not all_trades:
            print(f"No trades data found for ticker '{series_ticker}'.")
            continue

        all_trades = add_fees_columns(all_trades)
        trades_df = pd.DataFrame(all_trades)

        # Compute analytics
        analytics_df = compute_trade_analytics(trades_df)

        excel_filename = f'{series_ticker}_trades.xlsx'
        csv_filename = f'{series_ticker}_trades.csv'

        save_to_excel_with_analytics(all_trades, analytics_df, excel_filename)
        save_to_csv(all_trades, csv_filename)

if __name__ == "__main__":
    main()