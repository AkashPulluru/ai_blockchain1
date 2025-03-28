import requests
import pandas as pd

# Explicit API endpoint with the series ticker directly in URL
api_url = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMARMAD"
headers = {
    "accept": "application/json",
    # Include authorization header if necessary:
    # "Authorization": "Bearer YOUR_API_KEY"
}

# Fetch markets data without additional params since it's explicitly included in URL
def fetch_markets():
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    return response.json()

# Save the retrieved data to Excel
def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Data successfully saved to {filename}")

def main():
    series_ticker = 'KXMARMAD'
    markets_data = fetch_markets()

    markets = markets_data.get('markets', [])
    if not markets:
        print(f"No markets found for the series ticker '{series_ticker}'.")
        return

    save_to_excel(markets, f'{series_ticker}_markets.xlsx')

if __name__ == "__main__":
    main()
