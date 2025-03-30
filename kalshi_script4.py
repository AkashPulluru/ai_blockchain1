import requests
import pandas as pd

# Series ticker
series_ticker = 'KXMARMAD'

# API endpoint
events_api_url = f"https://api.elections.kalshi.com/trade-api/v2/events?series_ticker={series_ticker}"

headers = {
    "accept": "application/json",
}

# Function to fetch events
def fetch_events():
    response = requests.get(events_api_url, headers=headers)
    response.raise_for_status()
    return response.json()

# Function to save events data to Excel
def save_events_to_excel(events_data, filename):
    df = pd.DataFrame(events_data)
    df.to_excel(filename, index=False)
    print(f"Events data successfully saved to {filename}")

# Main function
def main():
    events_data = fetch_events()

    events = events_data.get('events', [])
    if not events:
        print(f"No events found for series ticker '{series_ticker}'.")
        return

    save_events_to_excel(events, f'{series_ticker}_events.xlsx')

if __name__ == "__main__":
    main()