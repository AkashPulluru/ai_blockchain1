import requests
import pandas as pd
import time

ETHERSCAN_API_KEY = '3RQ9X5Z9IAMFN468YSNXME1HSYUI8HWXAT'
CHAINLINK_CONTRACT_ADDRESS = '0x514910771AF9Ca656af840dff83E8264EcF986CA'

# Function to get transactions for Chainlink from Etherscan
def get_chainlink_transactions(start_block, end_block, page=1, offset=10000):
    url = 'https://api.etherscan.io/api'
    params = {
        'module': 'account',
        'action': 'tokentx',
        'contractaddress': CHAINLINK_CONTRACT_ADDRESS,
        'startblock': start_block,
        'endblock': end_block,
        'page': page,
        'offset': offset,
        'sort': 'asc',
        'apikey': ETHERSCAN_API_KEY
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    result = response.json()
    if result['status'] != '1':
        print('Etherscan API returned error:', result.get('message', 'Unknown error'))
        return []
    return result['result']


# Example function to fetch Oracle events (simplified)
def get_oracle_events(transaction_hash):
    url = 'https://api.etherscan.io/api'
    params = {
        'module': 'logs',
        'action': 'getLogs',
        'txhash': transaction_hash,
        'apikey': ETHERSCAN_API_KEY
    }
    response = requests.get(url, params=params)
    result = response.json()
    if result['status'] != '1':
        return []
    return result['result']

# Main function to gather and merge data
def gather_chainlink_data(start_block, end_block):
    all_transactions = []
    page = 1
    while True:
        print(f"Fetching transactions page {page}...")
        transactions = get_chainlink_transactions(start_block, end_block, page)
        if not transactions:
            break
        for tx in transactions:
            tx_hash = tx['hash']
            oracle_events = get_oracle_events(tx_hash)
            tx['oracle_events'] = oracle_events
            all_transactions.append(tx)
        page += 1
        time.sleep(0.2)  # Avoid hitting rate limits

    df = pd.json_normalize(all_transactions)
    df.to_csv('chainlink_transaction_data.csv', index=False)
    print('Data saved to chainlink_transaction_data.csv')

# Example usage (specify your block range)
gather_chainlink_data(start_block=17000000, end_block=17000003)
