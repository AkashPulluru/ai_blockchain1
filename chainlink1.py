import requests
import pandas as pd
import time

ETHERSCAN_API_KEY = '3RQ9X5Z9IAMFN468YSNXME1HSYUI8HWXAT'
CHAINLINK_CONTRACT_ADDRESS = '0x514910771AF9Ca656af840dff83E8264EcF986CA'

def get_chainlink_transactions(start_block, end_block, page=1, offset=1000):
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
        if result.get('message') == 'No transactions found':
            return []
        if 'Result window is too large' in result.get('result', ''):
            raise ValueError('Pagination limit exceeded')
        print('Etherscan API returned error:', result.get('message', 'Unknown error'))
        return []
    return result['result']

def get_oracle_events(transaction_hash):
    url = 'https://api.etherscan.io/api'
    params = {
        'module': 'logs',
        'action': 'getLogs',
        'txhash': transaction_hash,
        'apikey': ETHERSCAN_API_KEY
    }
    response = requests.get(url, params=params, timeout=30)
    result = response.json()
    if result['status'] != '1':
        return []
    return result['result']

def gather_chainlink_data(start_block, end_block, initial_step=20):
    all_transactions = []
    current_start = start_block

    while current_start <= end_block:
        step = initial_step
        success = False
        while not success:
            current_end = min(current_start + step - 1, end_block)
            try:
                transactions = []
                page = 1
                print(f"Fetching blocks {current_start}-{current_end} with step {step}...")
                while True:
                    tx_batch = get_chainlink_transactions(current_start, current_end, page, offset=1000)
                    if not tx_batch:
                        break
                    transactions.extend(tx_batch)
                    if len(tx_batch) < 1000:
                        break
                    page += 1
                    time.sleep(0.2)

                # Fetch oracle events
                for tx in transactions:
                    tx['oracle_events'] = get_oracle_events(tx['hash'])
                    time.sleep(0.1)  # avoid rate limiting
                all_transactions.extend(transactions)
                success = True
            except ValueError:
                # Reduce step if limit exceeded
                if step <= 2:
                    print(f"Block {current_start} is too dense; skipping one block ahead.")
                    current_end = current_start
                    success = True  # Skip problematic block
                else:
                    step = max(1, step // 2)
                    print(f"Reducing step size to {step} due to API limit.")

        current_start = current_end + 1

    df = pd.json_normalize(all_transactions)
    df.to_csv('chainlink_transaction_data.csv', index=False)
    print(f'Data saved ({len(df)} transactions) to chainlink_transaction_data.csv')

# Example usage:
gather_chainlink_data(start_block=17000000, end_block=17010000, initial_step=100)
