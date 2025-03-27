
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_income_and_stock_data(company_name):
    prompt = f"""
    Generate realistic financial data for {company_name} for the past 10 years in JSON format, structured as follows:

    {{
      "income_statement": [
        {{"year": 2023, "revenue": float, "cogs": float, "sgna": float, "net_income": float, "free_cash_flow": float}},
        {{"year": 2022, ...}},
        ... (10 years total)
      ],
      "stock_prices": [
        {{"year": 2023, "stock_price": float}},
        {{"year": 2022, ...}},
        ... (10 years total)
      ],
      "irr": float (calculate IRR from the stock prices)
    }}

    Use actual company data.
    """

    response = client.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.7
    )

    financial_data = json.loads(response.choices[0].text.strip())

    return financial_data

if __name__ == "__main__":
    company = input("Enter the company name: ")
    data = generate_income_and_stock_data(company)

    # Output data
    print(json.dumps(data, indent=4))