import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    financial_data = json.loads(completion.choices[0].message.content)

    return financial_data


if __name__ == "__main__":
    company = input("Enter the company name: ")
    data = generate_income_and_stock_data(company)

    # Output data
    print(json.dumps(data, indent=4))