import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load README content for additional context
def load_readme(filename="chatgptreadme.md"):
    with open(filename, "r") as file:
        return file.read()


def generate_income_and_stock_data(company_name, readme_content):
    prompt = f"""
    Using actual available data, generate financial data for {company_name} for the past 10 years in the following JSON format:

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
      "irr": float
    }}

    README Context:\n{readme_content}

    Respond only with the JSON data, without additional text.
    """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )

    response_content = completion.choices[0].message.content

    # Remove markdown formatting
    response_content = re.sub(r'^```json|```$', '', response_content.strip(), flags=re.MULTILINE)

    try:
        financial_data = json.loads(response_content)
    except json.JSONDecodeError as e:
        print("Failed to decode JSON response:", e)
        print("Response content:", response_content)
        return None

    return financial_data


def save_to_excel(data, company_name):
    income_df = pd.DataFrame(data['income_statement'])
    stock_df = pd.DataFrame(data['stock_prices'])
    irr_df = pd.DataFrame([{'irr': data['irr']}])

    with pd.ExcelWriter(f"{company_name}_financial_data.xlsx") as writer:
        income_df.to_excel(writer, sheet_name='Income Statement', index=False)
        stock_df.to_excel(writer, sheet_name='Stock Prices', index=False)
        irr_df.to_excel(writer, sheet_name='IRR', index=False)

    print(f"Financial data saved to {company_name}_financial_data.xlsx")


if __name__ == "__main__":
    company = input("Enter the company name: ")
    readme_content = load_readme()
    data = generate_income_and_stock_data(company, readme_content)

    if data:
        save_to_excel(data, company)
    else:
        print("No valid data returned.")