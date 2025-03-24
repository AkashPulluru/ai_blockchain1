import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=("scratch key"))

# Load CSV file
def load_ethereum_data(csv_path="ethereum_prices.csv"):
    df = pd.read_csv(csv_path)
    return df

# Generate analytics prompt from data
def generate_prompt_from_data(df):
    sample = df.head(10).to_csv(index=False)  # Small sample for context
    columns = ', '.join(df.columns)
    
    prompt = f"""
You are a data analyst. I have Ethereum price data with the following columns: {columns}.

Here is a small sample of the data:
{sample}

Based on this structure, please answer:
1. What are some interesting trends or anomalies you can identify?
2. What statistical insights could we derive (e.g., volatility, moving averages, seasonal patterns)?
3. What visualizations would help represent this data clearly?
4. What features could be useful for a machine learning model predicting price movement?
5. Please do the analysis whenever possible

Please format your response in clear sections.
    """
    return prompt

# Query OpenAI API
def analyze_data_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()

# Save results to file
def save_results(output, filename="ethereum_analysis.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(output)

# Main
def main():
    print("Loading data...")
    df = load_ethereum_data()

    print("Generating prompt...")
    prompt = generate_prompt_from_data(df)

    print("Analyzing with GPT...")
    result = analyze_data_with_gpt(prompt)

    print("Saving results...")
    save_results(result)
    print("Analysis complete and saved to ethereum_analysis.txt")

if __name__ == "__main__":
    main()
