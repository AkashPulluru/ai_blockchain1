import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file (recommended)
load_dotenv()
client = OpenAI(api_key="")

# Fetch and parse URL content
def get_url_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = '\n'.join([para.get_text() for para in paragraphs])
    return content

# Create summary using updated OpenAI SDK
def summarize_content(content):
    prompt = f"""
    Summarize the following content into exactly 10 sentences, clearly describing the main points and important implications:

    {content[:8000]}  # Truncated to fit token limit
    """

    completion = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo" if preferred
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800,
    )

    summary = completion.choices[0].message.content.strip()
    return summary

# Save summary to file
def save_summary(url, summary):
    filename = "url_summary.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"URL: {url}\n\nSummary:\n{summary}")

# Main function
def main():
    url = input("Enter the URL: ")
    print("Fetching content...")
    content = get_url_content(url)

    print("Generating summary...")
    summary = summarize_content(content)

    print("Saving summary...")
    save_summary(url, summary)
    print("Summary saved successfully.")

if __name__ == "__main__":
    main()
