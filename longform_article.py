import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def sanitize_filename(filename):
    # Remove illegal characters and replace spaces with underscores
    filename = re.sub(r'[^\w\-_\. ]', '_', filename)
    return filename.replace(' ', '_')

def generate_deep_dive(topic):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert researcher and insightful writer tasked with creating deep, "
                    "nuanced, and detailed explorations of various topics. Your responses should be engaging, "
                    "thoroughly researched, and rich in context."
                ),
            },
            {
                "role": "user",
                "content": f"Please provide a comprehensive, longform deep dive into the following topic: {topic}",
            },
        ],
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    topic = input("Enter the topic for your deep dive: ")
    deep_dive = generate_deep_dive(topic)

    filename = sanitize_filename(topic) + ".txt"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(deep_dive)

    print(f"\nGenerated Deep Dive saved to {filename}\n")
