import os
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Initialize the OpenAI client (recommended way from README)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_deep_dive(topic):
    # Use the recommended Chat Completions API structure
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

    # Accessing the response as recommended
    return completion.choices[0].message.content

if __name__ == "__main__":
    topic = input("Enter the topic for your deep dive: ")
    deep_dive = generate_deep_dive(topic)
    print("\nGenerated Deep Dive:\n")
    print(deep_dive)