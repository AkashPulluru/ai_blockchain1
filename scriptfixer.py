import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# Configure your OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get the corrected version of a script
def correct_script(script_content):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant tasked with correcting Python scripts. Provide the corrected script only, without explanations."},
            {"role": "user", "content": f"Please find and correct any errors in this Python script:\n\n{script_content}"},
        ]
    )
    corrected_script = response.choices[0].message.content
    return corrected_script

# Main function to automate corrections up to 5 iterations
def automate_corrections(filename):
    with open(filename, "r") as file:
        script_content = file.read()

    for i in range(5):
        corrected_content = correct_script(script_content)

        # Check if corrections have stabilized
        if corrected_content.strip() == script_content.strip():
            print(f"Script corrections stabilized after {i+1} iterations.")
            break

        script_content = corrected_content

    output_filename = f"corrected_{filename}"
    with open(output_filename, "w") as file:
        file.write(script_content)

    print(f"Final corrected script written to '{output_filename}'.")

if __name__ == "__main__":
    script_filename = input("Enter the filename of the script to correct: ")
    automate_corrections(script_filename)
