import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get the corrected version of a script
def correct_script(script_content, readme_content):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant tasked with correcting Python scripts. Provide the corrected script only, without explanations."},
            {"role": "user", "content": f"Please find and correct any errors in this Python script. Refer to the following README for context:\n\n{readme_content}\n\nScript:\n\n{script_content}"},
        ]
    )
    return response.choices[0].message.content

# Main function to automate corrections up to 5 iterations. Pastes number
def automate_corrections(filename, readme_filename="chatgptreadme.md"):
    with open(filename, "r") as file:
        script_content = file.read()

    with open(readme_filename, "r") as readme_file:
        readme_content = readme_file.read()

    for i in range(5):
        corrected_content = correct_script(script_content, readme_content)

        if corrected_content.strip() == script_content.strip():
            print(f"Script corrections stabilized after {i+1} iterations.")
            break

        script_content = corrected_content

    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_corrected{ext}"

    with open(output_filename, "w") as file:
        file.write(script_content)

    print(f"Final corrected script written to '{output_filename}'.")

if __name__ == "__main__":
    script_filename = input("Enter the filename of the script to correct: ")
    automate_corrections(script_filename)