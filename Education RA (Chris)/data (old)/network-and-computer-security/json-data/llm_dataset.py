import os
import json

# Path to the folder containing JSON files
data_folder = "."

# Output list to store the formatted data for ChatGPT batch API
chatgpt_input = []

def process_json_file(file_path):
    """
    Process a single JSON file and append questions to the chatgpt_input list in the desired format.
    """
    print(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

        for i, item in enumerate(data):
            # Extract parent question
            parent_question = item.get("question", "")

            # Process each subquestion
            for j, sub in enumerate(item.get("subquestions", [])):
                sub_question = sub.get("question", "")
                answer = sub.get("answer", "")

                # Create a ChatGPT input in the batch API format
                input_item = {
                    "custom_id": "request_" + file_path[-12:-5] + "_" + str(i) + "_" + str(j),
                    "method": "POST", 
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are an AI model tasked with evaluating OpenCourseWare (OCW) questions and answers. "
                                    "All equations in the questions are written in LaTeX format. If the answers include any equations, "
                                    "ensure they are also written in LaTeX format. Provide accurate and contextually relevant responses "
                                    "to each question."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"{parent_question}\n\n: {sub_question}"
                            }
                        ],
                        "max_tokens": 2000
                    }
                }
                chatgpt_input.append(input_item)


print("total number of files", len(os.listdir(data_folder)))

# Iterate through each JSON file in the folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".json"):
        file_path = os.path.join(data_folder, file_name)
        process_json_file(file_path)

# Save the processed data to a new JSON file
output_file = "chatgpt_batch_input.jsonl"
with open(output_file, "w", encoding="utf-8") as out_file:
    json.dump(chatgpt_input, out_file, indent=4)

print(f"Processed data has been saved to {output_file}")
