import os
import json

# Path to the folder containing JSON files
data_folder = "."

# Output list to store the formatted data for ChatGPT batch API

def process_json_files(data_folder):
    """
    Process a single JSON file and append questions to the chatgpt_input list in the desired format.
    """
    total_data = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_folder, file_name)
            
            print(file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                for i, item in enumerate(data):
                    for j, sub in enumerate(item.get("subquestions", [])):
                        sub["custom_id"] = "request_" + file_path[-12:-5] + "_" + str(i) + "_" + str(j)

                total_data += data
    return total_data


print("total number of files", len(os.listdir(data_folder)))
total_data = process_json_files(data_folder)

# Save the processed data to a new JSON file
output_file = "chatgpt_dataset_combined.jsonl"
with open(output_file, "w", encoding="utf-8") as out_file:
    json.dump(total_data, out_file, indent=4)

print(f"Processed data has been saved to {output_file}")
