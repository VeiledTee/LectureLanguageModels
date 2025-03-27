import os
import json
from tqdm import tqdm

def generate_custom_id(filename: str, index: int) -> str:
    """Generate a custom ID from filename and question index."""
    base_name = filename.replace('_processed.json', '').replace('.json', '')
    file_id_base = base_name[0:5] + base_name[-5:] if len(base_name) >= 5 else base_name.zfill(5)
    return f"{file_id_base}{index:03d}"

def process_single_file(file_path: str) -> list:
    """Process a single JSON file and return questions with custom IDs."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get('questions', [])
        processed = []
        
        for idx, question in enumerate(questions):
            question['custom_id'] = generate_custom_id(os.path.basename(file_path), idx)
            processed.append(question)
        
        return processed
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def combine_json_files(input_dir: str, output_file: str) -> None:
    """
    Combine all JSON files from input directory into a single output file.
    
    Args:
        input_dir: Path to directory containing JSON files
        output_file: Path for the output combined JSON file
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    combined_data = []
    
    for json_file in tqdm(json_files, desc=f"Processing files in {input_dir}"):
        file_path = os.path.join(input_dir, json_file)
        combined_data.extend(process_single_file(file_path))
    
    if combined_data:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4)
        print(f"Combined {len(combined_data)} questions into {output_file}")
    else:
        print("No valid data to combine")

# if __name__ == "__main__":
#     # Example usage
#     input_directory = "output_json_files/introduction_to_algorithms"
#     output_file_path = "combined_json_files/combined_introduction_to_algorithms.json"
#     combine_json_files(input_directory, output_file_path)