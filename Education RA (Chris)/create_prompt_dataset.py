import json

def create_chatgpt_input_dataset(input_file, output_file, shots=0):
    """
    Creates a ChatGPT batch dataset with few-shot examples using reverse indexing.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output .jsonl file.
        shots (int): Number of previous examples to include (default: 0 for zero-shot).
    """
    try:
        # Read the input JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        
        # Validate that input_data is a list
        if not isinstance(input_data, list):
            raise ValueError("Input JSON must be a list of objects")
        
        if shots < 0:
            raise ValueError("Number of shots must be non-negative")
        
        # Write to the .jsonl file
        with open(output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(input_data):
                # System prompt
                system_message = {
                    "role": "system",
                    "content": (
                        "You are an AI model tasked with evaluating OpenCourseWare (OCW) questions and answers. "
                        "All equations in the questions are written in LaTeX format. If the answers include any equations, "
                        "ensure they are also written in LaTeX format. Provide accurate and contextually relevant responses "
                        "to each question. Use the following examples (if any) to guide your response style and format."
                    )
                }
                
                # Prepare messages array starting with system prompt
                messages = [system_message]
                
                # Add few-shot examples using reverse indexing
                num_examples = min(shots, i)  # Limit to available previous items
                for j in range(1, num_examples + 1):  # Start from 1 to num_examples
                    prev_item = input_data[i - j]  # Access previous item directly
                    example_prompt = (
                        f"Context: {prev_item['context']}\n\n"
                        f"**Problem {prev_item['problem_number']}**\n\n"
                        f"Question: {prev_item['question']}"
                    )
                    example_answer = prev_item.get("answer", "No answer provided")
                    messages.append({"role": "user", "content": example_prompt})
                    messages.append({"role": "assistant", "content": example_answer})
                
                # Add the current question
                user_content = (
                    f"Context: {item['context']}\n\n"
                    f"**Problem {item['problem_number']}**\n\n"
                    f"Question: {item['question']}\n\n"
                    "Provide an answer to the question above. Ensure the answer is accurate, contextually relevant, "
                    "and uses LaTeX format for any mathematical equations."
                )
                messages.append({"role": "user", "content": user_content})
                
                # Create the batch entry
                batch_entry = {
                    "custom_id": item["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": messages,
                    }
                }
                
                # Write to file
                f.write(json.dumps(batch_entry) + "\n")
        
        print(f"Batch .jsonl file created: {output_file} with {len(input_data)} entries "
              f"using {shots}-shot learning")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

# # Example usage
# if __name__ == "__main__":
#     create_chatgpt_input_dataset("input.json", "batch_requests.jsonl", shots=2)