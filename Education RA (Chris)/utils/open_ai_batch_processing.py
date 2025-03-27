from openai import OpenAI
import os
import time

def upload_and_create_batch(
    jsonl_file_path: str,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    verbose: bool = True
) -> dict:
        # Check if file exists
    if not os.path.exists(jsonl_file_path):
        raise FileNotFoundError(f"File not found: {jsonl_file_path}")

    # Initialize OpenAI client
    client = OpenAI()

    # Step 1: Upload the JSONL file
    try:
        with open(jsonl_file_path, "rb") as file:
            file_response = client.files.create(
                file=file,
                purpose="batch"
            )
        file_id = file_response.id
        if verbose:
            print(f"File uploaded successfully. File ID: {file_id}")
    except Exception as e:
        raise Exception(f"Failed to upload file: {str(e)}")

    # Step 2: Create the batch job
    try:
        batch_response = client.batches.create(
            input_file_id=file_id,
            endpoint=endpoint,
            completion_window=completion_window
        )
        batch_id = batch_response.id
        if verbose:
            print(f"Batch created successfully. Batch ID: {batch_id}")
    except Exception as e:
        raise Exception(f"Failed to create batch: {str(e)}")

    # Step 3: Get initial batch status
    try:
        batch_status = client.batches.retrieve(batch_id)
        if verbose:
            print(f"Batch Status: {batch_status.status}")
            print(f"Request Counts: {batch_status.request_counts}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not retrieve batch status: {str(e)}")
        batch_status = None

    # Return relevant IDs and status
    return {
        "file_id": file_id,
        "batch_id": batch_id,
        "batch_status": batch_status.status if batch_status else "unknown"
    }

def save_batch_response(batch, output_file):
    client = OpenAI()
    batch_status = client.batches.retrieve(batch["batch_id"])
    while (batch_status.status != "completed"):
        batch_status = client.batches.retrieve(batch["batch_id"])
        if batch_status.status == "completed":
            output_file_id = batch_status.output_file_id
            processed_output_file = client.files.content(output_file_id)
            with open(output_file, "wb") as f:
                f.write(processed_output_file.read())
            print(f"Batch output downloaded to {output_file}")
        else:
            print("pending...")
        time.sleep(10)