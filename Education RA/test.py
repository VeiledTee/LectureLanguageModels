import os
import base64
import time
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration using .env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
INPUT_ROOT_DIR = os.getenv("LECTURE_NOTE_DIR")
OUTPUT_ROOT_DIR = os.getenv("LECTURE_OUTPUT_DIR")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.0"))
IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "png").lower()


def encode_image(image_path: str) -> str:
    """Encode image to base64 with format check"""
    if not image_path.lower().endswith(f".{IMAGE_FORMAT}"):
        raise ValueError(f"Invalid image format, expected {IMAGE_FORMAT}")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_to_md_with_gpt(image_path: str) -> str:
    """Process image with configured OpenAI model"""
    base64_image = encode_image(image_path)

    return openai.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail for documentation purposes:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/{IMAGE_FORMAT};base64,{base64_image}"
                }}
            ]
        }],
        max_tokens=int(os.getenv("MAX_TOKENS")),
        temperature=float(os.getenv("GENERATION_TEMPERATURE", "0.3")),
        top_p=float(os.getenv("TOP_P", "0.9"))
    ).choices[0].message.content


def process_directory(image_paths: list[str]) -> list[tuple[str, str]]:
    """Process all images in a directory with retries"""
    directory_results = []
    for img_path in image_paths:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Processing: {img_path} (Attempt {attempt + 1}/{MAX_RETRIES})")
                description = image_to_md_with_gpt(img_path)
                directory_results.append((os.path.basename(img_path), description))
                time.sleep(REQUEST_DELAY)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to process {img_path}: {str(e)}")
                    directory_results.append((os.path.basename(img_path), f"ERROR: {str(e)}"))
                time.sleep(2 ** attempt)

    return directory_results


def save_aggregated_results(directory: str, results: list[tuple[str, str]]):
    """Save results to a file named {dir_name}_aggregated_results.txt in the OUTPUT_ROOT_DIR"""
    # Get the base directory name (e.g., "transistors-gates")
    dir_name = os.path.basename(directory)
    output_file = f"{dir_name}_aggregated_results.txt"

    # Ensure the OUTPUT_ROOT_DIR exists (e.g., "Computer_Architecture/Processed")
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_ROOT_DIR, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        for filename, description in results:
            f.write(f"=== {filename} ===\n")
            f.write(f"{description}\n\n")
        print(f"Saved {filename} to {output_file}")



if __name__ == "__main__":
    print("running...")
    if not os.path.isdir(INPUT_ROOT_DIR):
        raise ValueError(f"Input directory does not exist: {INPUT_ROOT_DIR}")

    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        print(f"Processing directory: {root}")
        image_files = [os.path.join(root, f) for f in files if f.lower().endswith(f".{IMAGE_FORMAT}")]
        print(f"Found images: {image_files}")

        if not image_files:
            continue

        results = process_directory(image_files)
        save_aggregated_results(root, results)