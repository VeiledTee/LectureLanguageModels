import os
import base64
import time
import openai
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_env_var(name: str) -> str:
    """Get required environment variable with error handling"""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


# Configuration using .env variables
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
OPENAI_MODEL_NAME = get_env_var("OPENAI_MODEL_NAME")
INPUT_ROOT_DIR = get_env_var("LECTURE_NOTE_DIR")
OUTPUT_ROOT_DIR = get_env_var("LECTURE_OUTPUT_DIR")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.0"))
IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "jpeg").lower()


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
        max_tokens=int(get_env_var("MAX_TOKENS")),
        temperature=float(os.getenv("GENERATION_TEMPERATURE", "0.3")),
        top_p=float(os.getenv("TOP_P", "0.9"))
    ).choices[0].message.content


def process_directory(image_paths: List[str]) -> List[Tuple[str, str]]:
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


def save_aggregated_results(directory: str, results: List[Tuple[str, str]]):
    """Save results with directory structure preservation"""
    relative_path = os.path.relpath(directory, INPUT_ROOT_DIR)
    output_dir = os.path.join(OUTPUT_ROOT_DIR, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "aggregated_descriptions.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Image Descriptions for Directory: {relative_path}\n\n")
        f.write(f"Generated with {OPENAI_MODEL_NAME} (Temperature: {os.getenv('GENERATION_TEMPERATURE')})\n\n")

        for filename, description in results:
            f.write(f"=== {filename} ===\n")
            f.write(f"{description}\n\n")
            f.write("-" * 50 + "\n\n")


def main():
    openai.api_key = OPENAI_API_KEY

    # Walk through directory structure
    for root, dirs, files in os.walk(INPUT_ROOT_DIR):
        image_files = [os.path.join(root, f) for f in files if f.lower().endswith(f".{IMAGE_FORMAT}")]

        if not image_files:
            continue

        print(f"\nProcessing directory: {root}")
        results = process_directory(image_files)
        save_aggregated_results(root, results)


if __name__ == "__main__":
    main()
