import os
import re
import openai
from PIL import Image
import base64
from io import BytesIO
from pydantic import BaseModel

# Specify the folder path where images are stored
folder_path = r'C:\Users\Chris Joel\Desktop\cs-research-project\Evaluation Scripts\lecture_notes'

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

responses = []


# Define the Pydantic model for the response
class QAResponse(BaseModel):
    question: str
    answer: str

# Iterate over each lecture folder
for lecture_folder in os.listdir(folder_path):
    lecture_folder_path = os.path.join(folder_path, lecture_folder)
    if os.path.isdir(lecture_folder_path):
        # Iterate over each image in the lecture folder
        for image_file in os.listdir(lecture_folder_path):
            image_path = os.path.join(lecture_folder_path, image_file)
            if os.path.isfile(image_path) and image_file.endswith('.jpeg'):
                # Load the image
                image = Image.open(image_path)

                # Convert image to base64
                image_base64 = image_to_base64(image)
                # Use OpenAI API to extract text from the image
                response = openai.beta.chat.completions.parse(
                    model="gpt-4o",
                    response_format=QAResponse,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                    "url": image_base64
                                },
                                },
                                {
                                    "type" : "text",
                                    "text": "Convert the given image into JSON format, ensuring that questions and answers are structured under 'question' and 'answer' properties. If the image contains graphical content, describe it in text format instead.."
                                }
                            ]
                        }
                    ]
                )
                parsed_response = response.choices[0].message.parsed
                responses.append(parsed_response)
                print(parsed_response)


with open('responses.json', 'w') as f:
    json.dump([response.dict() for response in responses], f, indent=2)