import os
import re
import shutil
from pdf2image import convert_from_path

# Specify the folder path
folder_path = r'C:\Users\Chris Joel\Downloads\6.857-spring-2014\static_resources'

# List all files in the folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Filter the list of files that end with "lec" followed by a number and have a ".pdf" extension
pattern = re.compile(r'lec(\d+)\.pdf$', re.IGNORECASE)
filtered_files = [file for file in files if pattern.search(file)]

# Create a new folder called "lecture_notes" in the current directory
new_folder_path = os.path.join(os.getcwd(), 'lecture_notes')
os.makedirs(new_folder_path, exist_ok=True)

# Print the path of the new folder
print(f"Lecture notes folder created at: {new_folder_path}")

# Convert each page of the filtered PDF files to images and store them in the appropriate folder
for file in filtered_files:
    match = pattern.search(file)
    if match:
        lec_number = match.group(1)
        lec_folder_path = os.path.join(new_folder_path, f'lec_{lec_number}')
        os.makedirs(lec_folder_path, exist_ok=True)
        
        pdf_path = os.path.join(folder_path, file)
        images = convert_from_path(pdf_path)
        
        for i, image in enumerate(images):
            image_path = os.path.join(lec_folder_path, f'page_{i+1}.jpeg')
            image.save(image_path, 'JPEG')
            print(f"Saved: {image_path}")

