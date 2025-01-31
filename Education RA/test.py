import os
import subprocess

import modal

# Initialize the volume
volume = modal.Volume.from_name("my-persisted-volume", create_if_missing=True)

# List files in the '/root/' directory
print(volume.listdir("/root/"))

# Iterate through files in the '/root/' directory recursively
for filename in volume.iterdir(path="/root/", recursive=True):
    if filename.path.endswith("_answerless.txt"):
        print(f"Found file: {filename.path}")

        # Construct the local path where you want to save the file
        local_path = os.path.join(
            "/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI Course",
            os.path.basename(filename.path),
        )

        command = [
            "modal",
            "volume",
            "get",
            "my-persisted-volume",
            filename.path,
            local_path,
        ]

        # Run the command
        subprocess.run(command, check=True)
