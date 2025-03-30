import requests
from PIL import Image
import io
from os.path import dirname, isdir, join as path_join
from os import mkdir

url = "http://localhost:8000/style-transfer/"
image_path = path_join(dirname(__file__), "example_images", "bald_eagle_portrait.jpg")
output_dir = path_join(dirname(__file__), "outputs")
if not isdir(output_dir):
    mkdir(output_dir)
output_path = path_join(output_dir, "bald_eagle_portrait_processed.png")

# Open the file and send with explicit MIME type
with open(image_path, "rb") as f:
    files = {"file": (f.name, f, "image/jpeg")}  # Explicitly set content-type
    response = requests.post(url, files=files)

# Check if request was successful
if response.status_code != 200:
    print(f"Error: {response.status_code} - {response.text}")
    exit()

# Get the processed image bytes from the response
image_bytes = response.content

# Save the image to disk
with open(output_path, "wb") as f:
    f.write(image_bytes)

# Open and display the image
processed_image = Image.open(output_path)
processed_image.show()