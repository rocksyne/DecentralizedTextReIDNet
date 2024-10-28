import os
from PIL import Image
import random

def create_image_canvas_incrementally(image_directory, output_path_prefix, first_image_name=None, max_images=5):
    # Create a 512x512 black canvas
    canvas_size = (512, 512)
    max_width = 80  # Maximum width for the images

    # Get a list of all images in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))]

    # If a first image is specified, ensure it is in the list and move it to the beginning
    if first_image_name:
        if first_image_name not in image_files:
            raise ValueError(f"The specified first image '{first_image_name}' is not found in the directory.")
        image_files.remove(first_image_name)
        image_files.insert(0, first_image_name)

    # Limit the number of images to max_images, including the first image
    image_files = image_files[:max_images]

    # Store the images
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = Image.open(image_path)
        # Resize image to max width while preserving aspect ratio
        image.thumbnail((max_width, canvas_size[1]), Image.LANCZOS)
        images.append(image)

    # Iterate over the number of images to be overlayed
    for count in range(1, len(images) + 1):
        canvas = Image.new('RGB', canvas_size, 'black')
        current_images = images[:count]
        
        # Calculate the total width of all images and the horizontal spacing
        total_width = sum(img.width for img in current_images)
        remaining_space = 512 - total_width
        spacing = remaining_space // (count + 1)

        current_x = spacing

        for image in current_images:
            # Position the image horizontally and randomly vertically
            y_position = random.randint(0, 512 - image.height)
            canvas.paste(image, (current_x, y_position))
            current_x += image.width + spacing

        # Save the final image
        output_path = f"{output_path_prefix}_{count}.jpg"
        canvas.save(output_path, 'JPEG')

# Usage
image_directory = '../data/gallery/camerafour'  # Replace with your image directory path
output_path_prefix = 'final_image'  # Prefix for output images
first_image_name = None #'1420_c6s3_085567_00.jpg'  # The specific image to be first (or None)
max_images = 10  # Limit the number of images to be used
create_image_canvas_incrementally(image_directory, output_path_prefix, first_image_name, max_images)
