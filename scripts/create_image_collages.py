from PIL import Image
import os
import random

def create_image_collages(image_dir, output_dir='output'):
    """
    Creates multiple image collages, each on a 512x512 black canvas, incrementally adding images
    from the specified directory. Each collage will contain an increasing number of images, up to 20.

    Args:
    image_dir (str): The directory containing the images to be pasted onto the canvas.
    output_dir (str): The directory to save the resulting collage images. Default is 'output'.

    Returns:
    None
    """
    def is_position_valid(new_x, new_y, new_w, new_h, positions):
        for (x, y, w, h) in positions:
            if (new_x < x + w and new_x + new_w > x and
                new_y < y + h and new_y + new_h > y):
                return False
        return True

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a black canvas of size 512x512
    canvas_size = (512, 512)
    canvas_color = (0, 0, 0)

    # Get a list of image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

    # Limit the number of images to 20
    image_files = image_files[:20]

    for num_images in range(1, len(image_files) + 1):
        # Create a new canvas for each number of images
        canvas = Image.new('RGB', canvas_size, canvas_color)
        positions = []

        for i in range(num_images):
            # Open the image
            img = Image.open(os.path.join(image_dir, image_files[i]))

            # Resize the image to a width of 64 pixels while maintaining the aspect ratio
            w_percent = (64 / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((64, h_size), Image.Resampling.LANCZOS)

            # Find a valid position for the image
            placed = False
            while not placed:
                x = random.randint(0, canvas_size[0] - img.width)
                y = random.randint(0, canvas_size[1] - img.height)
                if is_position_valid(x, y, img.width, img.height, positions):
                    positions.append((x, y, img.width, img.height))
                    placed = True

            # Paste the image onto the canvas
            canvas.paste(img, (x, y))

        # Save the resulting collage
        output_file = os.path.join(output_dir, f'gallery_{num_images}_{"person" if num_images == 1 else "persons"}.jpg')
        canvas.save(output_file)

# Example usage
create_image_collages('../data/gallery/camerafortythree')
