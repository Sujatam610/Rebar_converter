import os
import fitz  # PyMuPDF
import tqdm
from PIL import Image
import io


Image.MAX_IMAGE_PIXELS = None
def process_file(file, file_bytes, uploaded_file, output_folder):
    """Processes the given file and saves images to a user-specific folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
    except AttributeError:
        file_extension = os.path.splitext(uploaded_file)[1][1:].lower()

    if file_extension == "pdf":
        return extract_images_from_pdf(file, output_folder)
    elif file_extension in ["jpg", "jpeg", "png"]:
        return save_image(file_bytes, output_folder, uploaded_file.name)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or an image file.")

def extract_images_from_pdf(uploaded_pdf, output_folder):
    """Extracts full-page images from the PDF at 500 DPI and resizes if necessary."""
    doc = fitz.open(stream=uploaded_pdf, filetype="pdf")  # Open from bytes
    image_paths = []

    for page_num in tqdm.trange(len(doc), desc="Extracting images"):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=500)  # High-resolution extraction
        img_path = os.path.join(output_folder, f"page_{page_num+1}.png")

        # Save the extracted image
        pix.save(img_path)

        # Resize the image if it exceeds a safe limit
        resize_large_image(img_path, max_pixels=178956970)  

        image_paths.append(img_path)

    return image_paths

def resize_large_image(image_path, max_pixels=178956970):
    """Resizes the image if it exceeds max pixels limit while maintaining aspect ratio."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            total_pixels = width * height

            if total_pixels > max_pixels:
                scale_factor = (max_pixels / total_pixels) ** 0.5  # Calculate scaling factor
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                img = img.resize((new_width, new_height), Image.LANCZOS)
                img.save(image_path)  # Overwrite with resized image

                print(f"Resized {image_path} to {new_width}x{new_height}")

    except Exception as e:
        print(f"Error resizing image: {e}")

def save_image(uploaded_image, output_folder, filename):
    try:
        img = Image.open(io.BytesIO(uploaded_image))  # Handle file-like object
        MAX_WIDTH, MAX_HEIGHT = 5000, 5000  # Set max dimensions

        if img.size[0] * img.size[1] > 178956970:  # If exceeding limit
            img.thumbnail((MAX_WIDTH, MAX_HEIGHT))  # Resize proportionally

        save_path = os.path.join(output_folder, filename)
        img.save(save_path)
        return [save_path]
    except Exception as e:
        print(f"Error processing image: {e}")
        raise e

