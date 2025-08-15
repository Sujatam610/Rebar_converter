import os
import fitz  # PyMuPDF
import tqdm
from PIL import Image
import io


Image.MAX_IMAGE_PIXELS = None
def process_file(file, file_bytes, uploaded_file, output_folder):
    """Processes the given file and saves images to a user-specific folder."""
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"📁 Created output folder: {output_folder}")
            
        print(f"🔄 Processing file: {uploaded_file.name if hasattr(uploaded_file, 'name') else str(uploaded_file)} in folder: {output_folder}")
        
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower() if hasattr(uploaded_file, 'name') else uploaded_file.split(".")[-1].lower()
        except (AttributeError, IndexError):
            file_extension = os.path.splitext(str(uploaded_file))[1][1:].lower()
            
        print(f"📋 File extension detected: {file_extension}")
        
        if file_extension == "pdf":
            return extract_images_from_pdf(file, output_folder)
        elif file_extension in ["jpg", "jpeg", "png"]:
            filename = uploaded_file.name if hasattr(uploaded_file, 'name') else f"image.{file_extension}"
            return save_image(file_bytes, output_folder, filename)
        else:
            error_msg = f"Unsupported file format: {file_extension}. Please provide a PDF or an image file."
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
            
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        raise e

def extract_images_from_pdf(uploaded_pdf, output_folder):
    """Extracts full-page images from the PDF at optimized DPI and ensures proper page ordering."""
    try:
        doc = fitz.open(stream=uploaded_pdf, filetype="pdf")  # Open from bytes
        image_paths = []
        total_pages = len(doc)
        print(f"📄 PDF has {total_pages} pages")

        for page_num in tqdm.trange(total_pages, desc="Extracting images"):
            page = doc.load_page(page_num)
            
            # Get page dimensions to calculate appropriate DPI
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            
            # Calculate DPI to keep image under safe memory limits
            # Target max pixels: ~50 million (safe for most systems)
            max_pixels = 50000000
            current_pixels_at_72dpi = page_width * page_height
            
            if current_pixels_at_72dpi > 0:
                max_dpi = min(600, int((max_pixels / current_pixels_at_72dpi) ** 0.5 * 72))
                dpi = max(150, max_dpi)  # Minimum 150 DPI for readability
            else:
                dpi = 300  # Default fallback
            
            print(f"📏 Page {page_num+1}: {page_width:.0f}x{page_height:.0f} pts, using {dpi} DPI")
            
            # Extract at calculated DPI
            pix = page.get_pixmap(dpi=dpi)
            
            # Create filename with proper zero-padding for correct sorting
            img_path = os.path.join(output_folder, f"page_{page_num+1:03d}.png")

            # Save the extracted image
            pix.save(img_path)
            print(f"📸 Extracted page {page_num+1} -> {os.path.basename(img_path)} ({pix.width}x{pix.height})")

            # Additional resize if still too large
            resize_large_image(img_path, max_pixels=50000000)  

            image_paths.append(img_path)
            
            # Clear pixmap from memory
            pix = None

        doc.close()  # Properly close the document
        
        # Sort paths to ensure correct page order
        image_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"✅ Successfully extracted {len(image_paths)} pages in order")
        return image_paths
        
    except Exception as e:
        print(f"❌ Error extracting images from PDF: {e}")
        return []

def resize_large_image(image_path, max_pixels=50000000):
    """Resizes the image if it exceeds max pixels limit while maintaining aspect ratio."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            total_pixels = width * height

            if total_pixels > max_pixels:
                scale_factor = (max_pixels / total_pixels) ** 0.5  # Calculate scaling factor
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                # Use LANCZOS for better quality when downscaling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img.save(image_path, format='PNG', optimize=True)  # Overwrite with resized image

                print(f"🔧 Resized {os.path.basename(image_path)} from {width}x{height} to {new_width}x{new_height}")
            else:
                print(f"✅ Image {os.path.basename(image_path)} size OK: {width}x{height}")

    except Exception as e:
        print(f"❌ Error resizing image: {e}")

def save_image(uploaded_image, output_folder, filename):
    """Save and optimize uploaded image file"""
    try:
        print(f"🖼️ Processing image: {filename}")
        img = Image.open(io.BytesIO(uploaded_image))  # Handle file-like object
        
        # Set max dimensions for optimization
        MAX_WIDTH, MAX_HEIGHT = 5000, 5000  
        original_size = img.size
        print(f"📏 Original image size: {original_size[0]}x{original_size[1]}")

        # Check if image exceeds pixel limit
        if img.size[0] * img.size[1] > 178956970:  
            print(f"🔧 Image too large, resizing...")
            img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)  # Resize proportionally
            print(f"📏 Resized to: {img.size[0]}x{img.size[1]}")

        save_path = os.path.join(output_folder, filename)
        img.save(save_path, format='PNG', optimize=True)
        print(f"✅ Image saved: {save_path}")
        
        return [save_path]
        
    except Exception as e:
        error_msg = f"Error processing image {filename}: {e}"
        print(f"❌ {error_msg}")
        raise e

