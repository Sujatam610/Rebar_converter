import os
import time
import json
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from doc_parser import process_file
from dotenv import load_dotenv
load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def process_and_extract(file, file_bytes, file_name, output_folder=None, batch_size=2, max_retries=3):
    """Processes the document, extracts images, and sends them in batches for processing."""
    try:
        image_paths = process_file(file, file_bytes, file_name, output_folder) 
        print(f"📄 Total pages extracted: {len(image_paths)}")
        
        # Sort image paths to ensure proper page order
        image_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'page_' in x else 0)
        
        for i, path in enumerate(image_paths):
            print(f"   Page {i+1}: {os.path.basename(path)}")

        if not image_paths:
            print("❌ No images extracted from document")
            return []

        extracted_data = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        page_count = 1
        
        while image_paths:
            batch = image_paths[:batch_size]
            image_paths = image_paths[batch_size:]
            
            print(f"🔄 Processing batch {page_count}/{total_batches} with {len(batch)} pages...")
            batch_data = extract_file_details(batch, max_retries=max_retries)
            print(f"📊 Batch {page_count} returned {len(batch_data)} results")
            
            if batch_data:  # Only extend if data is returned
                extracted_data.extend(batch_data)
            page_count += 1

            # Clean up batch images
            for img in batch:
                try:
                    print(f"🗑️ Cleaning up: {os.path.basename(img)}")
                    os.remove(img)
                except Exception as e:
                    print(f"⚠️ Failed to remove {img}: {e}")

        print(f"✅ Total extraction completed: {len(extracted_data)} data chunks from {total_batches} batches")
        return extracted_data
        
    except Exception as e:
        print(f"❌ Error in process_and_extract: {e}")
        return []

def extract_file_details(image_paths, max_retries=3, retry_delay=2):
    prompt = """Extract ALL tabular data from the image and convert to JSON format.

TASK: Find every table in the image and extract ALL rows and columns exactly as they appear.

INSTRUCTIONS:
1. Identify ALL tables in the image (ignore any non-tabular content)
2. For each table, extract the column headers exactly as written
3. Extract ALL data rows, preserving the exact values from each cell
4. Create a JSON object for each row using the column headers as keys
5. If a cell is empty, use null
6. If column headers are missing, use generic names like "Col1", "Col2", etc.

OUTPUT FORMAT: Return a JSON array where each object represents one table row:
[
  {
    "Column_Header_1": "cell_value_1",
    "Column_Header_2": "cell_value_2",
    "Column_Header_3": "cell_value_3"
  },
  {
    "Column_Header_1": "cell_value_4", 
    "Column_Header_2": "cell_value_5",
    "Column_Header_3": "cell_value_6"
  }
]

RULES:
1. Extract from ALL tables found in the image
2. Use exact column header names (clean spaces, special chars to underscores)
3. Extract cell values exactly as shown (numbers, text, symbols)
4. Include ALL rows from ALL tables
5. Return only valid JSON, no explanations
6. If no tables found, return []
7. For merged cells, repeat the value for each cell it spans
8. Maintain the original data types (numbers as numbers, text as text)

COLUMN NAMING:
- Use actual header text from the table
- Replace spaces with underscores: "Bar Mark" → "Bar_Mark"
- Remove special characters: "Qty." → "Qty"
- If no headers, use: "Col1", "Col2", "Col3", etc.

Extract every table and every row you find in the image."""

    extracted_info = []
    
    for i, image_path in enumerate(image_paths):
        print(f"🔍 Processing page {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Optimize image before sending
                optimized_image = optimize_image_for_api(image_path)
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt, optimized_image],
                    config=types.GenerateContentConfig(
                        temperature=0.0,  # Zero temperature for maximum consistency
                        max_output_tokens=500000,  
                        response_mime_type="application/json",  
                    )
                )

                # Log response details
                response_text = response.text.strip()
                print(f"📄 Page {i+1} response length: {len(response_text)} characters")
                
                # Validate JSON response
                try:
                    test_json = json.loads(response_text) if response_text else []
                    if isinstance(test_json, list):
                        print(f"✅ Page {i+1} valid JSON with {len(test_json)} records")
                    else:
                        print(f"⚠️ Page {i+1} JSON not a list, converting...")
                        response_text = "[]"
                except json.JSONDecodeError:
                    print(f"⚠️ Page {i+1} invalid JSON, setting to empty array")
                    response_text = "[]"
                
                extracted_info.append(response_text)
                success = True
                print(f"✅ Successfully processed page {i+1}: {os.path.basename(image_path)}")
                
            except (ServerError, APIError) as e:
                retry_count += 1
                print(f"⚠️ Page {i+1} attempt {retry_count} failed: {str(e)}")
                
                if retry_count < max_retries:
                    print(f"🔄 Retrying page {i+1} in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed to process page {i+1} after {max_retries} attempts")
                    # Return empty JSON array for failed images
                    extracted_info.append("[]")
            
            except Exception as e:
                print(f"❌ Unexpected error processing page {i+1}: {str(e)}")
                extracted_info.append("[]")
                break

    print(f"🎯 Total pages processed: {len(extracted_info)}")
    return extracted_info


def optimize_image_for_api(image_path, max_size=(2048, 2048), quality=85):
    """Optimize image for API processing to reduce timeout risk"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"🔧 Resized image to {img.size} for API optimization")
            
            return img.copy()
    except Exception as e:
        print(f"⚠️ Image optimization failed: {e}, using original")
        return Image.open(image_path)
