import os
from PIL import Image
from google import genai
from google.genai import types
from doc_parser import process_file
from dotenv import load_dotenv
load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def process_and_extract(file, file_bytes, file_name, output_folder=None, batch_size=2):
    """Processes the document, extracts images, and sends them in batches for processing."""
    image_paths = process_file(file, file_bytes, file_name, output_folder) 

    extracted_data = []
    while image_paths:
        batch = image_paths[:batch_size]
        image_paths = image_paths[batch_size:]

        batch_data = extract_file_details(batch)
        extracted_data.extend(batch_data)

        for img in batch:
            os.remove(img)

    return extracted_data

def extract_file_details(image_paths):
    prompt = """Extract the details from the provided content in a **strictly structured JSON format**. Follow the exact structure below:
            [
            {
                "DWG #": "<Drawing Number>",
                "ITEM": "<Item Number>",
                "GRADE": 400W,
                "WEIGHT": "<Total Weight>",
                "BUNDLE": "<Bundle Number>",
                "MARK": "<Marking Identifier>",
                "QUANTITY": "<Total Quantity>",
                "SIZE": "<Size Specifications>",
                "TYPE": "<Material Type>",
                "TOTAL LENGTH": "<Total Length>",
                "A": "<Dimension A>",
                "B": "<Dimension B>",
                "C": "<Dimension C>",
                "D": "<Dimension D>",
                "E": "<Dimension E>",
                "F": "<Dimension F>",
                "G": "<Dimension G>",
                "H": "<Dimension H>",
                "J": "<Dimension J>",
                "K": "<Dimension K>",
                "O": "<Dimension O>",
                "R": "<Dimension R>"
            }
            ]
            Use code with caution.
            Json
            Instructions:
            - **GRADE value will always be 400W so without looking at document just write 400W in each entry**
            - Extract values **exactly** as found in the content without modifying, interpreting, or adding any new information.
            - **DO NOT omit any key**. If no value is found for a key, assign null.
            - If a key appears multiple times with different values (e.g., multiple rows in a table), create a **new dictionary entry** in the JSON list for each row, with the corresponding values for all keys. Ensure that each dictionary represents a complete record from the source. **If a row is missing a "Bar Mark", still create a dictionary and fill the "MARK" field with null, extracting any other available data from the row.**
            - Maintain the **exact order** of keys as listed above.
            - **Strictly return JSON format only** with no additional tokens, explanations, or comments.
            - If the output tokens exceed the maximum, omit only the last complete dictionary in the array and return the remaining valid JSON. Do not return a partial dictionary.
            - If you can only extract a single key from a record, create a complete dictionary for that record, filling the other keys with null. Extract all records from the content even if just one key is present.
            - **Fill DWG# with References mentioned on the table. aso ensure that DWG# will only contain one reference (like R1, or R2, or R3 ,....)**
            - MUST: If you do not get a table having header "BENDING DETAILS", then skip it.
            - MUST: **You must carefully see the values of column and those values must be listed in that column, you sometime halucinated and right in some other columns, so consider it carefully**

            Example Output: DO NOT CONFUSE WITH FOLLOWING VALUES, THESE ARE JUST EXAMPLES.

            [
            {
                "DWG #": "A123",
                "ITEM": "10",
                "GRADE": "SS400",
                "WEIGHT": "250 kg",
                "BUNDLE": "BND-45",
                "MARK": "X-202",
                "QUANTITY": "5",
                "SIZE": "200x100",
                "TYPE": "Steel",
                "TOTAL LENGTH": "5000 mm",
                "A": "20 mm",
                "B": "30 mm",
                "C": null,
                "D": null,
                "E": null,
                "F": null,
                "G": null,
                "H": "10 mm",
                "J": null,
                "K": null,
                "O": "15 mm",
                "R": "25 mm"
            },
            {
                "DWG #": "A123",
                "ITEM": "12",
                "GRADE": "SS400",
                "WEIGHT": "150 kg",
                "BUNDLE": "BND-50",
                "MARK": "Y-305",
                "QUANTITY": "3",
                "SIZE": "150x75",
                "TYPE": "Steel",
                "TOTAL LENGTH": "3000 mm",
                "A": null,
                "B": null,
                "C": null,
                "D": "12 mm",
                "E": null,
                "F": null,
                "G": null,
                "H": "5 mm",
                "J": null,
                "K": "8 mm",
                "O": null,
                "R": null
            }
            ]
        **NOTE: MUST FOLLOW THE FOLLOWING POINTS "STRICTLY"**
        - DO NOT STOP AFTER 42 DICTIONARIES, CREATE AS MANY AS YOU CAN (OR AS MANY AVAILABLE IN THE GIVEN DOCUMENT). DO NOT MISS ANY ENTRY.
        - **EVERY EXTRACTED CELL ENTRY MUST BE CONSISTENT WITH THE GIVEN DOCUMENT.**
        - **STRICTLY EXTRACT DATA ONLY FROM THE "BENDING DETAILS" TABLES. IGNORE ALL OTHER TABLES COMPLETELY.** 
        - **IF NO "BENDING DETAILS" TABLE IS FOUND, **MUST** RETURN AN EMPTY JSON ARRAY [].**
        - Please ensure each value is correctly placed under its respective column (A, B, C, D, E, F, G, J, K, O, R). Mistakes in column placement may result in penalties.
        - Always verify that values are inserted into the correct columns. Do not move values from one column to another, even if a previous column is empty, as this would lead to inconsistent data.
    """

    extracted_info = []
    for image_path in image_paths:
        with open(image_path, "rb") as img_file:
            image = Image.open(img_file)

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    temperature=1,
                    max_output_tokens=1000000,
                )
            )

            extracted_info.append(response.text)

    return extracted_info
