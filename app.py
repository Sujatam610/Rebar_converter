import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import json
import tempfile
import io
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configure the API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def ensure_arrow_compatibility(df):
    """Ensure DataFrame is compatible with Arrow serialization"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    # Define column type mappings
    integer_columns = ['ITEM', 'QUANTITY']
    float_columns = ['TOTAL LENGTH', 'Total Weight (kg)', 'Mass per Meter', 
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'O', 'R']
    string_columns = ['DWG #', 'GRADE', 'WEIGHT', 'BUNDLE', 'MARK', 'SIZE', 'TYPE', 'Rebar Size']
    
    # Convert integer columns
    for col in integer_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype('int64')
    
    # Convert float columns  
    for col in float_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0).astype('float64')
    
    # Convert string columns
    for col in string_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).replace(['nan', 'None', 'null', '<NA>'], '').astype('object')
    
    # Force all string columns and any remaining object columns to plain object dtype
    for col in df_copy.columns:
        if pd.api.types.is_string_dtype(df_copy[col]) or df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).replace(['nan', 'None', 'null', '<NA>'], '').astype('object')
    
    print(f"🔧 Arrow compatibility ensured: {df_copy.dtypes.to_dict()}")
    return df_copy

# Set page config
st.set_page_config(
    page_title="Rebar AI - Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_rebar_data_from_uploaded_file(uploaded_file):
    """
    Extract rebar data from uploaded PDF file
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Upload to Gemini
        st.info("📤 Uploading file to AI service...")
        sample_pdf = genai.upload_file(path=temp_path, display_name=f"Rebar PDF - {uploaded_file.name}")
        
        # Create extraction prompt using logic from model.py
        prompt = """Extract the details from the provided rebar/bar list content in a **strictly structured JSON format**.

CRITICAL: The document contains multiple rebar/bar tables. You MUST extract ALL records completely.

IMPORTANT TABLE FORMATS TO LOOK FOR:
1. Format 1: Contains columns like "Item | No. Pcs | Size | Length | Mark | Type | A-R dimensions"
2. Format 2: Contains columns like "Bar Mark | Qty | Size | Total length | Type | A-R dimensions"
3. Format 3: Simplified records with ONLY "Qty | Size | Total length | Type" (NO Mark column)
4. Any other table format containing reinforcement details

SCANNING INSTRUCTIONS:
1. Scan the ENTIRE document for ALL tables
2. Look for multiple pages or sections
3. Some tables may be split across pages
4. Continue until ALL records are found (approximately 44+ total)
5. Pay special attention to column names and map them correctly
6. CRITICAL: Don't miss simplified records with only Qty, Size, Total Length, Type!

Follow the exact JSON structure below:
[
{
    "DWG #": "<Drawing Number or Reference>",
    "ITEM": "<Item Number>",
    "GRADE": "400W",
    "WEIGHT": "<Total Weight>",
    "BUNDLE": "<Bundle Number>",
    "MARK": "<Bar Mark or Marking Identifier>",
    "QUANTITY": "<Quantity/QTY/No. Pcs>",
    "SIZE": "<Size Specifications>",
    "TYPE": "<Material Type>",
    "TOTAL LENGTH": "<Total Length/Length>",
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

CRITICAL MAPPING INSTRUCTIONS:
1. **Table Format 1 (From first image/table)**: 
   - Map "Item" → "ITEM"
   - Map "No. Pcs" → "QUANTITY"
   - Map "Size" → "SIZE" (like 10M, 15M)
   - Map "Length" → "TOTAL LENGTH"
   - Map "Mark" → "MARK" (like 10A09, 15A02)
   - Map "Type" → "TYPE" (like T1, 17)
   - Map A-R columns directly to respective dimensions

2. **Table Format 2 (From second image/table)**:
   - Map "Bar Mark" → "MARK" (like 04)
   - Map "Qty" → "QUANTITY" 
   - Map "Size" → "SIZE" (like 15)
   - Map "Total length" → "TOTAL LENGTH"
   - Map "Type" → "TYPE" (like 0)
   - Map A-R columns directly to respective dimensions

3. **Table Format 3 (Simplified Records)**:
   - These records may appear at the end of tables or as standalone rows
   - CRITICAL: These records have ONLY these fields:
     - "Qty" → "QUANTITY" (like 4, 6, 8)
     - "Size" → "SIZE" (like 15)
     - "Total length" → "TOTAL LENGTH" (like 1750)
     - "Type" → "TYPE" (like 0)
   - For these simplified records, set "MARK" to null 
   - These are valid and important records that MUST be included!

4. **Additional Rules**:
   - **GRADE**: Always use "400W" for all entries
   - **DWG #**: Extract drawing reference if available (like R-09)
   - **If no value exists** for a field, use null (not empty string)
   - **Create separate JSON objects** for each table row
   - **EXTRACT ALL records** - there should be 44+ records in this document
   - **Format SIZE properly**: If size is just a number (e.g., "15"), add "M" (e.g., "15M")

FIELD SPECIFIC DETAILS:
1. **MARK**: 
   - First image/table: Values like "10A09", "15A02", etc.
   - Second image/table: Values like "04", etc.
   - IMPORTANT: May be blank/null in simplified records!
2. **SIZE**: 
   - First image/table: Values like "10M", "15M" (already has M suffix)
   - Second image/table: Values like "15" (needs M suffix added → "15M")
   - IMPORTANT: Always add "M" suffix if missing (e.g., "15" → "15M")
3. **QUANTITY/No. Pcs/Qty**: 
   - First image/table: Values like "52", "54", "73"
   - Second image/table: Values like "20", "27", "250"
   - Latest image: Values like "4", "6", "8", etc.
4. **TOTAL LENGTH/Length**: 
   - First image/table: Values like "1020", "1030", "7600" 
   - Second image/table: Values like "300", "400", "500"
   - Latest image: Values like "1750", "1800", "1950", etc.
5. **TYPE**: 
   - First image/table: Values like "T1", "17"
   - Second image/table: Values like "0" 
   - Latest image: Values like "0"
6. **DWG #**:
   - Try to extract from document header/title (e.g., "R-13" or "BARLIST FOR R-09+R-12_04")

RESPONSE FORMAT:
- **MUST start with [** and **end with ]**
- **MUST be valid JSON** - no trailing commas, proper quotes
- **EXTRACT ALL RECORDS** - there should be 44+ records in this document
- **Complete objects only** - better to have all complete records than partial ones
- **If no rebar tables found**, return []

VERIFICATION:
- Count your extracted records before responding - MUST find approximately 44 records total
- DOUBLE-CHECK for any simplified records (Format 3) that might be missed - these have no MARK values
- Verify that ALL formats are correctly identified and mapped, especially simplified records
- Double-check that MARK, SIZE, QUANTITY and TOTAL LENGTH are correctly mapped for each format
- Verify that column names are properly mapped according to the format instructions

EXAMPLES (DO NOT copy these values, extract from actual document):

Example 1 - Record with Bar Mark:
{
    "DWG #": "R-13",
    "ITEM": null,
    "GRADE": "400W",
    "WEIGHT": null,
    "BUNDLE": null,
    "MARK": "04",
    "QUANTITY": "20",
    "SIZE": "15",
    "TYPE": "0",
    "TOTAL LENGTH": "300",
    "A": null,
    "B": null,
    "C": null,
    "D": null,
    "E": null,
    "F": null,
    "G": null,
    "H": null,
    "J": null,
    "K": null,
    "O": null,
    "R": null
}

Example 2 - Simplified record :
{
    "DWG #": "R-13",
    "ITEM": null,
    "GRADE": "400W",
    "WEIGHT": null,
    "BUNDLE": null,
    "MARK": null,
    "QUANTITY": "4",
    "SIZE": "15",
    "TYPE": "0",
    "TOTAL LENGTH": "1750",
    "A": null,
    "B": null,
    "C": null,
    "D": null,
    "E": null,
    "F": null,
    "G": null,
    "H": null,
    "J": null,
    "K": null,
    "O": null,
    "R": null
}
"""

        # Generate content with enhanced settings from model.py
        st.info("🤖 AI analyzing document...")
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        response = model.generate_content(
            [prompt, sample_pdf],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,  # Zero temperature for consistent output
                max_output_tokens=120000,  # Increased to get all 44 records
                top_p=1.0,
                top_k=1
            )
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Parse JSON response with enhanced logic from app.py
        json_text = response.text.strip()
        print(f"🔍 Processing response: {len(json_text)} characters")
        
        # Clean the JSON string more aggressively (from app.py)
        cleaned_json = json_text.replace("```", "").replace("json", "").strip()
        
        # Remove any markdown-like formatting
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json[3:].strip()
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json[:-3].strip()
        
        cleaned_json = cleaned_json.strip()
        
        if not cleaned_json:
            st.error("Empty response from AI")
            return None
        
        # Check if it's a valid JSON array
        if not cleaned_json.startswith('['):
            st.warning("Response doesn't start with '[', attempting to fix...")
            start_idx = cleaned_json.find('[')
            if start_idx != -1:
                cleaned_json = cleaned_json[start_idx:]
            else:
                st.error("No valid JSON array found in response")
                return None
        
        rebar_data = None
        
        # Handle potential truncated JSON (enhanced from app.py)
        if not cleaned_json.endswith(']'):
            st.warning("⚠️ JSON appears truncated, attempting to fix...")
            
            # Try to find the last complete object
            last_complete_brace = cleaned_json.rfind('},')
            if last_complete_brace > 0:
                # Cut off at the last complete object and close the array
                cleaned_json = cleaned_json[:last_complete_brace + 1] + ']'
                print(f"✅ Fixed truncated JSON, keeping {cleaned_json.count('{')} complete records")
                try:
                    rebar_data = json.loads(cleaned_json)
                    st.info(f"✅ Successfully processed {len(rebar_data)} complete records")
                except json.JSONDecodeError:
                    st.warning("Failed to parse fixed JSON, trying individual object extraction...")
                    rebar_data = None
            
            # If the above failed, try more sophisticated parsing (from app.py)
            if rebar_data is None:
                objects = []
                start = 0
                while True:
                    # Look for different patterns of object start
                    patterns = ['{\n    "DWG #":', '{ "DWG #":', '{"DWG #":']
                    obj_start = -1
                    
                    for pattern in patterns:
                        obj_start = cleaned_json.find(pattern, start)
                        if obj_start != -1:
                            break
                    
                    if obj_start == -1:
                        break
                    
                    # Find the matching closing brace
                    brace_count = 0
                    obj_end = obj_start
                    for i, char in enumerate(cleaned_json[obj_start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                obj_end = obj_start + i + 1
                                break
                    
                    if brace_count == 0:
                        obj_text = cleaned_json[obj_start:obj_end]
                        try:
                            obj = json.loads(obj_text)
                            objects.append(obj)
                            print(f"✅ Extracted object: {obj.get('MARK', 'Unknown')}")
                        except json.JSONDecodeError as e:
                            print(f"❌ Failed to parse object: {e}")
                            pass
                    
                    start = obj_end
                
                if objects:
                    rebar_data = objects
                    print(f"✅ Salvaged {len(objects)} complete objects from truncated JSON")
                    st.info(f"✅ Salvaged {len(objects)} complete records")
                else:
                    st.error("❌ Could not extract any complete records")
                    return None
        else:
            # Parse normally if JSON is complete
            try:
                rebar_data = json.loads(cleaned_json)
                print(f"✅ Successfully parsed complete JSON with {len(rebar_data)} records")
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse complete JSON: {e}")
                return None
        
        if not rebar_data:
            st.error("No rebar data found in the document")
            return None
        
        return rebar_data
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def parse_and_structure_data(rebar_data):
    """Parse and structure data using logic from app.py"""
    if not rebar_data:
        return None
    
    # Define expected columns in proper order (flexible for different table types)
    expected_columns = [
        "DWG #", "ITEM", "GRADE", "WEIGHT", "BUNDLE", "MARK", "QUANTITY", "SIZE", "TYPE", "TOTAL LENGTH",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "O", "R"
    ]

    df = pd.DataFrame(rebar_data)
    
    # Handle different column naming variations
    column_mappings = {
        "DWG_NUMBER": "DWG #",
        "BAR_MARK": "MARK",
        "BARMARK": "MARK",
        "QTY": "QUANTITY",
        "NO. PCS": "QUANTITY", 
        "NO.PCS": "QUANTITY",
        "TOTAL_LENGTH": "TOTAL LENGTH",
        "TOTALLENGTH": "TOTAL LENGTH"
    }
    
    # Apply column mappings
    df = df.rename(columns=column_mappings)  # Fix pandas warning by avoiding inplace
    
    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    # Reorder columns according to expected order
    df = df[expected_columns]
    
    # Sort by item number to maintain order
    df = df.sort_values(['ITEM'], na_position='last').reset_index(drop=True)
    
    print(f"✅ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    print(f"📊 Columns: {list(df.columns)}")
    
    return df

def validate_and_clean_dataframe(df):
    """Validate and clean the DataFrame before processing - logic from app.py"""
    try:
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean text fields - remove extra whitespace
        text_columns = ["MARK", "SIZE", "TYPE"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
                df[col] = df[col].replace('None', '')
        
        # Clean numeric fields
        numeric_columns = ["ITEM", "QUANTITY", "TOTAL LENGTH", 
                          "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "O", "R"]
        for col in numeric_columns:
            if col in df.columns:
                # Extract numeric values from strings
                df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').iloc[:, 0]
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ DataFrame validated and cleaned: {len(df)} rows")
        return df
    
    except Exception as e:
        print(f"❌ DataFrame validation error: {e}")
        st.error(f"Error validating data: {e}")
        return df

def transform_dataframe(structured_df):
    """Transform DataFrame with exact logic from app.py"""
    try:
        rebar_mass_per_meter = {
            "10M": 0.785, "15M": 1.570, "20M": 2.355,
            "25M": 3.925, "30M": 5.495, "35M": 7.850,
            "45M": 11.775, "55M": 19.625
        }

        # Create a copy to avoid modifying original
        df = structured_df.copy()
        
        # Clean column names
        df.columns = df.columns.str.strip().str.upper()

        # Handle different table formats for SIZE extraction (exact logic from app.py)
        if "SIZE" in df.columns:
            # First try to extract sizes that already have 'M' 
            df["Rebar Size"] = df["SIZE"].astype(str).str.extract(r'(\d+M)')
            
            # For rows where no 'M' was found, extract just numbers and add 'M'
            mask = df["Rebar Size"].isna()
            if mask.any():
                size_numbers = df.loc[mask, "SIZE"].astype(str).str.extract(r'(\d+)').iloc[:, 0]
                df.loc[mask, "Rebar Size"] = size_numbers + 'M'
                
            print(f"🔧 Size extraction: {df['Rebar Size'].value_counts().to_dict()}")
        else:
            st.error("SIZE column not found in structured data.")
            return None

        # Handle different table formats for TOTAL LENGTH extraction
        if "TOTAL LENGTH" in df.columns:
            df["TOTAL LENGTH"] = (
                df["TOTAL LENGTH"]
                .astype(str)
                .str.replace(r"[^0-9.]", "", regex=True)
            )
            df["TOTAL LENGTH"] = pd.to_numeric(df["TOTAL LENGTH"], errors='coerce')
            print(f"🔧 Length processing: {df['TOTAL LENGTH'].describe()}")
        else:
            st.error("TOTAL LENGTH column not found in structured data.")
            return None

        # Handle different formats for QUANTITY (exact logic from app.py)
        quantity_cols = ["QUANTITY", "QTY", "NO. PCS", "NO.PCS"]
        quantity_col = None
        for col in quantity_cols:
            if col in df.columns:
                quantity_col = col
                break
        
        if quantity_col:
            df["QUANTITY"] = pd.to_numeric(df[quantity_col], errors='coerce')
            df["QUANTITY"] = df["QUANTITY"].fillna(1)  # Fix pandas warning by avoiding inplace
            print(f"🔧 Quantity processing from {quantity_col}: {df['QUANTITY'].describe()}")
        else:
            st.warning("QUANTITY column not found. Assuming quantity of 1 for all items.")
            df["QUANTITY"] = 1

        # Remove rows with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=["Rebar Size", "TOTAL LENGTH"])  # Fix pandas warning by avoiding inplace
        print(f"🔧 Dropped {initial_count - len(df)} rows with missing critical data")
        
        # Map rebar sizes to mass per meter
        df["Mass per Meter"] = df["Rebar Size"].map(rebar_mass_per_meter)
        
        # Show which sizes were not found
        missing_sizes = df[df["Mass per Meter"].isna()]["Rebar Size"].unique()
        if len(missing_sizes) > 0:
            print(f"⚠️ Unknown rebar sizes: {missing_sizes}")
            st.warning(f"Unknown rebar sizes found: {missing_sizes}")

        # Remove rows where rebar size is not found in mass mapping
        before_mass_filter = len(df)
        df = df.dropna(subset=["Mass per Meter"])
        print(f"🔧 Dropped {before_mass_filter - len(df)} rows with unknown rebar sizes")

        # Updated calculation: Convert length to meters, multiply by quantity and mass per meter
        df["Total Weight (kg)"] = (df["TOTAL LENGTH"] / 1000) * df["QUANTITY"] * df["Mass per Meter"]
        
        # Ensure all data types are correct for display and Arrow compatibility
        for col in df.columns:
            if col in ['ITEM', 'QUANTITY']:
                # Convert to integers, handling NaN values
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            elif col in ['TOTAL LENGTH', 'Total Weight (kg)', 'Mass per Meter'] or col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'O', 'R']:
                # Convert to float, handling NaN values
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                # Convert to string and handle None/NaN values
                df[col] = df[col].astype(str).replace(['nan', 'None', 'null'], '')
        
        print(f"✅ DataFrame transformed successfully: {len(df)} rows with weight calculations")
        print(f"📊 Total weight: {df['Total Weight (kg)'].sum():.2f} kg")
        return df
        
    except Exception as e:
        st.error(f"Error transforming DataFrame: {e}")
        print(f"❌ Transform error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def display_summary_metrics(df):
    """Display key metrics in Streamlit - enhanced from app.py"""
    if not df.empty:
        # Check for weight column
        weight_col = None
        for col in ['Total Weight (kg)', 'TOTAL_WEIGHT_KG']:
            if col in df.columns:
                weight_col = col
                break
        
        if weight_col:
            # Remove NaN values for calculations
            valid_df = df.dropna(subset=[weight_col])
            
            if not valid_df.empty:
                total_weight = valid_df[weight_col].sum()
                total_records = len(df)
                
                # Find rebar size column
                size_col = None
                for col in ['Rebar Size', 'REBAR_SIZE']:
                    if col in valid_df.columns:
                        size_col = col
                        break
                
                unique_sizes = len(valid_df[size_col].unique()) if size_col else 0
                heaviest_type = valid_df.loc[valid_df[weight_col].idxmax(), size_col] if size_col and not valid_df.empty else "N/A"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="🏋️ Total Weight",
                        value=f"{total_weight:.2f} kg",
                        delta=f"{total_weight*2.20462:.2f} lbs"
                    )
                
                with col2:
                    st.metric(
                        label="🔧 Rebar Types",
                        value=f"{unique_sizes}",
                        delta="Different sizes"
                    )
                
                with col3:
                    st.metric(
                        label="⚡ Heaviest Type",
                        value=f"{heaviest_type}",
                        delta=f"{valid_df.loc[valid_df[weight_col].idxmax(), weight_col]:.2f} kg" if not valid_df.empty else "0 kg"
                    )
            else:
                st.warning("No valid weight data available for metrics")
        else:
            # Basic metrics without weight
            total_records = len(df)
            st.metric("📊 Total Records", total_records)

def create_csv_download(df, filename_prefix="rebar_data"):
    """Create CSV download functionality"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return csv_data, filename
    except Exception as e:
        st.error(f"Error creating CSV: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏗️ Rebar AI - Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload PDF directly and extract rebar data with AI-powered analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About This Tool")
        st.write("""
        **Direct PDF Analysis**
        - Upload PDF files directly
        - AI processes documents without image conversion
        - Faster and more accurate extraction
        - Automatic weight calculations
        """)
        
        st.subheader("🎯 Supported Tables")
        st.write("""
        - Rebar Schedule
        - Bar List / Barlist  
        - Bending Details
        - Reinforcement Schedule
        """)
        
        st.subheader("📊 Output Features")
        st.write("""
        - Complete raw data
        - Weight calculations
        - Summary statistics
        - CSV download
        """)

    # Main content
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📁 Choose a PDF file containing rebar data",
        type=["pdf"],
        help="Upload a PDF file containing rebar tables, bar lists, or reinforcement schedules"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # File info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 **File:** {uploaded_file.name}")
        with col2:
            st.info(f"📊 **Size:** {uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.info(f"🕒 **Uploaded:** {datetime.now().strftime('%H:%M:%S')}")

        # Process button
        if st.button("🚀 Extract Rebar Data", type="primary"):
            with st.spinner("🔄 Processing PDF with AI..."):
                # Extract data
                rebar_data = extract_rebar_data_from_uploaded_file(uploaded_file)
                
                if rebar_data:
                    print(f"🎯 Extracted {len(rebar_data)} raw records from AI")
                    
                    # Parse and structure data using app.py logic
                    structured_df = parse_and_structure_data(rebar_data)
                    
                    if structured_df is not None:
                        print(f"📊 Structured DataFrame: {len(structured_df)} rows")
                        
                        # Validate and clean the data
                        cleaned_df = validate_and_clean_dataframe(structured_df)
                        print(f"🧹 Cleaned DataFrame: {len(cleaned_df)} rows")
                        
                        # Transform data with weight calculations
                        processed_df = transform_dataframe(cleaned_df)
                        
                        if processed_df is not None:
                            print(f"⚖️ Processed DataFrame with weights: {len(processed_df)} rows")
                            
                            # Instead of using ensure_arrow_compatibility, use a direct approach
                            # to convert data types before storage
                            processed_df_clean = processed_df.copy()
                            cleaned_df_clean = cleaned_df.copy()
                            
                            # Fix data types explicitly to avoid Arrow issues
                            for df_to_fix in [processed_df_clean, cleaned_df_clean]:
                                for col in df_to_fix.columns:
                                    if col in ['ITEM', 'QUANTITY']:
                                        df_to_fix[col] = pd.to_numeric(df_to_fix[col], errors='coerce').fillna(0).astype('int64')
                                    elif col in ['TOTAL LENGTH', 'Total Weight (kg)', 'Mass per Meter'] or col.upper() in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'O', 'R']:
                                        df_to_fix[col] = pd.to_numeric(df_to_fix[col], errors='coerce').fillna(0.0).astype('float64')
                                    else:
                                        # Convert to simple string
                                        df_to_fix[col] = df_to_fix[col].fillna('').astype(str)
                            
                            # Store in session state with proper data types
                            st.session_state['df'] = processed_df_clean
                            st.session_state['raw_df'] = cleaned_df_clean
                            st.session_state['original_count'] = len(rebar_data)
                            st.session_state['extraction_successful'] = True
                            
                            # Show detailed extraction info
                            st.success(f"✅ Successfully extracted and processed {len(processed_df)} rebar records!")
                            st.info(f"📋 Processing Pipeline: {len(rebar_data)} raw → {len(structured_df)} structured → {len(cleaned_df)} cleaned → {len(processed_df)} with weights")
                        else:
                            st.session_state['extraction_successful'] = False
                            st.error("❌ Failed to process extracted data")
                    else:
                        st.session_state['extraction_successful'] = False
                        st.error("❌ Failed to structure extracted data")
                else:
                    st.session_state['extraction_successful'] = False
                    st.error("❌ Failed to extract rebar data from the PDF")

    # Display results if extraction was successful
    if st.session_state.get('extraction_successful', False) and 'df' in st.session_state:
        df = st.session_state['df']
        raw_df = st.session_state.get('raw_df', df)
        
        # Display summary metrics
        st.subheader("📊 Summary Metrics")
        display_summary_metrics(df)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Weight Summary", "📥 Downloads"])
        
        with tab1:
            st.subheader("📋 Extracted Raw Data")
            
            # Display data with filters
            col1, col2 = st.columns(2)
            with col1:
                size_col = None
                for col in ['Rebar Size', 'REBAR_SIZE', 'SIZE']:
                    if col in df.columns:
                        size_col = col
                        break
                
                if size_col:
                    sizes = df[size_col].dropna().unique()
                    selected_sizes = st.multiselect("Filter by Rebar Size", sizes, default=sizes)
                    if selected_sizes:
                        filtered_df = df[df[size_col].isin(selected_sizes)]
                    else:
                        filtered_df = df
                else:
                    filtered_df = df
            
            with col2:
                dwg_col = None
                for col in ['DWG #', 'DWG_NUMBER']:
                    if col in df.columns:
                        dwg_col = col
                        break
                
                if dwg_col:
                    drawings = df[dwg_col].dropna().unique()
                    selected_drawings = st.multiselect("Filter by Drawing", drawings, default=drawings)
                    if selected_drawings:
                        filtered_df = filtered_df[filtered_df[dwg_col].isin(selected_drawings)]
            
            # Convert to display-friendly format and fix Arrow serialization issues
            display_df = filtered_df.copy()
            
            # Manual conversion to explicit types that Streamlit can handle
            for col in display_df.columns:
                if col in ['ITEM', 'QUANTITY']:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype('int64')
                elif col in ['TOTAL LENGTH', 'Total Weight (kg)', 'Mass per Meter'] or col.upper() in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'O', 'R']:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0.0).astype('float64')
                else:
                    # Convert any other column to simple string
                    display_df[col] = display_df[col].fillna('').astype(str)
            
            # Directly display the converted DataFrame
            try:
                st.dataframe(display_df, use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error displaying data: {str(e)}")
                # Fallback to simplified display
                st.write("Using simplified data display due to Arrow conversion error:")
                simplified_df = display_df.astype(str)
                st.dataframe(simplified_df, use_container_width=True, height=400)
            
            # Data statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Displayed Records", len(filtered_df))
            with col2:
                st.metric("Total Records", len(df))
            with col3:
                mark_col = None
                for col in ['MARK', 'BAR_MARK']:
                    if col in filtered_df.columns:
                        mark_col = col
                        break
                if mark_col:
                    non_null_marks = filtered_df[mark_col].dropna().shape[0]
                    st.metric("Records with Mark", non_null_marks)
            with col4:
                weight_col = None
                for col in ['Total Weight (kg)', 'TOTAL_WEIGHT_KG']:
                    if col in filtered_df.columns:
                        weight_col = col
                        break
                if weight_col:
                    total_weight = filtered_df[weight_col].sum()
                    st.metric("Filtered Weight", f"{total_weight:.2f} kg")
        
        with tab2:
            st.subheader("📈 Weight Summary by Rebar Size")
            
            # Find weight and size columns
            weight_col = None
            size_col = None
            quantity_col = None
            
            for col in ['Total Weight (kg)', 'TOTAL_WEIGHT_KG']:
                if col in df.columns:
                    weight_col = col
                    break
            
            for col in ['Rebar Size', 'REBAR_SIZE']:
                if col in df.columns:
                    size_col = col
                    break
            
            for col in ['QUANTITY', 'QUANTITY_NUM']:
                if col in df.columns:
                    quantity_col = col
                    break
            
            if weight_col and size_col:
                # Create summary DataFrame
                agg_dict = {weight_col: 'sum', 'ITEM': 'count'}
                if quantity_col:
                    agg_dict[quantity_col] = 'sum'
                
                summary_df = df.groupby(size_col).agg(agg_dict).round(2)
                
                # Rename columns
                new_columns = ['Total Weight (kg)', 'Number of Items']
                if quantity_col:
                    new_columns.insert(1, 'Total Quantity')
                summary_df.columns = new_columns
                
                summary_df['Total Weight (lbs)'] = (summary_df['Total Weight (kg)'] * 2.20462).round(2)
                
                # Reorder columns
                if 'Total Quantity' in summary_df.columns:
                    summary_df = summary_df[['Number of Items', 'Total Quantity', 'Total Weight (kg)', 'Total Weight (lbs)']]
                else:
                    summary_df = summary_df[['Number of Items', 'Total Weight (kg)', 'Total Weight (lbs)']]
                
                # Ensure Arrow compatibility for summary DataFrame
                summary_df_display = summary_df.reset_index()
                
                # Manual conversion to explicit types that Streamlit can handle
                for col in summary_df_display.columns:
                    if col in ['Number of Items', 'Total Quantity']:
                        summary_df_display[col] = pd.to_numeric(summary_df_display[col], errors='coerce').fillna(0).astype('int64')
                    elif col in ['Total Weight (kg)', 'Total Weight (lbs)']:
                        summary_df_display[col] = pd.to_numeric(summary_df_display[col], errors='coerce').fillna(0.0).astype('float64')
                    else:
                        # Convert any other column to simple string
                        summary_df_display[col] = summary_df_display[col].fillna('').astype(str)
                
                # Directly display the converted DataFrame
                try:
                    st.dataframe(summary_df_display, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying summary data: {str(e)}")
                    # Fallback to simplified display
                    st.write("Using simplified data display due to Arrow conversion error:")
                    simplified_df = summary_df_display.astype(str)
                    st.dataframe(simplified_df, use_container_width=True)
                
                # Create charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Weight Distribution")
                    st.bar_chart(summary_df['Total Weight (kg)'])
                
                with col2:
                    st.subheader("Item Count Distribution")
                    st.bar_chart(summary_df['Number of Items'])
            else:
                st.warning("Weight calculation data not available")
        
        with tab3:
            st.subheader("📥 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Complete Processed Data**")
                processed_csv, processed_filename = create_csv_download(df, "rebar_complete_processed")
                if processed_csv:
                    st.download_button(
                        label="📄 Download Complete Data CSV",
                        data=processed_csv,
                        file_name=processed_filename,
                        mime="text/csv",
                        help="Download all extracted data including weight calculations"
                    )
                    st.success(f"✅ Complete data ready: {len(df)} records")
            
            with col2:
                # Find weight and size columns for summary
                weight_col = None
                size_col = None
                
                for col in ['Total Weight (kg)', 'TOTAL_WEIGHT_KG']:
                    if col in df.columns:
                        weight_col = col
                        break
                
                for col in ['Rebar Size', 'REBAR_SIZE']:
                    if col in df.columns:
                        size_col = col
                        break
                
                if weight_col and size_col:
                    st.write("**Weight Summary**")
                    summary_df = df.groupby(size_col).agg({
                        weight_col: 'sum',
                        'ITEM': 'count'
                    }).round(2)
                    summary_df.columns = ['Total Weight (kg)', 'Number of Items']
                    
                    summary_csv, summary_filename = create_csv_download(summary_df, "rebar_weight_summary")
                    if summary_csv:
                        st.download_button(
                            label="📊 Download Weight Summary CSV",
                            data=summary_csv,
                            file_name=summary_filename,
                            mime="text/csv",
                            help="Download weight summary grouped by rebar size"
                        )
                        st.success(f"✅ Summary ready: {len(summary_df)} rebar types")
            
            # Raw data download option
            if 'raw_df' in st.session_state:
                st.write("**Raw Extracted Data (Before Processing)**")
                raw_csv, raw_filename = create_csv_download(raw_df, "rebar_raw_extracted")
                if raw_csv:
                    st.download_button(
                        label="📋 Download Raw Data CSV",
                        data=raw_csv,
                        file_name=raw_filename,
                        mime="text/csv",
                        help="Download raw extracted data before processing"
                    )
            
            # Additional information
            st.info("""
            📋 **Download Information:**
            - **Complete Data**: All extracted fields with calculated weights and processing
            - **Weight Summary**: Totals grouped by rebar size
            - **Raw Data**: Original extracted data before any processing
            - **Timestamps**: All files include date/time stamps
            - **Format**: Standard CSV compatible with Excel and other tools
            """)

        # Additional debugging info (can be hidden in production)
        with st.expander("🔧 Debug Information", expanded=False):
            st.write("**Processing Pipeline:**")
            if 'original_count' in st.session_state:
                st.write(f"- Original AI Response: {st.session_state['original_count']} records")
            if 'raw_df' in st.session_state:
                st.write(f"- After Structuring: {len(st.session_state['raw_df'])} records")
            st.write(f"- Final Processed: {len(df)} records")
            
            st.write("**DataFrame Info:**")
            st.write(f"- Shape: {df.shape}")
            st.write(f"- Columns: {list(df.columns)}")
            
            if not df.empty:
                st.write("**Column Data Types:**")
                dtype_df = df.dtypes.reset_index()
                dtype_df.columns = ['Column', 'Data Type']
                
                # Convert to strings to ensure compatibility
                dtype_df['Column'] = dtype_df['Column'].astype(str)
                dtype_df['Data Type'] = dtype_df['Data Type'].astype(str)
                
                try:
                    st.dataframe(dtype_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying dtype information: {str(e)}")
                    st.dataframe(dtype_df.astype(str), use_container_width=True)
                
                st.write("**Sample Data (First 3 rows):**")
                sample_df = df.head(3).copy()
                
                # Manual conversion to explicit types that Streamlit can handle
                for col in sample_df.columns:
                    if col in ['ITEM', 'QUANTITY']:
                        sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(0).astype('int64')
                    elif col in ['TOTAL LENGTH', 'Total Weight (kg)', 'Mass per Meter'] or col.upper() in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'O', 'R']:
                        sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce').fillna(0.0).astype('float64')
                    else:
                        # Convert any other column to simple string
                        sample_df[col] = sample_df[col].fillna('').astype(str)
                
                try:
                    st.dataframe(sample_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying sample data: {str(e)}")
                    # Fallback to simplified display
                    simplified_df = sample_df.astype(str)
                    st.dataframe(simplified_df, use_container_width=True)
                
                # Show size and weight mapping
                if 'Rebar Size' in df.columns:
                    st.write("**Rebar Size Distribution:**")
                    size_counts = df['Rebar Size'].value_counts()
                    size_counts_df = size_counts.to_frame('Count')
                    
                    # Convert to string to ensure compatibility
                    size_counts_df.index = size_counts_df.index.astype(str)
                    size_counts_df['Count'] = size_counts_df['Count'].astype('int64')
                    
                    try:
                        st.dataframe(size_counts_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying size counts: {str(e)}")
                        st.dataframe(size_counts_df.astype(str), use_container_width=True)
                
                if 'Total Weight (kg)' in df.columns:
                    st.write("**Weight Statistics:**")
                    weight_stats = df['Total Weight (kg)'].describe()
                    weight_stats_df = weight_stats.to_frame('Value')
                    
                    # Convert to ensure compatibility
                    weight_stats_df.index = weight_stats_df.index.astype(str)
                    weight_stats_df['Value'] = weight_stats_df['Value'].astype('float64')
                    
                    try:
                        st.dataframe(weight_stats_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying weight stats: {str(e)}")
                        st.dataframe(weight_stats_df.astype(str), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("🏗️ **Rebar AI Direct Analyzer** - Powered by Google Gemini AI")

if __name__ == "__main__":
    # Initialize session state
    if 'extraction_successful' not in st.session_state:
        st.session_state['extraction_successful'] = False
    
    main()
