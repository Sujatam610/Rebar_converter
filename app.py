import streamlit as st
import pandas as pd
from model import process_and_extract
import json
import io
import uuid
import shutil
from datetime import datetime

# Set page config for better layout
st.set_page_config(
    page_title="Rebar AI - Document Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def display_metrics(final_df):
    """Display key metrics"""
    if not final_df.empty:
        total_weight = final_df['Total Weight (kg)'].sum()
        total_types = len(final_df)
        heaviest_type = final_df.loc[final_df['Total Weight (kg)'].idxmax(), 'Rebar Size']
        
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
                value=f"{total_types}",
                delta="Different sizes"
            )
        
        with col3:
            st.metric(
                label="⚡ Heaviest Type",
                value=f"{heaviest_type}",
                delta=f"{final_df.loc[final_df['Total Weight (kg)'].idxmax(), 'Total Weight (kg)']:.2f} kg"
            )

def parse_output(response):
    """Parses and structures the extracted data properly into a DataFrame with expected columns."""
    all_data = []
    
    try:
        for json_string in response:
            json_string = json_string.strip()
            
            try:
                data = json.loads(json_string.replace("```", "").replace("json", ""))
                if isinstance(data, list):
                    all_data.extend(data) 
                else:
                    st.warning(f"Unexpected structure in JSON string: {json_string}. Expected a list.")

            except json.JSONDecodeError as e:
                st.error(f"Failed to decode JSON string: {json_string}. Error: {e}")
                return None

        if not all_data:
            st.warning("No data was extracted or parsed.")
            return None

        expected_columns = [
            "DWG #", "ITEM", "GRADE", "WEIGHT", "BUNDLE", "MARK", "QUANTITY", "SIZE", "TYPE", "TOTAL LENGTH",
            "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "O", "R"
        ]

        df = pd.DataFrame(all_data)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        return df[expected_columns]

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def transform_dataframe(structured_df):
    """Transforms the structured DataFrame to compute rebar weight based on size, total length, and quantity."""
    rebar_mass_per_meter = {
        "10M": 0.785, "15M": 1.570, "20M": 2.355,
        "25M": 3.925, "30M": 5.495, "35M": 7.850,
        "45M": 11.775, "55M": 19.625
    }

    structured_df = structured_df.rename(columns=lambda x: x.strip().upper())

    if "SIZE" in structured_df.columns:
        structured_df["Rebar Size"] = structured_df["SIZE"].astype(str).str.extract(r'(\d+M)')
    else:
        st.error("Rebar Size column not found in structured data.")
        return None

    if "TOTAL LENGTH" in structured_df.columns:
        structured_df["TOTAL LENGTH"] = (
            structured_df["TOTAL LENGTH"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        structured_df["TOTAL LENGTH"] = pd.to_numeric(structured_df["TOTAL LENGTH"], errors='coerce')
    else:
        st.error("TOTAL LENGTH column not found in structured data.")
        return None

    # Ensure QUANTITY is numeric
    if "QUANTITY" in structured_df.columns:
        structured_df["QUANTITY"] = pd.to_numeric(structured_df["QUANTITY"], errors='coerce')
        structured_df["QUANTITY"].fillna(1, inplace=True)  # Default to 1 if missing or invalid
    else:
        st.warning("QUANTITY column not found. Assuming quantity of 1 for all items.")
        structured_df["QUANTITY"] = 1

    structured_df.dropna(subset=["Rebar Size", "TOTAL LENGTH"], inplace=True)
    structured_df["Mass per Meter"] = structured_df["Rebar Size"].map(rebar_mass_per_meter)

    # Updated calculation: Convert length to meters, multiply by quantity and mass per meter
    structured_df["Total Weight (kg)"] = (structured_df["TOTAL LENGTH"] / 1000) * structured_df["QUANTITY"] * structured_df["Mass per Meter"]

    final_df = structured_df.groupby("Rebar Size", as_index=False)["Total Weight (kg)"].sum()
    return final_df
    

def main():
    # Header
    st.markdown('<h1 class="main-header">🏗️ Rebar AI Document Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Extract and analyze rebar details from construction drawings</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Processing options
        st.subheader("Processing Options")
        batch_size = st.slider("Batch Size", min_value=1, max_value=10, value=2, 
                              help="Number of images to process in each batch")
        
        # About section
        st.subheader("ℹ️ About")
        st.write("""
        This tool uses AI to extract rebar information from construction drawings including:
        - Drawing numbers
        - Item details  
        - Dimensions (A-R)
        - Quantities and weights
        - Bending details
        """)

    # Main content
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📁 Choose a construction drawing file",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload a PDF or image file containing rebar bending details"
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

        # Generate unique folder for each user session
        user_folder = f"extracted_images_{uuid.uuid4()}"

        file_bytes = uploaded_file.read()
        file_buffer = io.BytesIO(file_bytes)  # Ensure bytes-like object

        # Processing with progress
        with st.spinner("🔄 Processing document... This may take a few moments"):
            progress_bar = st.progress(0)
            progress_bar.progress(25)
            
            extracted_data = process_and_extract(file_buffer, file_bytes, uploaded_file, 
                                               output_folder=user_folder, batch_size=batch_size)
            progress_bar.progress(75)
            
            structured_df = parse_output(extracted_data)
            progress_bar.progress(100)

        if structured_df is not None and not structured_df.empty:
            st.success(f"✅ Successfully extracted {len(structured_df)} records!")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📋 Raw Data", "📊 Summary"])
            
            with tab1:
                st.subheader("📋 Extracted Raw Data")
                st.dataframe(structured_df, use_container_width=True)
                
            with tab2:
                transformed_df = transform_dataframe(structured_df)
                
                if transformed_df is not None:
                    # Display metrics
                    display_metrics(transformed_df)
                    
                    st.subheader("📊 Weight Summary by Rebar Size")
                    st.dataframe(transformed_df, use_container_width=True)
                else:
                    st.warning("⚠️ Transformation failed. Check extracted data.")
        else:
            st.error("❌ No valid data found in the document. Please check if the document contains 'BENDING DETAILS' tables.")
        
        # Cleanup
        shutil.rmtree(user_folder, ignore_errors=True)

    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()
