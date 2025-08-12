import streamlit as st
import pandas as pd
from model import process_and_extract
import json
import io
import uuid
import shutil
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

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

def parse_output(response):
    """Parses and structures the extracted data properly into a DataFrame with expected columns."""
    all_data = []
    
    try:
        for json_string in response:
            json_string = json_string.strip()
            
            try:
                # Clean up the JSON string
                json_string = json_string.replace("```json", "").replace("```", "").strip()
                data = json.loads(json_string)
                if isinstance(data, list):
                    all_data.extend(data) 
                else:
                    st.warning(f"Unexpected structure in JSON string. Expected a list.")

            except json.JSONDecodeError as e:
                st.error(f"Failed to decode JSON string. Error: {e}")
                continue

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

    structured_df = structured_df.copy()
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
        structured_df["QUANTITY"].fillna(1, inplace=True)
    else:
        st.warning("QUANTITY column not found. Assuming quantity of 1 for all items.")
        structured_df["QUANTITY"] = 1

    structured_df.dropna(subset=["Rebar Size", "TOTAL LENGTH"], inplace=True)
    structured_df["Mass per Meter"] = structured_df["Rebar Size"].map(rebar_mass_per_meter)

    # Calculate total weight
    structured_df["Total Weight (kg)"] = (structured_df["TOTAL LENGTH"] / 1000) * structured_df["QUANTITY"] * structured_df["Mass per Meter"]

    final_df = structured_df.groupby("Rebar Size", as_index=False)["Total Weight (kg)"].sum()
    return final_df, structured_df

def create_visualizations(final_df, detailed_df):
    """Create visualizations for the rebar data"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Weight Distribution by Rebar Size")
        if not final_df.empty:
            fig_pie = px.pie(final_df, values='Total Weight (kg)', names='Rebar Size', 
                           title="Total Weight Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Weight by Rebar Size")
        if not final_df.empty:
            fig_bar = px.bar(final_df, x='Rebar Size', y='Total Weight (kg)',
                           title="Total Weight by Rebar Size",
                           color='Total Weight (kg)',
                           color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)

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

def download_excel(df, filename):
    """Create download link for Excel file"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Rebar Data')
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download Excel Report</a>'
    return href

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
        
        st.subheader("📁 File Information")
        st.info("Supported formats: PDF, PNG, JPG, JPEG")
        
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
        file_buffer = io.BytesIO(file_bytes)

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
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Raw Data", "📊 Summary", "📈 Visualizations", "💾 Export"])
            
            with tab1:
                st.subheader("📋 Extracted Raw Data")
                st.dataframe(structured_df, use_container_width=True)
                
            with tab2:
                transformed_result = transform_dataframe(structured_df)
                if transformed_result is not None:
                    final_df, detailed_df = transformed_result
                    
                    # Display metrics
                    display_metrics(final_df)
                    
                    st.subheader("📊 Weight Summary by Rebar Size")
                    st.dataframe(final_df, use_container_width=True)
                else:
                    st.warning("⚠️ Transformation failed. Check extracted data.")
            
            with tab3:
                if 'final_df' in locals() and not final_df.empty:
                    create_visualizations(final_df, detailed_df)
                else:
                    st.warning("No data available for visualization")
            
            with tab4:
                st.subheader("💾 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV download for raw data
                    csv = structured_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Raw Data (CSV)",
                        data=csv,
                        file_name=f"rebar_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Excel download for summary
                    if 'final_df' in locals():
                        st.markdown(
                            download_excel(final_df, f"rebar_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"),
                            unsafe_allow_html=True
                        )
        else:
            st.error("❌ No valid data found in the document. Please check if the document contains 'BENDING DETAILS' tables.")
        
        # Cleanup
        shutil.rmtree(user_folder, ignore_errors=True)

    # Footer
    st.markdown("---")
    st.markdown("**Rebar AI** - Powered by Google Gemini AI | Built with Streamlit")

if __name__ == "__main__":
    main()
