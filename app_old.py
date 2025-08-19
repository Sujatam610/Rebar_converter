import streamlit as st
import pandas as pd
from model import process_and_extract
import json
import io
import uuid
import shutil
from datetime import datetime
import csv

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
    """Display key metrics for any data"""
    if not final_df.empty:
        total_rows = len(final_df)
        total_columns = len(final_df.columns)
        
        # Find columns with data
        data_columns = []
        for col in final_df.columns:
            non_null_count = final_df[col].notna().sum()
            if non_null_count > 0:
                data_columns.append(f"{col}: {non_null_count}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Total Rows",
                value=f"{total_rows}",
                delta="Data records"
            )
        
        with col2:
            st.metric(
                label="� Total Columns", 
                value=f"{total_columns}",
                delta="Data fields"
            )
        
        with col3:
            filled_columns = len([col for col in final_df.columns if final_df[col].notna().sum() > 0])
            st.metric(
                label="✅ Columns with Data",
                value=f"{filled_columns}",
                delta=f"{(filled_columns/total_columns*100):.0f}% filled"
            )

def parse_output(response):
    """Parses and structures the extracted data into a DataFrame with dynamic columns."""
    all_data = []
    page_order = []
    
    print(f"🔍 Parsing {len(response)} response chunks...")
    
    try:
        for i, json_string in enumerate(response):
            json_string = json_string.strip()
            print(f"📄 Processing chunk {i+1}: {len(json_string)} characters")
            page_order.append(f"Page {i+1}")
            
            try:
                cleaned_json = json_string.replace("```", "").replace("json", "").strip()
                if not cleaned_json or cleaned_json == "[]":
                    print(f"   ⚠️ Chunk {i+1} is empty, skipping")
                    continue
                    
                data = json.loads(cleaned_json)
                
                if isinstance(data, list):
                    print(f"   ✅ Chunk {i+1} contains {len(data)} records")
                    all_data.extend(data) 
                elif isinstance(data, dict):
                    all_data.append(data)
                    print(f"   ✅ Chunk {i+1} contains 1 record")
                else:
                    st.warning(f"Unexpected structure in JSON chunk {i+1}: Expected a list or dict, got {type(data)}")
                    print(f"   ⚠️ Chunk {i+1} has unexpected structure: {type(data)}")

            except json.JSONDecodeError as e:
                st.error(f"Failed to decode JSON chunk {i+1}: {e}")
                print(f"   ❌ Chunk {i+1} JSON decode failed: {e}")
                print(f"   📝 Content preview: {cleaned_json[:200]}...")

        print(f"🎯 Total records collected: {len(all_data)} from {len(response)} chunks")
        print(f"📋 Page processing order: {', '.join(page_order)}")
        
        if not all_data:
            st.warning("No tabular data was found in the document. Please ensure the document contains tables.")
            print("⚠️ No data extracted - all chunks were empty")
            return None

        # Create DataFrame with dynamic columns
        df = pd.DataFrame(all_data)
        
        # Clean column names for better CSV compatibility
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        print(f"✅ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        print(f"📊 Columns: {list(df.columns)}")
        
        return df

    except Exception as e:
        st.error(f"An unexpected error occurred during parsing: {e}")
        print(f"❌ Unexpected parsing error: {e}")
        return None

def create_csv_download(df, filename_prefix="rebar_data"):
    """Create CSV download functionality"""
    try:
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return csv_data, filename
    except Exception as e:
        st.error(f"Error creating CSV: {e}")
        return None, None

def validate_and_clean_dataframe(df):
    """Validate and clean the DataFrame with dynamic columns"""
    try:
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean all columns dynamically
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
                df[col] = df[col].replace('None', '')
                df[col] = df[col].replace('', None)
            else:  # Numeric columns
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ DataFrame validated and cleaned: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    except Exception as e:
        print(f"❌ DataFrame validation error: {e}")
        st.error(f"Error validating data: {e}")
        return df
def analyze_dataframe(df):
    """Analyze the DataFrame and provide insights"""
    try:
        if df.empty:
            return None
            
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_data': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'text_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Calculate statistics for numeric columns
        numeric_stats = {}
        for col in analysis['numeric_columns']:
            if df[col].notna().sum() > 0:
                numeric_stats[col] = {
                    'count': df[col].notna().sum(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'sum': df[col].sum()
                }
        
        analysis['numeric_stats'] = numeric_stats
        
        print(f"✅ DataFrame analysis completed")
        return analysis
        
    except Exception as e:
        st.error(f"Error analyzing DataFrame: {e}")
        print(f"❌ Analysis error: {e}")
        return None
    

def main():
    # Header
    st.markdown('<h1 class="main-header">🏗️ Rebar AI - Document Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Extract tabular data from any document and convert to CSV</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Processing options
        st.subheader("Processing Options")
        batch_size = st.slider("Batch Size", min_value=1, max_value=5, value=1, 
                              help="Number of images to process in each batch. Lower values reduce timeout risk.")
        
        # API settings
        with st.expander("⚙️ Advanced Settings"):
            st.info("These settings help handle timeout issues")
            max_retries = st.slider("Max Retries", min_value=1, max_value=5, value=3,
                                  help="Number of retry attempts for failed API calls")
            st.session_state['max_retries'] = max_retries
        
        # About section
        st.subheader("ℹ️ About")
        st.write("""
        This tool uses AI to extract tabular data from any document including:
        - Tables with any structure
        - Dynamic column detection
        - Automatic data type recognition
        - CSV export functionality
        - Multi-page support
        """)

    # Main content
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📁 Choose any document with tables",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload a PDF or image file containing tabular data"
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
            status_text = st.empty()
            
            progress_bar.progress(25)
            status_text.text("📄 Extracting images from document...")
            
            try:
                max_retries = st.session_state.get('max_retries', 3)
                extracted_data = process_and_extract(file_buffer, file_bytes, uploaded_file, 
                                                   output_folder=user_folder, batch_size=batch_size,
                                                   max_retries=max_retries)
                progress_bar.progress(75)
                status_text.text("🤖 AI processing completed, parsing results...")
                
                structured_df = parse_output(extracted_data)
                progress_bar.progress(100)
                status_text.text("✅ Processing completed successfully!")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("")
                st.error(f"❌ Processing failed: {str(e)}")
                st.info("💡 **Troubleshooting tips:**")
                st.write("- Try reducing the batch size in the sidebar")
                st.write("- Ensure the document contains tables")
                st.write("- Check your internet connection")
                st.write("- The document might be too large or complex")
                return

        if structured_df is not None and not structured_df.empty:
            st.success(f"✅ Successfully extracted {len(structured_df)} records from {len(structured_df.columns)} columns!")
            
            # Validate and clean the data
            cleaned_df = validate_and_clean_dataframe(structured_df)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Analysis", "📥 Downloads"])
            
            with tab1:
                st.subheader("📋 Extracted Table Data")
                st.dataframe(cleaned_df, use_container_width=True)
                
                # Show data statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(cleaned_df))
                with col2:
                    st.metric("Total Columns", len(cleaned_df.columns))
                with col3:
                    filled_cells = cleaned_df.notna().sum().sum()
                    total_cells = len(cleaned_df) * len(cleaned_df.columns)
                    st.metric("Data Completeness", f"{(filled_cells/total_cells*100):.1f}%")
                
            with tab2:
                analysis = analyze_dataframe(cleaned_df)
                
                if analysis:
                    # Display metrics
                    display_metrics(cleaned_df)
                    
                    st.subheader("📊 Data Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Numeric Columns:**")
                        if analysis['numeric_columns']:
                            for col in analysis['numeric_columns']:
                                if col in analysis['numeric_stats']:
                                    stats = analysis['numeric_stats'][col]
                                    st.write(f"- **{col}**: {stats['count']} values, Sum: {stats['sum']:.2f}")
                        else:
                            st.write("No numeric columns found")
                    
                    with col2:
                        st.write("**Text Columns:**")
                        if analysis['text_columns']:
                            for col in analysis['text_columns']:
                                unique_count = cleaned_df[col].nunique()
                                st.write(f"- **{col}**: {unique_count} unique values")
                        else:
                            st.write("No text columns found")
                else:
                    st.warning("⚠️ Analysis failed. Check extracted data.")
            
            with tab3:
                st.subheader("📥 Download Options")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Complete Data CSV**")
                    raw_csv, raw_filename = create_csv_download(cleaned_df, "extracted_table_data")
                    if raw_csv:
                        st.download_button(
                            label="� Download Complete CSV",
                            data=raw_csv,
                            file_name=raw_filename,
                            mime="text/csv",
                            help="Download all extracted data as CSV"
                        )
                        st.success(f"✅ Data ready: {len(cleaned_df)} records, {len(cleaned_df.columns)} columns")
                
                with col2:
                    st.write("**Data Summary**")
                    st.write(f"- **Rows**: {len(cleaned_df)}")
                    st.write(f"- **Columns**: {len(cleaned_df.columns)}")
                    st.write(f"- **Column Names**: {', '.join(cleaned_df.columns)}")
                
                # Additional information
                st.info("""
                📋 **CSV File Contents:**
                - **Dynamic Columns**: Column names extracted directly from the table headers
                - **All Data Types**: Numbers, text, and mixed data preserved
                - **Multi-page**: Data from all pages combined
                - **Clean Format**: Ready for analysis in Excel, Google Sheets, or other tools
                """)
                
        else:
            st.error("❌ No valid data found in the document. Please check if the document contains tabular data.")
            st.info("""
            🔍 **Troubleshooting:**
            - Ensure the document contains tables with clear structure
            - Check if the document is readable (not scanned image)
            - Try uploading a different page or document
            - Verify the document quality and resolution
            """)
        
        # Cleanup
        shutil.rmtree(user_folder, ignore_errors=True)

    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()
