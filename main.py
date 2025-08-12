from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from model import process_and_extract
import json
import io
import uuid
import shutil
import os
from datetime import datetime
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def parse_output(response):
    """Parses and structures the extracted data properly into a DataFrame."""
    all_data = []
    
    try:
        for json_string in response:
            json_string = json_string.strip()
            
            try:
                json_string = json_string.replace("```json", "").replace("```", "").strip()
                data = json.loads(json_string)
                if isinstance(data, list):
                    all_data.extend(data) 
            except json.JSONDecodeError:
                continue

        if not all_data:
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
        return None

def transform_dataframe(structured_df):
    """Transforms the structured DataFrame to compute rebar weight."""
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
        return None

    if "TOTAL LENGTH" in structured_df.columns:
        structured_df["TOTAL LENGTH"] = (
            structured_df["TOTAL LENGTH"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        structured_df["TOTAL LENGTH"] = pd.to_numeric(structured_df["TOTAL LENGTH"], errors='coerce')
    else:
        return None

    if "QUANTITY" in structured_df.columns:
        structured_df["QUANTITY"] = pd.to_numeric(structured_df["QUANTITY"], errors='coerce')
        structured_df["QUANTITY"].fillna(1, inplace=True)
    else:
        structured_df["QUANTITY"] = 1

    structured_df.dropna(subset=["Rebar Size", "TOTAL LENGTH"], inplace=True)
    structured_df["Mass per Meter"] = structured_df["Rebar Size"].map(rebar_mass_per_meter)
    structured_df["Total Weight (kg)"] = (structured_df["TOTAL LENGTH"] / 1000) * structured_df["QUANTITY"] * structured_df["Mass per Meter"]

    final_df = structured_df.groupby("Rebar Size", as_index=False)["Total Weight (kg)"].sum()
    return final_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Generate unique folder for processing
            user_folder = f"extracted_images_{uuid.uuid4()}"
            
            # Read file content
            file_bytes = file.read()
            file_buffer = io.BytesIO(file_bytes)
            
            # Process the file
            extracted_data = process_and_extract(file_buffer, file_bytes, file, output_folder=user_folder)
            
            # Parse the output
            structured_df = parse_output(extracted_data)
            
            if structured_df is not None and not structured_df.empty:
                # Transform data
                transformed_df = transform_dataframe(structured_df)
                
                # Convert to JSON for response
                raw_data = structured_df.to_dict('records')
                summary_data = transformed_df.to_dict('records') if transformed_df is not None else []
                
                # Calculate metrics
                total_weight = sum(item['Total Weight (kg)'] for item in summary_data)
                total_types = len(summary_data)
                
                response = {
                    'success': True,
                    'raw_data': raw_data,
                    'summary_data': summary_data,
                    'metrics': {
                        'total_weight': round(total_weight, 2),
                        'total_types': total_types,
                        'total_records': len(raw_data)
                    }
                }
                
                # Cleanup
                shutil.rmtree(user_folder, ignore_errors=True)
                
                return jsonify(response)
            else:
                return jsonify({'error': 'No valid data found in the document'}), 400
                
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/export/<data_type>')
def export_data(data_type):
    # This would need to store the processed data temporarily
    # For now, returning a simple response
    return jsonify({'message': f'Export {data_type} functionality would be implemented here'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
