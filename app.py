import streamlit as st
import pandas as pd
import requests
import json
import time
from io import StringIO, BytesIO
import asyncio
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Product Categorization Tool",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FASTAPI_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .batch-result {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def check_fastapi_health():
    """Check if FastAPI server is running"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def validate_csv_data(df):
    """Validate uploaded CSV data"""
    errors = []
    
    if df.empty:
        errors.append("CSV file is empty")
        return errors
    
    if len(df.columns) == 0:
        errors.append("CSV file has no columns")
        return errors
    
    # Check for minimum required columns
    if len(df.columns) < 2:
        errors.append("CSV file should have at least 2 columns")
    
    return errors

def format_categorization_result(result):
    """Format categorization result for display"""
    if not result.get('results'):
        return None
    
    formatted_results = []
    for item in result['results']:
        # Format embedding info for display
        embedding_info = "None"
        if item.get('embedding'):
            embedding_length = len(item['embedding'])
            embedding_info = f"Vector ({embedding_length}D)"

        formatted_results.append({
            'Product': item.get('name', 'Unknown'),
            'Category': item.get('predicted_category', 'Unknown'),
            'Confidence': f"{item.get('confidence', 0):.2%}",
            'Description': item.get('description', 'No description available'),
            'Embedding': embedding_info,
            'Reasoning': item.get('reasoning', '')
        })
    
    return pd.DataFrame(formatted_results)

def stream_categorization_results(data, toggle, product_column):
    """Stream categorization results from FastAPI"""
    try:
        # Convert data to FurnitureItem format
        furniture_items = []
        for item in data:
            furniture_item = {
                "Product_Name": str(item.get('product_name', item.get('Product_Name', 'Unknown'))),
                "Type": item.get('Type'),
                "Category": item.get('Category'),
                "Style": item.get('Style'),
                "Tags": item.get('Tags'),
                "Price_Range_USD": item.get('Price_Range_USD'),
                "Product_URL": item.get(product_column) if toggle == 0 else None
            }
            furniture_items.append(furniture_item)
        
        payload = {
            "data": furniture_items,
            "toggle": toggle
        }
        
        response = requests.post(
            f"{FASTAPI_URL}/classify-category",
            json=payload,
            stream=True,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return
        
        # Initialize containers for streaming results
        status_container = st.container()
        results_container = st.container()
        
        all_results = []
        batch_count = 0
        
        with status_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data_str = line[6:]  # Remove 'data: ' prefix
                        result = json.loads(data_str)
                        
                        if result.get('processing_type') == 'complete':
                            status_text.success(f"✅ Processing complete! Total items processed: {result.get('total_processed', 0)}")
                            progress_bar.progress(1.0)
                            break
                        
                        elif result.get('processing_type') == 'error':
                            error_msg = result.get('error', 'Unknown error')
                            if '429' in error_msg or 'rate limit' in error_msg.lower():
                                status_text.warning(f"⚠️ Rate limited on batch {result.get('batch_id', 'unknown')} - using fallback methods")
                            else:
                                status_text.error(f"❌ Error in batch {result.get('batch_id', 'unknown')}: {error_msg}")
                            continue
                        
                        else:
                            # Regular batch result
                            batch_count += 1
                            batch_id = result.get('batch_id', batch_count)
                            processing_type = result.get('processing_type', 'unknown')
                            
                            # Update progress
                            status_text.info(f"🔄 Processing batch {batch_id} ({processing_type} mode)...")
                            
                            # Format and display results
                            formatted_df = format_categorization_result(result)
                            if formatted_df is not None:
                                all_results.append(formatted_df)
                                
                                with results_container:
                                    st.markdown(f"### Batch {batch_id} Results ({processing_type} mode)")
                                    st.dataframe(formatted_df, use_container_width=True)
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing stream data: {e}")
                        continue
        
        # Display final combined results
        if all_results:
            st.markdown("---")
            st.markdown("### 📊 Final Combined Results")
            combined_df = pd.concat(all_results, ignore_index=True)
            st.dataframe(combined_df, use_container_width=True)
            
            # Download button
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"categorization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Products", len(combined_df))
            with col2:
                confidence_values = combined_df['Confidence'].str.rstrip('%').astype(float)
                avg_confidence = confidence_values.dropna().mean() if not confidence_values.dropna().empty else 0.0
                st.metric("Average Confidence", f"{avg_confidence:.1f}%")
            with col3:
                unique_categories = combined_df['Category'].nunique()
                st.metric("Unique Categories", unique_categories)
            
            # Category breakdown
            st.markdown("### 📊 Category Breakdown")
            category_counts = combined_df['Category'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(category_counts)
            with col2:
                st.dataframe(category_counts.reset_index().rename(columns={'index': 'Category', 'Category': 'Count'}), use_container_width=True)
    
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The processing is taking longer than expected.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection error. Make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ Product Categorization Tool</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Health Check
        if check_fastapi_health():
            st.markdown('<div class="status-success">✅ FastAPI Server: Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ FastAPI Server: Offline</div>', unsafe_allow_html=True)
            st.error("Please start the FastAPI server before using this tool.")
            st.code("uvicorn main:app --reload")
            return
        
        st.markdown("---")
        
        # Processing Mode Selection
        st.markdown("### 🎯 Processing Mode")
        processing_mode = st.radio(
            "Choose categorization method:",
            options=[
                ("Text-based", 1),
                ("Image-based (URL scraping)", 0)
            ],
            format_func=lambda x: x[0],
            help="Text-based: Uses product descriptions and attributes\nImage-based: Scrapes images from product URLs"
        )
        
        toggle_value = processing_mode[1]
        
        st.markdown("---")
        
        # Instructions
        st.markdown("### 📋 Instructions")
        if toggle_value == 1:
            st.markdown("""
            **Text-based Processing:**
            1. Upload CSV with product data
            2. Ensure you have descriptive columns
            3. Processing: 10 items per batch
            4. Uses product descriptions for AI categorization
            5. **NEW**: Includes style classification (Modern, Japandi, Bohemian, etc.)
            6. **NEW**: Provides style tags and placement recommendations
            """)
        else:
            st.markdown("""
            **Image-based Processing:**
            1. Upload CSV with 'product' column containing URLs
            2. Images will be scraped from URLs
            3. Processing: 5 items per batch
            4. Uses AI vision for categorization
            5. **NEW**: Includes visual style analysis
            6. **NEW**: Provides detailed style tags and room placement
            
            ⚠️ **Note**: Some websites (like Wayfair) may block scraping. 
            The system will automatically fall back to text-based classification when needed.
            """)
    
    # Main content
    st.markdown('<div class="section-header">📁 File Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file containing product data for categorization"
    )
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                try:
                    df = pd.read_excel(uploaded_file)
                except ImportError:
                    st.error("❌ Excel support requires openpyxl. Please install it with: `pip install openpyxl`")
                    st.info("Alternatively, save your file as CSV and upload that instead.")
                    return
            else:
                st.error("❌ Unsupported file format. Please upload a CSV or Excel file.")
                return
            
            # Validate data
            validation_errors = validate_csv_data(df)
            
            if validation_errors:
                st.error("❌ Validation Errors:")
                for error in validation_errors:
                    st.error(f"• {error}")
                return
            
            # Display data preview
            st.markdown('<div class="section-header">👁️ Data Preview</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.markdown("**Dataset Info:**")
                st.write(f"• **Rows:** {len(df)}")
                st.write(f"• **Columns:** {len(df.columns)}")
                st.write(f"• **Mode:** {processing_mode[0]}")
                
                # Column selection for product URLs (if image mode)
                if toggle_value == 0:
                    st.markdown("**Product URL Column:**")
                    product_column = st.selectbox(
                        "Select column containing product URLs:",
                        options=df.columns.tolist(),
                        index=0 if 'product' not in df.columns else df.columns.tolist().index('product')
                    )
                else:
                    product_column = 'product'  # Default
            
            # Additional validation for image mode
            if toggle_value == 0:
                if product_column not in df.columns:
                    st.error(f"❌ Column '{product_column}' not found in the dataset")
                    return
                
                # Check for valid URLs
                sample_urls = df[product_column].dropna().head(5).tolist()
                if sample_urls:
                    st.markdown("**Sample URLs:**")
                    for i, url in enumerate(sample_urls, 1):
                        st.text(f"{i}. {url}")
            
            # Processing button
            st.markdown('<div class="section-header">🚀 Start Processing</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Start Categorization", type="primary", use_container_width=True):
                # Convert DataFrame to list of dictionaries
                data_list = df.to_dict('records')
                
                # Start processing
                st.markdown("---")
                st.markdown(f"### 🔄 Processing {len(data_list)} products in {processing_mode[0]} mode...")
                
                # Stream results
                stream_categorization_results(data_list, toggle_value, product_column)
        
        except Exception as e:
            st.error(f"❌ Error reading {file_extension.upper()} file: {str(e)}")
            st.info("Please make sure your file is properly formatted and not corrupted.")
    
    else:
        # Show example data format
        st.markdown('<div class="section-header">📖 Example Data Format</div>', unsafe_allow_html=True)
        
        if toggle_value == 1:
            # Text-based example
            example_data = {
                'product_name': ['iPhone 14', 'Nike Air Max', 'Coffee Maker'],
                'description': ['Latest smartphone with advanced camera', 'Running shoes for athletes', 'Automatic drip coffee maker'],
                'brand': ['Apple', 'Nike', 'Cuisinart'],
                'price': ['$999', '$120', '$79']
            }
        else:
            # Image-based example
            example_data = {
                'product': [
                    'https://example.com/iphone-14',
                    'https://example.com/nike-shoes',
                    'https://example.com/coffee-maker'
                ],
                'product_name': ['iPhone 14', 'Nike Air Max', 'Coffee Maker'],
                'brand': ['Apple', 'Nike', 'Cuisinart']
            }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        # Sample file downloads
        col1, col2 = st.columns(2)
        
        with col1:
            csv_sample = example_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv_sample,
                file_name=f"sample_data_{processing_mode[0].lower().replace('-', '_').replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create Excel sample
            try:
                excel_buffer = BytesIO()
                example_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Sample Excel",
                    data=excel_data,
                    file_name=f"sample_data_{processing_mode[0].lower().replace('-', '_').replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.error("Excel support requires openpyxl: `pip install openpyxl`")
                st.button("📥 Excel Not Available", disabled=True)

if __name__ == "__main__":
    main()