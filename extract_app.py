import streamlit as st
import pandas as pd
import os
from datetime import datetime
import easyocr
import cv2
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Handwritten Prescription Extractor",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Handwritten Prescription Text Extractor")
st.markdown("Upload a handwritten prescription image - AI will extract the text")

# Load EasyOCR (cached for performance)
@st.cache_resource
def load_ocr():
    with st.spinner("Loading OCR engine (first time may take 20-30 seconds)..."):
        return easyocr.Reader(['en'])

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    **How it works:**
    1. 📷 Upload a handwritten prescription image
    2. 🔍 AI extracts text using EasyOCR
    3. 📊 Download as CSV report
    
    **Note:** First time loading may take 20-30 seconds
    """)
    
    st.markdown("---")
    
    # Check OCR status
    try:
        load_ocr()
        st.success("✅ EasyOCR is ready")
    except Exception:
        st.error("❌ EasyOCR not loaded")
    
    st.markdown("---")
    st.caption("Powered by EasyOCR | Handwriting Recognition")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload Prescription")
    
    uploaded_file = st.file_uploader(
        "Choose a handwritten prescription image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True, caption="Uploaded Prescription")
        st.caption(f"File: {uploaded_file.name} | Size: {uploaded_file.size/1024:.1f} KB")

with col2:
    st.subheader("📄 Extracted Text")
    
    if uploaded_file:
        if st.button("🔍 Extract Text", type="primary", use_container_width=True):
            with st.spinner("Extracting text from handwritten prescription... (15-25 seconds)"):
                try:
                    # Load OCR
                    reader = load_ocr()
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Read image
                    image_bytes = uploaded_file.read()
                    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Convert to RGB (EasyOCR expects RGB)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Extract text using EasyOCR
                    result = reader.readtext(img_rgb, detail=1, paragraph=False)
                    extracted_lines = [item[1].strip() for item in result if len(item) > 1 and item[1].strip()]
                    extracted_text = '\n'.join(extracted_lines)
                    
                    if extracted_text and len(extracted_text.strip()) > 10:
                        st.success("✅ Text extracted successfully!")
                        
                        # Show extracted text
                        with st.expander("View Extracted Text", expanded=True):
                            st.text(extracted_text)
                        
                        # Statistics
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Characters", len(extracted_text))
                        with col_s2:
                            st.metric("Words", len(extracted_text.split()))
                        with col_s3:
                            st.metric("Detected Segments", len(extracted_lines))
                        
                        # Prepare CSV
                        record = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'filename': uploaded_file.name,
                            'extracted_text': extracted_text,
                            'characters': len(extracted_text),
                            'words': len(extracted_text.split())
                        }
                        
                        df = pd.DataFrame([record])
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        csv_filename = f"prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        # Save to output
                        os.makedirs('output/csv_reports', exist_ok=True)
                        df.to_csv(f"output/csv_reports/{csv_filename}", index=False)
                        
                        # Download button
                        st.markdown("---")
                        st.subheader("📥 Download Report")
                        
                        st.download_button(
                            label="📊 Download CSV Report",
                            data=csv_data,
                            file_name=csv_filename,
                            mime="text/csv",
                            use_container_width=True,
                            type="primary"
                        )
                        
                        st.caption(f"💾 Saved to: output/csv_reports/{csv_filename}")
                        
                        # Show sample of what was found
                        st.info(f"📝 Found {len(extracted_text.split())} words")
                        
                    else:
                        st.error("❌ No text could be extracted.")
                        st.info("""
                        **Tips for better results:**
                        - Ensure the image is clear and well-lit
                        - Handwriting should be reasonably readable
                        - Try taking a photo with better contrast
                        - Use darker ink on white paper
                        """)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("""
                    **Troubleshooting:**
                    - Make sure EasyOCR is installed: `pip install easyocr`
                    - First run downloads model files (takes time)
                    - Check your internet connection
                    """)
    else:
        st.info("👈 Upload a handwritten prescription image to begin")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Handwritten Prescription Extractor | Powered by EasyOCR"
    "</div>",
    unsafe_allow_html=True
)
