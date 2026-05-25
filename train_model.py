import streamlit as st
import pandas as pd
from datetime import datetime
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape

st.set_page_config(page_title="Prescription Entry", page_icon="📋", layout="wide")

st.title("📋 Prescription Data Entry System")
st.markdown("Upload prescription image and type the text manually")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload prescription image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)

with col2:
    st.subheader("📝 Manual Text Entry")
    
    prescription_text = st.text_area(
        "Type the prescription text exactly as you see it:",
        height=400,
        placeholder="Example:\nPatient Name: Mr. Masud Iqbal\nAge: 45\nMedicines:\n- Tab. Cardizam 30: 1-1-1\n- Tab. Odrel 75: 0-1-0\n- Cap. Dexitend 30: 1-0-1"
    )
    
    if st.button("Generate Report", type="primary") and prescription_text:
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': uploaded_file.name if uploaded_file else "manual",
            'extracted_text': prescription_text,
            'characters': len(prescription_text),
            'words': len(prescription_text.split())
        }
        
        df = pd.DataFrame([record])
        csv_data = df.to_csv(index=False).encode('utf-8')
        csv_filename = f"prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pdf_filename = csv_filename.replace('.csv', '.pdf')
        
        os.makedirs('output', exist_ok=True)
        df.to_csv(f"output/{csv_filename}", index=False)

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Prescription Report", styles['Title']),
            Spacer(1, 12),
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
            Spacer(1, 12),
        ]

        for line in prescription_text.splitlines():
            if line.strip():
                story.append(Paragraph(escape(line), styles['Normal']))
                story.append(Spacer(1, 6))

        doc.build(story)
        pdf_data = pdf_buffer.getvalue()

        with open(f"output/{pdf_filename}", "wb") as f:
            f.write(pdf_data)
        
        st.success("✅ Report generated successfully!")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("📊 Download CSV", csv_data, csv_filename, "text/csv")
        with col_d2:
            st.download_button("📄 Download PDF", pdf_data, pdf_filename, "application/pdf")
