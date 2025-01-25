import streamlit as st
from similarity_search import vec, Synthesizer
import base64
import PyPDF2
import pandas as pd
from fpdf import FPDF
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def chunk_text(text, chunk_size=8000):
    """Split text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def read_pdf(file):
    """Extract text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_txt(file):
    """Read text from an uploaded TXT file."""
    return file.getvalue().decode("utf-8")

def get_pdf_download_link(pdf_path):
    """Generate a download link for the PDF file."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="contract_analysis_report.pdf">Download PDF Report</a>'
        return href

def process_large_text(text):
    """Process large text in chunks and combine results."""
    chunks = chunk_text(text)
    all_results = []
    
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        
        results = vec.search(chunk, limit=3)
        if not results.empty:
            all_results.append(results)
    
    if not all_results:
        return pd.DataFrame()

    combined_results = pd.concat(all_results, ignore_index=True)
    
    if 'content' not in combined_results.columns:
        raise ValueError("Expected 'content' column not found in search results.")

    combined_results = combined_results.drop_duplicates(subset=['content'])
    
    if {'agreement_date', 'effective_date', 'expiration_date'}.issubset(combined_results.columns):
        combined_results['metadata'] = combined_results.apply(
            lambda x: {
                'agreement_date': x['agreement_date'],
                'effective_date': x['effective_date'],
                'expiration_date': x['expiration_date']
            },
            axis=1
        )
    else:
        combined_results['metadata'] = [{} for _ in range(len(combined_results))]
    
    return combined_results

def create_pdf_report(response, filename="contract_analysis_report.pdf", no_results=False):
    """Generate PDF report from analysis or fallback message if no results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_font("Helvetica", "B", 16) 
    pdf.cell(0, 10, "Contract Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    if no_results:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "No relevant results found in the contract.", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(5)
        pdf.multi_cell(0, 7, "Please review the contract content or embeddings.")
    else:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Analysis Summary:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        # for para in response.get("answer", "").split('\n'):
        for para in response.answer.split('\n'): 
            cleaned_para = para.strip()
            
            if cleaned_para:
                pdf.multi_cell(w=0, h=7, text=cleaned_para,markdown=True) # added markdown=True 
                pdf.ln(3)
        
    pdf.output(filename)

def main():
    st.title("Contract Analysis System")
    
    uploaded_file = st.file_uploader("Upload Contract Document", type=['pdf', 'txt'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                contract_text = read_pdf(uploaded_file)
                st.success("PDF file uploaded successfully!")
            else:
                contract_text = read_txt(uploaded_file)
                st.success("TXT file uploaded successfully!")
            
            with st.expander("View Extracted Text"):
                st.text_area("Contract Text", contract_text, height=200)
            
            if st.button("Analyze Contract"):
                with st.spinner('Analyzing contract...'):
                    results = process_large_text(contract_text)
                    
                    if results.empty:
                        create_pdf_report(None, filename="contract_analysis_report.pdf", no_results=True)
                        st.warning("No relevant results found for the contract.")
                        st.markdown(get_pdf_download_link("contract_analysis_report.pdf"), unsafe_allow_html=True)
                        return
                    
                    response = Synthesizer.generate_response(
                        question=contract_text,
                        # context=results[['content', 'metadata']].to_dict(orient="records")
                        context=results[['content', 'metadata']]  
                    )
                    
                    create_pdf_report(response, "contract_analysis_report.pdf")
                    
                    st.header("Analysis Report")
                    
                    for para in response.answer.split('\n'):
                        cleaned_para = para.strip()
                        
                        if cleaned_para:
                            st.write(cleaned_para)
                    
                    st.subheader("Context Assessment")
                    # st.write(f"Sufficient context available: {response.get('enough_context', False)}")
                    st.write(f"Sufficient context available: {response.enough_context}") 
                    st.markdown(get_pdf_download_link("contract_analysis_report.pdf"), unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
