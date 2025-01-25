from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from fpdf import FPDF
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize VectorStore
vec = VectorStore()

def create_pdf_report(response, filename="contract_analysis_report.pdf"):
    """
    Create a PDF report based on the response.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Contract Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Analysis Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Analysis Summary:", ln=True)
    pdf.set_font("Helvetica", "", 11)

    paragraphs = response.get("answer", "").split('\n')
    for para in paragraphs:
        cleaned_para = para.strip()
        if cleaned_para:
            pdf.multi_cell(0, 7, cleaned_para)
            pdf.ln(3)

    # Detailed Analysis
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Detailed Analysis:", ln=True)
    pdf.set_font("Helvetica", "", 11)

    thought_process = response.get("thought_process", [])
    for thought in thought_process:
        cleaned_thought = thought.strip()
        if cleaned_thought:
            pdf.multi_cell(0, 7, cleaned_thought)
            pdf.ln(3)

    # Context Assessment
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Context Assessment:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    context_info = f"Sufficient context available: {response.get('enough_context', False)}"
    pdf.multi_cell(0, 7, context_info)

    # Save the PDF
    pdf.output(filename)

def analyze_contract(question: str, contract_file: str, output_pdf: str):
    """
    Analyzes the given contract file by performing similarity search and generating a PDF report.
    """
    # Load the contract data
    try:
        df = pd.read_csv(contract_file)
        logging.info(f"Loaded {len(df)} contracts from {contract_file}")
    except Exception as e:
        logging.error(f"Failed to load contract file: {e}")
        return

    # Perform similarity search
    try:
        results = vec.search(question, limit=3)
        if results.empty:
            logging.warning("No similar contracts found. Please refine your question.")
            return
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        return

    # Prepare metadata for the results
    try:
        results['metadata'] = results.apply(
            lambda x: {
                'agreement_date': x.get('agreement_date', None),
                'effective_date': x.get('effective_date', None),
                'expiration_date': x.get('expiration_date', None),
            },
            axis=1
        )
        logging.info("Metadata processed successfully.")
    except Exception as e:
        logging.error(f"Error processing metadata: {e}")
        return

    # Generate the response
    try:
        response = Synthesizer.generate_response(
            question=question,
            context=results[['content', 'metadata']].to_dict(orient="records")
        )
        logging.info("Response generated successfully.")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return

    # Generate the PDF report
    try:
        create_pdf_report(response, filename=output_pdf)
        logging.info(f"Report generated successfully: {output_pdf}")
    except Exception as e:
        logging.error(f"Error creating PDF report: {e}")
        return

if __name__ == "__main__":
    # Define inputs
    QUESTION = """[LOGO]
    AMENDMENT TO SECTION 2, PART B OF THE CO-BRANDING AGREEMENT
    This amendment to Section 2 (titled "Term"), Part B of the Co-Branding
    Agreement is made effective December 9, 1996 by and between PC Quote, Inc.
    (hereinafter referred to as "PCQ") and A.B. Watley, Inc. (hereinafter
    referred to as "ABW"), who are also the parties contracted in the
    aforementioned Co-Branding Agreement. This Amendment shall apply to said PCQ
    and ABW and all of their subsidiaries and related companies.
    """
    CONTRACT_FILE = "data/modified.csv"
    OUTPUT_PDF = "contract_analysis_report.pdf"
    
    # Analyze the contract
    analyze_contract(QUESTION, CONTRACT_FILE, OUTPUT_PDF)
