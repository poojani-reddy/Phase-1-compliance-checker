import logging
from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time
from fpdf import FPDF

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def truncate_text(text: str, max_bytes: int = 9900) -> str:
    """Truncate text to ensure it doesn't exceed the byte limit."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text

    # Binary search to find the right cutoff point
    left, right = 0, len(text)
    while left < right:
        mid = (left + right + 1) // 2
        if len(text[:mid].encode('utf-8')) <= max_bytes:
            left = mid
        else:
            right = mid - 1

    return text[:left]

def generate_analysis_report(analysis_results, filename="analysis_report.pdf"):
    """Generate a PDF report summarizing contract analysis results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, "Contract Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Analysis Summary
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(0, 10, "Analysis Summary", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.ln(5)

    for contract_id, details in analysis_results.items():
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.cell(0, 10, f"Contract ID: {contract_id}", ln=True)
        pdf.set_font("Helvetica", size=11)
        for key, value in details.items():
            pdf.cell(0, 10, f"{key}: {value}", ln=True)
        pdf.ln(5)

    pdf.output(filename)
    logging.info(f"Analysis report saved to {filename}")

def analyze_and_insert(csv_path: str, vec_store: VectorStore, date_columns: list):
    """Process the CSV file and insert data into the vector store, along with contract analysis."""
    try:
        vec_store.delete(delete_all=True)
        logging.info("Deleted all existing embeddings")
    except Exception as e:
        logging.error(f"Error deleting embeddings: {e}")
        return
    # return
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return

    # Validate required columns
    required_columns = date_columns + ['contract']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return

    # Convert date columns to datetime
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    processed_data = []
    for idx, row in df.iterrows():
        try:
            metadata = {
                "agreement_date": row["Agreement Date"].isoformat() if pd.notna(row["Agreement Date"]) else None,
                "effective_date": row["Effective Date"].isoformat() if pd.notna(row["Effective Date"]) else None,
                "expiration_date": row["Expiration Date"].isoformat() if pd.notna(row["Expiration Date"]) else None,
            }

            content_parts = [
                f"{col}: {str(row[col]).strip()}"
                for col in df.columns if col != "contract" and col not in date_columns and pd.notna(row[col])
            ]

            if pd.notna(row["contract"]):
                content_parts.append(f"Contract Text: {str(row['contract']).strip()}")

            full_content = "\n".join(content_parts)
            truncated_content = truncate_text(full_content)

            embedding = vec_store.get_embedding(truncated_content)

            record = {
                "id": uuid_from_time(datetime.now()),
                "metadata": metadata,
                "contents": truncated_content,
                "embedding": embedding
            }
            processed_data.append(record)

            if (idx + 1) % 10 == 0:
                logging.info(f"Processed {idx + 1} contracts...")

        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")
            continue

    if processed_data:
        try:
            insert_df = pd.DataFrame(processed_data)
            print("Inserting into vector store::",insert_df.shape)
            vec_store.create_tables() 
            vec_store.upsert(insert_df)
            vec_store.create_index()
            logging.info(f"Successfully inserted {len(processed_data)} contract entries")
        except Exception as e:
            logging.error(f"Error during vector store insertion: {e}")
    else:
        logging.warning("No valid data to insert into the vector store.")
    #commented below 
    # try:
    #     analysis_results = vec_store.analyze_contracts(df)
    #     logging.info("Contract analysis completed successfully.")
    #     generate_analysis_report(analysis_results)
    # except Exception as e:
    #     logging.error(f"Error during contract analysis: {e}")

if __name__ == "__main__":
    CSV_PATH = r"C:\pgv\data\modified.csv"
    # CSV_PATH = "data/contracts_dataset.csv"
    DATE_COLUMNS = ['Agreement Date', 'Effective Date', 'Expiration Date']

    vec = VectorStore()
    analyze_and_insert(CSV_PATH, vec, DATE_COLUMNS)
