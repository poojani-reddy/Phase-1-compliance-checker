import streamlit as st
from datetime import datetime
import pandas as pd
from transformers import pipeline
import uuid
from typing import Any, Dict, List, Type
from pydantic import BaseModel

# Initialize Hugging Face pipeline for embeddings
@st.cache_resource
def load_embedding_pipeline():
    return pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

embedding_pipeline = load_embedding_pipeline()

class LLMFactory:
    def __init__(self, provider: str = "huggingface", model_name: str = "gpt2"):
        self.provider = provider
        self.model_name = model_name
        if provider == "huggingface":
            self.client = pipeline("text-generation", model=self.model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        # Combine messages into a single prompt
        prompt = self._format_messages(messages)

        # Generate response using Hugging Face
        response = self.client(
            prompt,
            max_length=kwargs.get("max_tokens", 100),
            temperature=kwargs.get("temperature", 0.7)
        )

        # Parse the response into the expected format
        result = self._parse_response(response, response_model)
        return result

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt string."""
        formatted_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
        return "\n\n".join(formatted_messages)

    def _parse_response(self, response: Any, response_model: Type[BaseModel]) -> BaseModel:
        """Parse Hugging Face response into the expected Pydantic model."""
        response_text = response[0]["generated_text"]

        # If it's a SynthesizedResponse, create the proper structure
        if response_model.__name__ == 'SynthesizedResponse':
            return response_model(
                answer=response_text,
                thought_process=["Analyzed context", "Generated response using Hugging Face model"],
                enough_context=True
            )

        # For other response types
        return response_model(**{"content": response_text})

st.title("Dataset Processor and Vector Store Integration")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (Excel file):", type=["xlsx"])

if uploaded_file:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)
    st.write("### Preview of Uploaded Dataset:")
    st.write(df.head())

    # Prepare data for insertion
    def prepare_record(row):
        """Prepare a record for processing."""
        content = (
            f"JD NAME: {row['JD NAME']}\n"
            f"Job Description: {row['JD']}\n"
            f"RESUME: {row['RESUME']}\n"
            f"Interview_Details: {row['Q AND A']}"
        )
        embedding = embedding_pipeline(content)[0][0]  # Extract embedding vector

        return pd.Series(
            {
                "id": str(uuid.uuid1()),
                "metadata": {
                    "Acceptance": row['TAG'],
                    "Category": row['JD NAME'],
                    "created_at": datetime.now().isoformat(),
                },
                "contents": content,
                "embedding": embedding,
            }
        )

    records_df = df.apply(prepare_record, axis=1)

    st.write("### Processed Records:")
    st.write(records_df.head())

    # Placeholder for Vector Store integration
    st.write("### Vector Store Integration:")
    st.write(
        "This step would involve inserting the processed records into a vector store, such as Pinecone or another database."
    )

    # Option to save processed records as a CSV for debugging
    if st.button("Download Processed Records"):
        processed_csv = records_df.to_csv(index=False)
        st.download_button(
            label="Download CSV", 
            data=processed_csv, 
            file_name="processed_records.csv", 
            mime="text/csv"
        )

else:
    st.info("Please upload an Excel file to begin.")
