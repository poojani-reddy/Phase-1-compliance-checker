import logging
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime
import os

import pandas as pd
from config.settings import get_settings
from timescale_vector import client
import google.generativeai as genai

class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings and Timescale Vector client."""
        self.settings = get_settings()
        import google.generativeai as genai
        genai.configure(api_key="your_google_api_key")
        
  # Configure Gemini for embedding generation
        self.vector_settings = self.settings.vector_store
        
        self.vec_client = client.Sync(
            self.settings.database.service_url,
            self.vector_settings.table_name,
            self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval,
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using Gemini.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        try:
            start_time = time.time()
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
            )
            embedding = response.get("embedding", [])
            if not embedding:
                raise ValueError("Empty embedding returned.")
            elapsed_time = time.time() - start_time
            logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
            return embedding
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return []

    def create_tables(self) -> None:
        """Create the necessary tables in the database."""
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to speed up similarity search if it doesn't exist."""
        try:
            self.vec_client.create_embedding_index(client.DiskAnnIndex())
            logging.info(f"Created DiskANN index for {self.vector_settings.table_name}")
        except Exception as e:
            if "already exists" in str(e):
                logging.info(f"Index already exists for {self.vector_settings.table_name}")
            else:
                logging.error(f"Error creating index: {e}")
                raise

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database."""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        try:
            records = df.to_records(index=False)
            self.vec_client.upsert(list(records))
            logging.info(f"Inserted {len(df)} records into {self.vector_settings.table_name}")
        except Exception as e:
            logging.error(f"Failed to upsert records: {e}")

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        Args:
            query_text: The input text to search for similar embeddings.
            limit: The maximum number of results to return.
            metadata_filter: Filter results based on metadata.
            predicates: Predicates to apply to the query.
            time_range: Time range for filtering results.
            return_dataframe: Whether to return the results as a pandas DataFrame.

        Returns:
            Search results as a list of tuples or a pandas DataFrame.
        """
        query_embedding = self.get_embedding(query_text)
        # print("query_embedding",len(query_embedding))
        if not query_embedding:
            logging.error("Embedding generation failed for the query text.")
            return pd.DataFrame() if return_dataframe else []

        start_time = time.time()

        search_args = {"limit": limit}

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        try:
            
            results = self.vec_client.search(query_embedding, **search_args)
            elapsed_time = time.time() - start_time
            logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")
            # print("Search results:::",results)
            if return_dataframe:
                return self._create_dataframe_from_results(results)
            else:
                return results
        except Exception as e:
            logging.error(f"Error during vector search: {e}")
            return pd.DataFrame() if return_dataframe else []

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        try:
            df = pd.DataFrame(
                results, columns=["id", "metadata", "content", "embedding", "distance"]
            )
            df = pd.concat(
                [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
            )
            df["id"] = df["id"].astype(str)
            return df
        except Exception as e:
            logging.error(f"Error creating DataFrame from results: {e}")
            return pd.DataFrame()

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database."""
        try:
            if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
                raise ValueError(
                    "Provide exactly one of: ids, metadata_filter, or delete_all"
                )

            if delete_all:
                self.vec_client.delete_all()
                logging.info(f"Deleted all records from {self.vector_settings.table_name}")
            elif ids:
                self.vec_client.delete_by_ids(ids)
                logging.info(f"Deleted {len(ids)} records from {self.vector_settings.table_name}")
            elif metadata_filter:
                self.vec_client.delete_by_metadata(metadata_filter)
                logging.info(f"Deleted records matching metadata filter from {self.vector_settings.table_name}")
        except Exception as e:
            logging.error(f"Error deleting records: {e}")

            # import google.generativeai as genai

# Configure the API key
    # genai.configure(api_key="your_google_api_key")

    def analyze_contract(contract_text):
        prompt = f"Analyze the following contract and provide a summary of key terms, obligations, and risks:\n{contract_text}"
        response = genai.generate_text(prompt=prompt)
        return response


    def analyze_contract(self, contract_id: str) -> dict:
        """
        Perform analysis on a specific contract by ID.

        Args:
            contract_id: The unique ID of the contract to analyze.

        Returns:
            A dictionary containing contract insights.
        """
        try:
            # Retrieve the contract by ID
            result = self.vec_client.get_by_id(contract_id)
            if not result:
                logging.warning(f"No contract found with ID: {contract_id}")
                return {}

            content = result.get("content", "")
            metadata = result.get("metadata", {})

            insights = {
                "length": len(content),
                "summary": genai.summarize_content(content=content, model="models/text-summary-001").get("summary", ""),
                "metadata": metadata,
                "top_keywords": self.extract_keywords(content),
                "last_updated": metadata.get("last_updated", "Unknown"),
            }

            logging.info(f"Contract analysis completed for ID: {contract_id}")
            return insights
        except Exception as e:
            logging.error(f"Error analyzing contract {contract_id}: {e}")
            return {}

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract key phrases from the provided text.

        Args:
            text: The input text to analyze.
            top_n: Number of top keywords to extract.

        Returns:
            A list of extracted keywords or phrases.
        """
        try:
            words = text.split()
            keywords = sorted(set(words), key=words.count, reverse=True)[:top_n]
            return keywords
        except Exception as e:
            logging.error(f"Error extracting keywords: {e}")
            return []
