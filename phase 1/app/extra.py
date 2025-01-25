# %%
from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory
import langfuse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )

class Synthesizer:
    def __init__(self):
        # Initialize Langfuse client
        try:
            self.langfuse_client = langfuse.Client(api_key="your_langfuse_api_key")
            logging.info("Langfuse client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Langfuse client: {e}")
            raise

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the knowledge base.

        Returns:
            A SynthesizedResponse containing thought process and answer.
        """
        try:
            # Convert context to JSON
            context_str = Synthesizer.dataframe_to_json(
                context, columns_to_keep=["content", "category"]
            )

            # Fetch the system prompt
            prompt_id = "your_prompt_id"  # Replace with your actual prompt ID
            system_prompt = Synthesizer.fetch_prompt_from_langfuse(prompt_id)

            # Construct the message payload
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"# User question:\n{question}"},
                {"role": "assistant", "content": f"# Retrieved information:\n{context_str}"},
            ]

            # Initialize LLM and generate a response
            llm = LLMFactory("openai")
            response = llm.create_completion(
                response_model=SynthesizedResponse,
                messages=messages,
            )
            logging.info("Successfully generated synthesized response.")
            return response

        except Exception as e:
            logging.error(f"Error generating synthesized response: {e}")
            raise

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:
        """
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        try:
            json_str = context[columns_to_keep].to_json(orient="records", indent=2)
            logging.info("Context successfully converted to JSON.")
            return json_str
        except KeyError as e:
            logging.error(f"Error in converting DataFrame to JSON: Missing column - {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in DataFrame to JSON conversion: {e}")
            raise

    @staticmethod
    def fetch_prompt_from_langfuse(prompt_id: str) -> str:
        """
        Fetch the prompt from Langfuse using the prompt ID.

        Args:
            prompt_id (str): The ID of the prompt to fetch.

        Returns:
            str: The prompt text.
        """
        try:
            langfuse_client = langfuse.Client(api_key="your_langfuse_api_key")
            prompt = langfuse_client.get_prompt(prompt_id)
            logging.info(f"Successfully fetched prompt with ID: {prompt_id}")
            return prompt.text
        except Exception as e:
            logging.error(f"Error fetching prompt from Langfuse: {e}")
            raise

# %%
if __name__ == "__main__":
    # Load contracts and analyze
    try:
        # Read the contract data from a CSV file
        contract_data = pd.read_csv("data/contracts.csv")  # Replace with your actual file path
        
        # Validate required columns
        required_columns = ["content", "category"]
        if not all(col in contract_data.columns for col in required_columns):
            raise ValueError(f"Missing one or more required columns: {required_columns}")

        # Initialize synthesizer
        synthesizer = Synthesizer()

        # Define a sample question for analysis
        question = "Provide an analysis of the uploaded contracts and key insights."

        # Generate a synthesized response
        response = synthesizer.generate_response(question, contract_data)
        
        # Output the synthesized response
        print("Thought Process:\n", "\n".join(response.thought_process))
        print("\nAnswer:\n", response.answer)
        print("\nEnough Context:\n", response.enough_context)

    except Exception as e:
        logging.error(f"Error in contract analysis: {e}")
