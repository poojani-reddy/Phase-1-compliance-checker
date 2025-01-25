from typing import List
import pandas as pd
from pydantic import BaseModel, Field
# from transformers import pipeline
import google.generativeai as genai
        

class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class Synthesizer:
    # SYSTEM_PROMPT = """
    # # Role and Purpose
    # You are an AI assistant designed to evaluate resumes against specific parameters such as job descriptions, required skills, and experience. 
    # Your task is to generate a detailed, structured report indicating whether the resume is suitable for the given job description and provide a 
    # score and reasoning.

    # # Guidelines:
    # 1. Assess the resume based on the job description, required skills, experience, and other provided parameters.
    # 2. Clearly state whether the resume meets the criteria and why.
    # 3. Provide a suitability score between 0 and 100, where:
    #     - 0-40: Poor fit
    #     - 41-70: Moderate fit
    #     - 71-100: Excellent fit
    # 4. Highlight key strengths of the resume and areas for improvement.
    # 5. Maintain a professional and constructive tone, offering actionable feedback to the candidate.
    # 6. If there is insufficient information to fully evaluate the resume, state this explicitly and suggest what is missing.
    # 7. Adhere to the following structured format for the response:
    
    # **Suitability Report:**
    # - Suitability Score: [Score] out of 100
    # - Verdict: [Good Fit / Moderate Fit / Poor Fit]

    # **Strengths:**
    # - [List key strengths of the resume in bullet points]

    # **Areas for Improvement:**
    # - [List specific weaknesses or missing elements in bullet points]

    # **Reasoning:**
    # - Provide a detailed explanation for the score, referencing specific aspects of the resume and job description.

    # **Additional Information (if needed):**
    # - Mention any missing details or additional context required for a complete evaluation.
    # """
    #  new prompt
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an AI assistant designed to evaluate contracts. 
    Your task is to generate a detailed, structured report with  score and reasoning.

    # Guidelines:
    1. Assess the contract text based on the payment terms, delivery timelines, confidentiality obligations, termination clauses, and liability limitations.
    2. Provide a compliance score between 0 and 100, where:
        - 0-40: Poor fit
        - 41-70: Moderate fit
        - 71-100: Excellent fit
    3. Highlight key strengths of the contract and areas for improvement.
    5. Maintain a professional and constructive tone, offering actionable feedback to the candidate.
    6. If there is insufficient information to fully evaluate the contract, state this explicitly and suggest what is missing.
    7. Adhere to the following structured format for the response:
    
    **Compliance Report:**
    - Suitability Score: [Score] out of 100
    - Verdict: [Good Fit / Moderate Fit / Poor Fit]

    **Strengths:**
    - [List key strengths of the contract in bullet points]

    **Areas for Improvement:**
    - [List specific weaknesses or missing elements in bullet points]

    **Reasoning:**
    - Provide a detailed explanation for the score, referencing specific aspects of the contract.

    **Additional Information (if needed):**
    - Mention any missing details or additional context required for a complete evaluation.
    """

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the knowledge base.

        Returns:
            A SynthesizedResponse containing thought process and answer.
        """
        
        context_str = Synthesizer.dataframe_to_json(
            context, 
            columns_to_keep=["content", "metadata"] 
            # columns_to_keep=["content", "Category", "Acceptance"]
        )
        
        # Combine the prompt, question, and context into a single input
        input_text = f"{Synthesizer.SYSTEM_PROMPT}\n\n# User question:\n{question}\n\n# Retrieved information:\n{context_str}"
        # using genai instead of hugging face
        genai.configure(api_key="your_google_api_key")
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(input_text)
        # print(response.text)
        synthesized_answer = response.text
        # # Load Hugging Face summarization pipeline
        # summarization_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")
        
        # # Generate a response
        # response = summarization_pipeline(input_text, max_length=512, min_length=100, do_sample=False)
        
        # synthesized_answer = response[0]["generated_text"]
        
        # Construct the SynthesizedResponse
        return SynthesizedResponse(
            thought_process=["Processed user question", "Synthesized response based on context"],
            answer=synthesized_answer,
            enough_context=bool(context_str.strip())
        )

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:
        """
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", indent=2)
        # context.to_csv("context.csv",index=False)
        # return context.to_json(orient="records", indent=2)
