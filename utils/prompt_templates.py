"""
Custom prompt templates for the LLM.
"""

from langchain_core.prompts import PromptTemplate


QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not contained in the context, say "I couldn't find the answer in the uploaded documents."

Always cite which document(s) and page(s) your answer comes from.

Context:
{context}

Question: {question}

Answer (with citations):"""


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_PROMPT_TEMPLATE,
)
