from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from ingestion import load_existing_index


def get_llm():
    return ChatGroq(
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        groq_api_key=config.GROQ_API_KEY,
    )


def ask_question(question, vectorstore=None):
    if vectorstore is None:
        vectorstore = load_existing_index()
    if vectorstore is None:
        return {
            "answer": "No documents have been indexed yet. Please upload documents first.",
            "sources": [],
        }

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K},
    )
    docs = retriever.invoke(question)

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[Document: {source}, Page: {page}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions based ONLY on the provided context. "
        "If the answer is not contained in the context, say 'I couldn't find the answer in the uploaded documents.' "
        "Always cite which document(s) and page(s) your answer comes from.\n\n"
        "IMPORTANT: If the user asks about a heading, section title, or topic name, "
        "look for content related to that topic even if the exact heading text is not present.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "input": question})

    sources = []
    seen = set()
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        key = f"{source_name}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source_name,
                "page": page,
                "snippet": doc.page_content[:200] + "...",
            })

    return {"answer": answer, "sources": sources}