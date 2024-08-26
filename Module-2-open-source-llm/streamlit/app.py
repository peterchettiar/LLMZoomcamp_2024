"""Importing the necessary libraries"""

import json
import streamlit as st
from openai import OpenAI
from elasticsearch import Elasticsearch


def initialize_openai() -> OpenAI:
    """Initializing the OpenAI client object with Ollama servers as the local API endpoint"""
    client = OpenAI(base_url="http://ollama:11434/v1/", api_key="ollama")

    return client


def initialize_elasticsearch(index_name: str) -> Elasticsearch:
    """Connects to the Elasticsearch client and defines a new index."""
    es_client = Elasticsearch("http://elasticsearch:9200")

    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
            }
        },
    }

    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    es_client.indices.create(index=index_name, body=index_settings)

    return es_client


def convert_json_to_list(json_file: str) -> list:
    """Converting our JSON file into a list"""

    with open(json_file, "rt", encoding="utf-8") as f_in:
        docs_raw = json.load(f_in)

    documents = []

    for course_dict in docs_raw:
        for doc in course_dict["documents"]:
            doc["course"] = course_dict["course"]
            documents.append(doc)

    return documents


def elastic_search(query: str, es_client: Elasticsearch, index_name: str) -> list:
    """Function to query the ElasticSearch index - as part of the RAG framework"""

    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {"term": {"course": "data-engineering-zoomcamp"}},
            }
        },
    }

    response = es_client.search(index=index_name, body=search_query)

    result_docs = []
    for hit in response["hits"]["hits"]:
        result_docs.append(hit["_source"])

    return result_docs


def build_prompt(query: str, search_results: list) -> str:
    """Function to build prompt with context and query"""

    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database. 
    Use only the facts from the CONTEXT when answering the QUESTION.
    If the CONTEXT doesn't contain the answer, output NONE

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip()  # so that there are no extra line breaks

    context = ""

    for doc in search_results:
        context += (
            f"section: {doc['section']}\n"
            f"question: {doc['question']}\n"
            f"answer: {doc['text']}\n\n"
        )

    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt


def llm(prompt: str, client: OpenAI) -> str:
    """Function that calls ollama models using the openai framework"""
    response = client.chat.completions.create(
        model="gemma2:2b", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def main():
    """main function for deploying our LLM chatbot to the streamlit web app"""

    openai_client = initialize_openai()

    es_client = initialize_elasticsearch("course-faqs")

    # loading and indexing our knowledge base
    documents = convert_json_to_list("documents.json")

    # indexing documents into ElasticSearch
    for doc in documents:
        es_client.index(index="course-faqs", document=doc)

    st.title("LLM Zoomcamp FAQs")

    user_input = st.text_input("Enter your input:")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            search_results = elastic_search(
                query=user_input, es_client=es_client, index_name="course-faqs"
            )

            prompt = build_prompt(query=user_input, search_results=search_results)

            output = llm(prompt=prompt, client=openai_client)

            st.success("Completed!")
            st.write(output)


if __name__ == "__main__":
    main()
