# Week 1 notes

### Table of contents

- [1.1 Introduction to LLM and RAG](#11-introduction-to-llm-and-rag)
  - [LLM](#llm)
  - [RAG](#rag)
  - [RAG Architecture](#rag-architecture)
  - [Course Outcome](#course-outcome)
- [1.2 Preparing the Environment](#12-preparing-the-environment)
  - [Installing Libraries](#installing-the-libraries)
  - [Alternative: installing anaconda or miniconda](#alternative-installing-anaconda-or-miniconda)
- [1.3 Retrieval](#13-retrieval)
  - [Introduction to Retrieval](#introduction-to-retrieval)
  - [Preparing the Documents](#preparing-the-documents)
  - [Indexing Documents with mini-Search Library](#indexing-documents-with-mini-search-library)
  - [Retrieving Documents for a Query](#retrieving-documents-for-a-query)

## 1.1 Introduction to LLM and RAG

A lot of the content in introduction was not part of the course, and it was based off my own research of the topic. I felt the need to add a little more detail as compared to what the course actually covers. Hope it is easier for the reader to understand.

### LLM

LLM stands for Large Language Models, and Generative Pre-Trained Transformers or simply GPTs are an example of a LLM. And yes, OpenAI's flagship product the ChatGPT-3/4 is an example of a LLM. So what exactly is an LLM?

LLMs are an instance of a foundation model, i.e. models that are pre-trained on large amounts of unlabelled and self-supervised data. The foundation model learns from patterns in the data in a way that produces generalizable and adaptable output. And LLMs are instances of foundation models applied specifically to text and text-like things.

LLMs are also among the biggest models when it comes to parameter count. For example, OpenAI ChatGPT-3 model has about a 175 billion parameters. That's insane but necessary for making the product more adaptable. Parameter is a value the model can change independently as it learns, and the more parameters a model has, the more complex it can be.

So how do they work? - LLMs can be said to be made of three things:

1. Data - Large amounts of text data used as inputs into LLMs
2. Architecture - As for architecture this is a neural network, and for GPT that is a `transformer` (transformer architecture enables the model to handle sequences of data as they are designed to understand the context of each word in a sentence)
3. Training - The aforementioned transformer architecture is trained on the large amounts of data used as input, and consquentially the model learns to predict the next word in a sentence

![image](https://github.com/peterchettiar/llm-search-engine/assets/89821181/a917fa0d-4b5d-40ef-ab95-5a4a214b2b69)

The image above is a good representation of how the ChatGPT-3 operates; you input a prompt and having gone through the transformer process, it gives a text response as well. The key concept here is to understand how the transformer architecture works but that is not the main objective for today. Hence, read this [article](https://www.datacamp.com/tutorial/how-transformers-work) to understand more about the transformer architecture in detail.

> Note: A transformer is a type of artificial intelligence model that learns to understand and generate human-like text by analyzing patterns in large amounts of text data.

### RAG

RAG stands for Retrieval-Augmentation Generation which is a technique that supplements text generation with information from private or proprietary data sources. The main purpose of having a RAG model in place together with a LLM is so that the relevance of the search experience can be improved. The RAG model adds context from various data sources to complement the original knowledge base of the LLM. This method allows the responses from the LLM to be more accurate and a generally faster response.

Following is a good visual representation of the implementation and orchestration of RAG:

![image](https://github.com/peterchettiar/llm-search-engine/assets/89821181/1df01240-c487-4ef3-99f4-d16157b8175c)

### RAG Architecture

### Course Outcome

## 1.2 Preparing the Environment

### Installing the Libraries

### Alternative: Installing Anaconda or Miniconda

## 1.3 Retrieval

### Introduction to Retrieval

In the RAG framework we have two components:
- The database/knowledge base
- LLM

For the database component we will use a simple search engine, the one that was implemented during the pre-course workshop. You can either follow along the notes in the course repo [here](https://github.com/alexeygrigorev/build-your-own-search-engine?tab=readme-ov-file) **OR** watch the workshop [here](https://www.youtube.com/watch?v=nMrGK5QgPVE).

For now we will use the toy search engine for illustrative purposes but later in the course, we will replace this search engine with [elasticsearch](https://en.wikipedia.org/wiki/Elasticsearch).

Workflow of a RAG Framework (for our example use case):
1. We start off with an input query and based on relevance (what defines relevance would be discussed as we progress with the lecture), we retrieve documents, say the 5 most relevant documents from the database/knowledge base 
2. The results from the previous step would be used as input into an LLM model, say OPENAI API for example - prompt + query + context as input to LLM
3. Output from the LLM will be the response to prompt

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/1a84e57d-5ceb-4e19-8b38-71bedf0c001d)

Moving forward, for our example use case, we need to download a python script written by the course instructor to implement a minimalistic text search engine to our framework. The `minsearch` package is not pip installable and hence need to run the following code on your jupyter notebook cell in order to download the python script to your local directory (directory where your jupyter notebook was created) and use it as a package:
```
!wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
```

### Preparing the Documents

This section basically describes the steps you need to take to prepare the ingredients needed to kickstart the example use case:
1. `FAQ documents` - this is the combination of all the FAQ documents across all the various courses run by DataTalks.club, again this FAQ documents is our knowledge base in which we will be querying from (those interested in knowing how the course instructor parsed the data from google docs into a `JSON` format please look at this notebook : [parse-faq.ipynb](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/parse-faq.ipynb))
2. Next we want to download our parsed JSON file the same way we did the `minsearch.py` file - `!wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/documents.json`
3. Now we should be able to `imnport minsearch` as well as load the `JSON` file using the `JSON` library - `import json`
4. Load the `JSON` file using the following code:
```python
with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)
```
5. Lastly, we want to convert the `JSON` object into a list like the following:
```python
documents = [] 

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)
```

### Indexing Documents with mini-Search Library

Create an instance of the `Index` class, specifying the text and keyword fields.
```python
index = minsearch.Index(
    text_fields=['question','text','section']
    keyword_fields=['course']
)
```

The `keyword_field` is a filter argument. As a `SQL` query it might look like the following:
```sql
SELECT
  *
FROM documents
WHERE course = 'data-engineering-zoomcamp'
```
The `text_field` is 

### Retrieving Documents for a Query

