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
- [1.4 Generattion with OpenAI](#14-generation-with-openai)
  - [Introduction](#introduction)
  - [Response Analysis](#respones-analysis)
  - [Building a Prompt Template and Context](#building-a-prommpt-template-and-context)
  - [Getting the Answer](#getting-the-answer)
- [1.5 Cleaned RAG Flow](#15-cleaned-rag-flow)
  - [Introduction](#introduction-1)
  - [Search](#search)
  - [Building the Prompt](#building-the-prompt)
  - [LLM](#llm-1)
  - [The RAG Flow](#the-rag-flow)
  - [The RAG Flow Function](#the-rag-flow-function)

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

The RAG architecture typically consists of two main components:
1. **Retriever**: This component retrieves relevant documents or information based on the input query.
2. **Generator**: This component generates a response using the retrieved documents as context.

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/1479fd6a-ff51-4f9a-aba4-96a6e1563a53)

### Course Outcome

By the end of this course, you will:
- Understand the basics of LLM and RAG.
- Be able to implement a simple RAG pipeline.
- Gain hands-on experience with indexing and searching documents.
- Learn to integrate OpenAI for answer generation.
- Understand how to use Elasticsearch for improved search capabilities.

## 1.2 Preparing the Environment

In this chapter, we will set up the environment required for the course. This includes installing necessary libraries and tools.

### Installing the Libraries

To get started, we need to install the following libraries:

```bash
pip install tqdm notebook==7.1.2 openai elasticsearch pandas scikit-learn
```

## Setting Up Jupyter Notebook

Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text. It is widely used in data science and machine learning.

To start Jupyter Notebook, run the following command:

```bash
jupyter notebook
```
### Alternative: Installing Anaconda or Miniconda

Anaconda and Miniconda are popular distributions of Python and R for scientific computing and data science. They come with a package manager called `conda` that makes it easy to install and manage packages.

To install Anaconda, follow the instructions on the [official website](https://www.anaconda.com/products/distribution).

To install Miniconda, follow the instructions on the [official website](https://docs.conda.io/en/latest/miniconda.html).

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
The `text_field` is the field in which the search is performed.

### Retrieving Documents for a Query

Next, we want to retreive the documents from our knowledge base for a given query. Say our query is as follows:
```python
q = 'the course has already started, can I still enroll?'
```
Since we already initialised the `index` object, we can now fit it with the provided documents.
```python
# simply fitting the index with the provided document

index.fit(documents)
```

Then all we have to do now is search through the documents to return the top 5 documents that are relevant to our query.
```python
# now to perform the search on the index with the given query, filters and boost parameters

boost = {'question': 3.0}  # adding 3 times more importance to 'question field'

results = index.search(
    query = q,
    filter_dict= {'course': 'data-engineering-zoomcamp'},
    boost_dict= boost,
    num_results= 5  # number of outputs - i.e. docs retreived
)
```

You would notice that there are additional arguments in our `search` method. 

`filter_dict`: Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.

`boost_dict`: Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.given the `question` field 3 times the importance.

`num_results`: The number of top results to return. Defaults to 10.

## 1.4 Generation with OpenAI

### Introduction

So what we've done so far is index our documents with a search engine, and how to perform a search with a user query.

And the responses we got were documents retrieved from our knowledge base. So what we're suppose to do now is to put these relevant documents, that may or may not contain the answer for our query, into a LLM as a context to the question that the user had asked.

We will now need to form a prompt and send it to `OpenAI GPT4-o` or your preferred LLM, with our mini toy-search engine as our database.

But before we proceed, I wanted to quickly share on the importance of `indexing`. Please note that I had generated the following response from `claude.ai`

In the context of Large Language Models (LLMs), indexing documents refers to the process of organizing and structuring a large collection of text data in a way that allows for efficient retrieval and utilization by the model. This is particularly important for enhancing an LLM's ability to access and use specific information quickly. Here's a breakdown of what this involves:

1. Document preprocessing:

  - Cleaning and formatting text
  - Removing irrelevant information
  - Tokenization (breaking text into words or subwords)

2. Creating searchable structures:

  - Building inverted indexes (mapping words to document locations)
  - Generating embeddings (vector representations of text)

3. Metadata tagging:

  - Adding descriptive information to documents
  - Categorizing content

4. Efficient storage:

- Organizing data for quick access
- Compressing information where possible

5. Retrieval mechanisms:

- Implementing search algorithms
- Enabling semantic search capabilities

6. Integration with LLM:

- Designing interfaces between indexed documents and the model
- Implementing retrieval-augmented generation techniques

The goal is to enable the LLM to quickly find and incorporate relevant information from a large corpus when generating responses, improving accuracy and reducing hallucinations.
This process is crucial for applications like question-answering systems, chatbots with access to specific knowledge bases, and other AI systems that need to combine general language understanding with specific, retrievable information.

### Respones Analysis

For this section, we are just going to make a comparison of the responses that we get from `OPENAI` with and without any context provided.

So if we want to generate responses just based on our question (`The course has already started, can I still enroll?`), we run the following code:

```python
# response from OpenAI without providing context

response = client.chat.completions.create(
    model='gpt-4o',
    messages=[{"role":"user", "content":q}]
)
```

The responses can be printed out using `pprint(response.choices[0].message.content)` and it should look like the following:

```txt
('Whether you can still enroll in a course that has already started typically '
 'depends on the policies of the institution or organization offering the '
 'course. Here are a few steps you can take to find out:\n'
 '\n'
 '1. **Check the Course Description or Website:** Look for any information '
 'regarding late enrollment.\n'
 '  \n'
 '2. **Contact the Instructor or Course Coordinator:** They may allow late '
 'enrollment under certain circumstances or provide details on how to catch '
 'up.\n'
 '\n'
 '3. **Reach Out to the Admissions Office:** They can provide information '
 "about the institution's policies on late enrollment and may help facilitate "
 'your enrollment.\n'
 '\n'
 '4. **Consider the Impact:** Evaluate how much content you have missed and '
 'whether you can realistically catch up.\n'
 '\n'
 '5. **Consult Academic Advisors:** They can offer guidance and might help '
 'with exceptions to policies.\n'
 '\n'
 'If the course allows for asynchronous learning (where you can access '
 'materials and complete work at your own pace), catching up might be easier. '
 'However, for courses with significant interactive or time-bound components '
 '(like group projects or frequent assessments), enrolling late could be more '
 'challenging.')
```

You would immediately realise that it is rather generic and random. So in order to get a response that is more in line with our knowledge base, we need to build a prompt that provides the relevant documents from our knowledge base as context. This is exactly what we would be doing in the next sections. 

### Building a Prommpt Template and Context

Feel free to play around and define your own prompt template, but a good starting point is as follows:

```python
# building a prompt template

prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database. 
Use only the facts from the CONTEXT when answering the QUESTION.
If the CONTEXT doesn't contain the answer, output NONE

QUESTION: {question}

CONTEXT: 
{context}
""".strip() # so that there are no extra line breaks
```

Next, we can build the context based on the `results` variable from before - top 5 most relevant documents retrieved from our knowledge base based on weighted cosine similarity with our query.

```python
# building the context

context = ""

for doc in results:
    context += f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
```

### Getting the Answer

Now that we have our `prompt_template` as well as our `context`, we can proceed to define and viewn the `prompt`.

```python
prompt = prompt_template.format(question=q, context=context).strip()

print(prompt)
```

This `prompt` would be provided as context into our `LLM`. Let's now generate the responses with context and analyse the stark difference when using the `RAG` framework.

```python
# getting the answers - responsese with context

response_with_context = client.chat.completions.create(
    model='gpt-4o',
    messages=[{"role":"user", "content":prompt}]
)

response_with_context.choices[0].message.content
```

The response you get will look like the following:
```text
Yes, even if you don't register, you're still eligible to submit the homeworks. Be aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.
```

Notice the main difference that its not so random and generic, your responses are very much in-line with the knowledge base. This is the power of `RAG`!

## 1.5 Cleaned RAG Flow

### Introduction

The objective of the subsquent sections is to clean the code that we have written so far by making it more modular as well as replacing the components used (e.g. our `minsearch` package) as we progress further into the course.

To summarise the RAG framework:

send a query to our search engine (this is the `index` object we initialised based on the `minsearch` library) --> search engine returns most relevant documents from the knowledge base --> build a prompt with the query and results from the search engine as context --> send prompt to LLM --> Get Answer.

These are the steps we have done so far based on the previous lecture videos. The code written so far was written cell by cell to enable clear understanding of the steps involved. Hence, the need for modularisation.

### Search

Assuming you had already initialised your `index` object, same as we did before, we can just proceed with writing a function to perform search. Keep in mind that the following function is by no means a dynamic one. There are a lot of 'Hard' coding done here. In the best case scenario, function should parameterized as far as possible.

```python
def search(query):

    boost = {'question': 3.0}  # adding 3 times more importance to 'question field'

    results = index.search(
        query = query,
        filter_dict= {'course': 'data-engineering-zoomcamp'},
        boost_dict= boost,
        num_results= 5  # number of outputs - i.e. docs retreived
    )

    return results
```
 we can test our function - `search('how do I run Kafka?')`

### Building the Prompt

Modularizing what we had done previously into a function:
```python
# funtion to build the prompt

def build_prompt(query, search_results):

    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database. 
    Use only the facts from the CONTEXT when answering the QUESTION.
    If the CONTEXT doesn't contain the answer, output NONE

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip() # so that there are no extra line breaks

    context = ""

    for doc in search_results:
        context += f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context)

    return prompt
```

This should return you the prompt in which is used as input into the LLM.

### LLM

Similary, let's write a function for the LLM.

```python
# now function for the LLM

def llm(prompt):

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role':'user',
                   'content': prompt}]
    )

    return response.choices[0].message.content
```

### The RAG Flow

So now that we have all our functions to generate the answer for our query, the RAG flow should look something like the following:
```python
# so now bringing them all together - the RAG flow
query = 'The course already started. Can I still enroll?''
res = search(query=query)
prompt = build_prompt(query=query, search_results=res)
answer = llm(prompt=prompt)
```

### The RAG Flow Function

Here we are simply going to convert our `RAG Flow` into a function.

```python
# we can put them into a RAG flow function

def rag(query):

    search_results = search(query=query)
    prompt = build_prompt(query=query, search_results=res)
    answer = llm(prompt=prompt)

    return answer
```

This makes the iteration process as well as the understanding of the `RAG Workflow` a lot simpler. If we need to change the search engine from `minsearch` to `elasticsearch`, then we just need to amend this function (change `search` in `search_results`). Similarly, if we want to change our model instead, we just need to amend `llm`.