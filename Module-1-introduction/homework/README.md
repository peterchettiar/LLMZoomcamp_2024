# Homework 1: Introduction for LLM Zoomcamp 2024

## Question 1. Running Elastic

The following command was provided for running a docker container for `elasticsearch`.

```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

However, `elasticsearch` kept dying on me - `ERROR: Elasticsearch exited unexpectedly.`. So I used the following command instead:

```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    --ulimit nofile=65536:65536 \
    --ulimit memlock=-1:-1 \
    --memory=4g \
    --cpus=2 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

Now that `elasticsearch` is running, we can run the following code, to print out the `version.build_hash` value:
```python
# initialising elasticsearch
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200')

print(es_client.info()['version']['build_hash'])
```

## Question 2. Indexing the Data

So now that we have `elasticsearch` up and running, we want to index the knowledge base for easy querying. The following code was provided to get the FAQ data:

```python
import requests 

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

But before we index FAQ `documents`, we need to create and `indices` object and to do so we need an `index_name` as well as `index_settings`. This is more like creating the structure of the index before add data to elastic. So, the following code snippet does both:

```python
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

index_name = "course-questions"

es_client.indices.create(index=index_name, body=index_settings)
```
Now we want to add the data into our elastic `index`. So we use the `index` method to do so as follows iteratively:

```python
from tqdm import tqdm

for doc in tqdm(documents):
    es_client.index(index=index_name,
                    document=doc)
```

> tqdm is a progress bar so as to monitor how long it takes to perform the indexing.

## Question 3. Searching

Our query is as follows.

```python
query = "How do I execute a command in a running docker container?"
```

Next we want to define our `search_query`.
```python
search_query = {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
            }
        }
    }
}
```

Now we want to genearate the response.
```python
response = es_client.search(index=index_name, body=search_query)
```

Lastly, we get the list of scores based on `best_fields`.
```python
res_score = []

for score in response['hits']['hits']:
    res_score.append(score['_score'])
```

This should give us the highest ranking result of `84.05`.

## Question 4. Filtering

So now we repeat the steps as before but this time with filtering - basically adding `filter` to your `search_query`.

```python
search_query = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"], # question are three times more important
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": 'machine-learning-zoomcamp'
                }
            }
        }
    }
}

response = es_client.search(index=index_name, body=search_query)
```
Run a similar loop as before but instead of `_score`, its now `_source`.

```python
result_docs = []

for hit in response['hits']['hits']:
    result_docs.append(hit['_source'])
```

Lastly, you need to print the `question` from your `results_docs` to get the solution to your question.
```python
result_docs[2]['question']
```

The answer : `How do I copy files from a different folder into docker container’s working directory?`

## Question 5. Prompt Length

Three things we need here:

1. Prompt Template
```python
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()
```

2. Context that is based on the context template provided.
```python

context = ""

for doc in result_docs:
    context += f"Q: {doc['question']}\nA: {doc['text']}\n\n".strip()
```

3. The prompt
```python
prompt = prompt_template.format(question="How do I execute a command in a running docker container?",
                                 context=context).strip()
```

And when you `len(prompt)` - you get `1458`. The closest answer would be `1462` in this case.

## Question 6. Tokens

We need to first `pip install tiktoken`. Next we run the following to get the number of tokens.
```python
from tiktoken import encoding_for_model
encoding = encoding_for_model("gpt-4o")
len(encoding.encode(prompt))
```

This gave me `321` but the closest was `322`, hence that would be the answer.