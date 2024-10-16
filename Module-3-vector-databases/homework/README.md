# Homework 3: Vector Search

## Question 1. Getting the embeddings model

1. Initialise the `SentenceTransformer` object with `multi-qa-distilbert-cos-v1` as the choice of model to convert text into vector embeddings.

```python
embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")
```

2. Use the initialised model to convert the user question into a vector query.

```python
# text that needs to be converted into a vector embeddings
user_question = "I just discovered the course. Can I still join it?"

# converting the text into a vector embedding using the encode method
v_q = embedding_model.encode(user_question)
```

3. Lastly, we want to find the first value of the resulting vector

```python
# What's the first value of the resulting vector
v_q[0]
```

> Answer: 0.078222655

## Question 2. Create the embeddings

1. Load the documents with ids that we prepared in the module.

   > Note: The code for this step was provided by course instructor

2. Create a subset of the questions - the questions for `machine-learning-zoomcamp`.

```python
# make sure the length of the final list is 375
# we only need a subset of of the questions - the questions for machine-learning-zoomcamp
documents_ml = [doc for doc in documents if doc['course'] == "machine-learning-zoomcamp"]
```

3. Create an embedding for both question and answer fields

```python
# create an embedding for both question and answer fields
embeddings = [embedding_model.encode(f"{doc['question']} {doc['text']}") for doc in documents_ml]
```

4. Convert embeddings like into a numpy array so as to get the shape of the array.

```python
# shape of our embeddings
import numpy as np

X = np.array(embeddings)

X.shape
```

> Answer: (375, 768)

## Question 3. Search

1. We need to generate the `scores` which is the cosine similarity between the embeddings matrix and the query vector.

2. The max score is calculated by `scores.max()`. If you want to actually return the best response for the query we can use `print(documents_ml[scores.argmax()]['text'])`.

> Answer: 0.6506573

## Question 4. Hit-rate for our search engine

1. After implementing our own `vector search` class and loading the `ground_truth` dataset (code block for doing so is provided by instructor in homwork), we need to calculate the hit rate using the following function:

```python

from typing import List, Dict

# now to calculate the hit-rate using our VectorSearchEngine
def hit_rate(dataset:List[Dict]):

    for data_dict in dataset:
        data_dict['results'] = search_engine.search(query_vector=embedding_model.encode(data_dict['question']), num_of_res=5)
        res_id = [res['id'] for res in data_dict['results']]
        if data_dict['document'] in res_id:
            data_dict['hit_rate'] = 1
        else:
            data_dict['hit_rate'] = 0

    hit_rate = [data_dict['hit_rate'] for data_dict in dataset]

    # reinstating our dataset into its original form
    dataset = [data_dict.pop("results", None) or data_dict.pop("hit_rate", None) for data_dict in dataset]

    return sum(hit_rate) / len(hit_rate)
```

> Answer: 0.9398907103825137

## Question 5. Indexing with ElasticSearch

1. We need to connect to `elasticsearch`.

```python
from elasticsearch import Elasticsearch

# initialising elasticsearch and checking to see if connection was successful
es_client = Elasticsearch("http://localhost:9200")

if es_client.ping():
    print("Connected to ElasticSearch!")
else:
    print("Connection Failed.")
```

2. Next, we want to define the schema for the index with the following settings - we have to include the `question_text_vector` as that will be field the search will be done on.

```python
# now to create a new index as well as defining the index settings

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
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_text_vector" :{
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
        }
    }
}}

index_name = "course-questions"

# lets delete and create a new index if it exists for ease of re-runs
if es_client.indices.exists(index="course-questions"):
    es_client.indices.delete(index="course-questions")

es_client.indices.create(index=index_name, body=index_settings)
```

3. Now we have to populate the index, but before that we need to modify our `documents` to include the `question_vector_search` key.

```python
# updating our documents to include question_text_vector
documents_ml = [documents_ml[index].update({'question_text_vector': embeddings[index]}) or documents_ml[index] for index in range(len(documents_ml))]

# next we simply have to populate our index using the bulk method
# the bulk method is a `pipeline` where you can perform multiple actions in single request
# for populating the index we only use the `index` action but there are others (e.g. delete, update or create)
index = {'index':{
    '_index': index_name}
    }

operations = [item for doc in documents_ml for item in (index, doc)]

response = es_client.bulk(operations=operations)
```

4. Last step would be to define a search function before performing a search with the given query from question 1.

```python
# last step for this function would be to define the search query function for elasticsearch
def elastic_search_knn(field, vector):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )

    result_docs = [hit['_source'] for hit in es_results['hits']['hits']]

    return result_docs

elastic_search_knn("question_text_vector", v_q)[0]["id"]
```

> Answer: The ID of the document with the highest score: ee58a693

## Question 6. Hit-Rate for ElasticSearch

Following is a modified hit-rate function specifically for `elasticsearch`:

```python
# now to calculate the hit-rate to measure the retreival performance of ElasticSearch
def hit_rate_es(dataset:List[Dict]):

    for data_dict in dataset:
        data_dict['results'] = elastic_search_knn(field="question_text_vector", vector=embedding_model.encode(data_dict["question"]))
        res_id = [res['id'] for res in data_dict['results']]
        if data_dict['document'] in res_id:
            data_dict['hit_rate_es'] = 1
        else:
            data_dict['hit_rate_es'] = 0

    hit_rate = [data_dict['hit_rate_es'] for data_dict in dataset]

    return sum(hit_rate) / len(hit_rate)

hit_rate_es(ground_truth)
```

> Answer: 0.9398907103825137 - this is the same as the direct text search, which means that the approximate search using elasticsearch is just as good as the literal one, perhaps only for this dataset.
