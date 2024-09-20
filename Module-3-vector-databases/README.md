# Week 3 notes

### Table of contents

- [3.1 Introduction to Vector Databases](#31-introduction-to-vector-databases)
  - [Vector Search](#vector-search)
  - [Vector Embeddings and Indexing](#vector-embeddings-and-indexing)
  - [Approximate Nearest Neighbour (ANN)](#approximate-nearest-neighbour-ann)
  - [Vector Search Data and Workflow](#vector-search-data-workflow)

## 3.1 Introduction to Vector Databases

In the evolving landscape of data management, vector databases have emerged as a critical solution for handling vast and diverse datasets (examples of unstructured data that make up to more than 80% of the data being generated today - social media posts, images, videos, audio). Unlike traditional databases, which are limited to structured data, vector databases excel in managing unstructured data and providing relevant results based on context.

> Note: A vector database indexes and stores vector embeddings for fast retrieval and similarity search.

Let's take an image of a cat as an example of handling unstructured data. Based on pixel values alone we cannot search for similar images. And since we cannot store unstructured data in relational databases, the only way to find similar cat images in said database is to assign tags or attributes to the image, often manually, to perform such searches. Again this is not ideal.

![image](https://github.com/user-attachments/assets/a060dbf3-4e8f-47d3-8cf0-5cc4595e6aa6)

Therefore, there was a need to come up with a more viable solution to represent unstructured data, the solution being vector search and vector embeddings!

> Please note that the terms vector search and vector database are related concepts in the field of data management and information retrieval, but they have distinct meanings.

### Vector Search

Vector search is a method of finding similar items in a dataset by comparing their vector representations (a.k.a `vector embeddings`, which will be discussed in the next section). Unlike traditional keyword-based search, which relies on exact matches, vector search uses mathematical representations of data to find items that are similar in meaning or context. This is a high-level summary and we will look a little deeper into this topic but at this stage, I think it would be prudent to make a comparison between `vector search` and `vector database`. Essentially they refer to the same thing, a process for converting unstructured data into `vector embeddings` and storing them as well as indexing the numeric representations for fast retrieval, but I guess the context in which the terms are used could be different. Hence, please find the following differences:

| Aspect           | Vector Search                                                     | Vector Database                                                               |
| ---------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Definition       | A technique to find similar items based on vector representations | A specialized database system for storing, managing, and querying vector data |
| Primary Function | Searching for similar vectors                                     | Storing and managing vector data, including search capabilities               |
| Storage          | Does not inherently involve storage                               | Provides persistent storage for vector data                                   |
| Implementation   | Can be implemented on various data structures                     | Purpose-built system for vector data                                          |
| Scope            | A method or operation                                             | A complete data management system                                             |
| Optimization     | Focuses on search algorithms                                      | Optimized for vector operations, indexing, and scaling                        |
| Features         | Primarily search functionality                                    | Includes data management, indexing, and querying capabilities                 |
| Use Cases        | Can be part of larger systems                                     | Standalone system for vector-based applications                               |
| Examples         | Cosine similarity, Euclidean distance, ANN algorithms             | Pinecone, Milvus, Faiss, Weaviate                                             |
| Scalability      | Depends on implementation                                         | Often designed for large-scale operations                                     |
| Performance      | Varies based on implementation                                    | Generally optimized for high-performance vector operations                    |

The idea behind the `vector search` concept is to basically convert our unstructured data like text documents or images into a numerical representation (your vector embedding) and subsequently be stored in a multi-dimensional vector space. This way it's easy for the machine to learn and understand, as well as yield more relevant results when performing semantic searches.

Using the same cat example as before, if you provide a cat image, this would be converted to a vector embedding, and `vector search` would return the vector embedding closest to our query vector embedding based on the Euclidean distance (i.e. straight line distance between two vectors in a multidimensional space) or cosine similarity (i.e. cosine of the angle between two vectors - range from -1 to 1 with 1 being an identical vector) in our vector database. And because we have an `index` structure that often includes a distance metric, the execution time is much shorter for the search process as opposed to having to calculate the distance for each vector embedding in our vector database.

So you may be wondering what the purpose of all this, is simply to enable the following use cases:

1. Long-term memory for LLMs
2. Semantic search; search based on the meaning or context
3. Similarity search for text, images, audio, or video data
4. Recommendation engine

### Vector Embeddings and Indexing

At this point we should already have a working knowledge of `vector embeddings` but the official definition by [elastic](https://www.elastic.co/what-is/vector-embedding) is:

_They are a way to convert words, sentences and other data into numbers that capture their meanings and relationships. They represent different data types as points in a multidimensional space, where similar data points are clustered closer together. These numerical representations help machines understand and process this data more effectively._

So the way to convert unstructured data to a `vector embedding` is through the use of ML Models, depending on the type of data you are working with. Following are a few examples of the type of embeddings:

| Type of Embedding   | Description                                                                                                                                                  | Examples/Techniques                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Word embeddings     | Represent individual words as vectors, capturing semantic relationships and contextual information from large text corpora.                                  | Word2Vec, GloVe, FastText                                                        |
| Sentence embeddings | Represent entire sentences as vectors, capturing the overall meaning and context of the sentences.                                                           | Universal Sentence Encoder (USE), SkipThought                                    |
| Document embeddings | Represent documents (anything from newspaper articles to academic papers) as vectors, capturing the semantic information and context of the entire document. | Doc2Vec, Paragraph Vectors                                                       |
| Image embeddings    | Represent images as vectors by capturing different visual features.                                                                                          | Convolutional neural networks (CNNs), ResNet, VGG                                |
| User embeddings     | Represent users in a system or platform as vectors, capturing user preferences, behaviors, and characteristics.                                              | Used in recommendation systems, personalized marketing, user segmentation        |
| Product embeddings  | Represent products in e-commerce or recommendation systems as vectors, capturing a product's attributes, features, and other semantic information.            | Used to compare, recommend, and analyze products based on vector representations |

The following is a simple guideline for creating a `vector embedding`:

| Step                  | Description                                                                                                                                                                                                                |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Data Collection    | Gather a large dataset representing the type of data for which you want to create embeddings (e.g., text or images).                                                                                                       |
| 2. Preprocessing      | Clean and prepare the data by removing noise, normalizing text, resizing images, or other tasks depending on the data type.                                                                                                |
| 3. Model Selection    | Choose a neural network model that fits your data and goals.                                                                                                                                                               |
| 4. Training           | Feed the preprocessed data into the model. The model learns patterns and relationships by adjusting its internal parameters (e.g., associating words that often appear together or recognizing visual features in images). |
| 5. Vector Generation  | As the model learns, it generates numerical vectors (embeddings) representing the meaning or characteristics of each data point.                                                                                           |
| 6. Quality Assessment | Evaluate the quality and effectiveness of the embeddings by measuring their performance on specific tasks or using human evaluation.                                                                                       |
| 7. Implementation     | Once satisfied with the embeddings' performance, use them for analyzing and processing your data sets.                                                                                                                     |

Now moving on to the concept of `indexing`. As mentioned previously, this is another important process to enable fast retrieval for `vector search`. There are a few types of indexing methods such as:

1. Flat index (brute force) - compares the query with every single vector stored in the database
2. Approximate Nearest Neighbour (ANN) Methods - as the name suggests, using algorithms to find the close vectors that are similar or approximate to the query vector
3. Tree-Based indexing - use a tree-like structure to partition the vector database thereby eliminating large portions of data during the search
4. Graph-based indexing - constructs a graph-like structure where each node represents a vector, and edges connect nodes based on proximity (similarity)

### Approximate Nearest Neighbour (ANN)

Since we are using `elasticsearch` as our choice of search engine, we can take a deeper look into their method for indexing - ANN algorithms.

Approximate Nearest Neighbour (ANN) is an algorithm that finds a data point in a data set that is very close to the query point, but not necessarily the absolute closest one. This is an upgrade from traditional NN algorithms that search through all the data to find the perfect match, which can be time-consuming as well as computationally expensive given that data sources get larger each year. Hence, ANNs are game changers as they use intelligent shortcuts and data structures to efficiently navigate the search space. So instead of taking up huge amounts of time and resources, it can identify data points with much less effort that are close enough to be useful in most practical scenarios.

Now that we know what ANNs are as well as their purpose of building vector indexes, we can proceed to understand how they work. Generally how these algorithms work is firstly a **dimensionality reduction** technique being deployed followed by a **defined metric** to calculate the similarity between the query vector and all other vectors in the table.

There are types of ANNs, to name a few:

1. KD-trees
2. Local-sensitivity hashing (LSH)
3. Annoy
4. Linear scan algorithm
5. Inverted file (IVF) indexes
6. Hierarchical Navigational Small Worlds (HNSW)

Let's take a closer look into LSH to get a deeper understanding of how ANNs work. LSH builds the index in the vector database by using a hashing function. Vector embeddings that are near each other are hashed to the same bucket. We can then store all these similar vectors in a single table or bucket. When a query vector is provided, its nearest neighbours can be found by hashing the query vector, and then computing the similarity metric for all the vectors in the table for all other vectors that hashed to the same value. This indexing strategy optimizes for speed and finding.

### Vector Search Data Workflow

To summarise what we have discussed, the below diagram visually describes the end-to-end workflow of `vector search`

![image](https://github.com/user-attachments/assets/5ec81fcd-8361-4db0-a4f7-6103ffca15fc)

So starting from the left-hand side of the image, we have the unstructured data sources where data is being pulled and converted into `vector embeddings` using ML models. Again, the data type determines the ML model being deployed for this transformation. For example, to convert word-to-word embeddings we use Word2Vec.

After the transformation, these `vector embeddings` undergo an indexing process using Approximate Nearest Neighbours (ANNs) such as Local-Sensitivity Hashing (LSH) where `vector embedding` is grouped with other `vector embeddings` with high similarity scores.

On the other side, the query goes through a similar process where the query is converted into an embedding as well as undergoing an indexing process. Naturally, the query index will enable the search of similar vector embedding indices based on the similarity score with the query index and finally provide the results.

## 3.2 Semantic Search Engine with ElasticSearch

### Introduction

In this chapter, we will explore how to build a semantic search engine using Elasticsearch and Python. Semantic search using Elasticsearch is a specific implementation of vector search that leverages Elasticsearch's capabilities to perform semantic search. It enhances traditional search by understanding the context and meaning behind the search terms, going beyond keyword matching to deliver more relevant results.
![image](https://github.com/user-attachments/assets/59f079f5-aa30-460d-8fe0-1a939fa7ded5)

**Why use Elasticsearch for Semantic Search?**

* _Scalability_: Elasticsearch can handle large volumes of data and high query loads.
* _Flexibility_: It supports various types of data, including text, numbers, and geospatial data.
* _Advanced Features_: Elasticsearch offers advanced search features like full-text search, filtering, and aggregations.

### Understanding Documents and Indexes in Elasticsearch

Elasticsearch is a distributed search engine that stores data in the form of documents. Two very important concepts in Elasticsearch are documents and indexes.

**Documents**

A document is a collection of fields with their associated values. Each document is a JSON object that contains data in a structured format. For example, a document representing a book might contain fields like title, author, and publication date.

**Indexes**

An index is a collection of documents that is stored in a highly optimized format designed to perform efficient searches. Indexes are similar to tables in a relational database, but they are more flexible and can store complex data structures.

To work with Elasticsearch, you need to organize your data into documents and then add all your documents into an index. This allows Elasticsearch to efficiently search and retrieve relevant documents based on the search queries.

### Steps to run Semantic Search using ElasticSearch

**Step 1: Setting up the environment**

The process involves setting up a Docker container, preparing data, generating embeddings with a pre-trained model, and indexing these embeddings into Elasticsearch.

First, check if Docker is running. If not, use a command from a previous module to start a Docker container for Elasticsearch:
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

I used the [docker-compose](https://github.com/peterchettiar/LLMzoomcamp_2024/blob/main/Module-2-open-source-llm/docker-compose.yaml) file from Week 2 but made a slight tweak by adding a `volume` flag. Instead of using the default directory in codespaces, I remounted onto a `/tmps` folder on the host machine so that there is more disk space available. The additional item is as follows:
```yaml
volumes:
  - /tmp/elasticsearch_data:/usr/share/elasticsearch/data
```
So to break it down, `/tmp/elasticsearch_data` is the data directory in the host machine and `/usr/share/elasticsearch/data` is the default directory inside the Elasticsearch container. Since, we made these changes we need to re-build the image using `docker-compose up --build -d` to make sure that the new changes are applied.

There may be also a possibility that the `elasticsearch` container exits unexpectedly. It's good practice to check the logs by running `docker logs elasticsearch` to see what the errors are. Chances are, it may have exited unexpectedly due to the changes we made when mounting volumes. The reason for this is that we do not have the permissions to access these folders by default. Hence, we need to change this by running the command `sudo chown -R 1000:1000 /tmp/elasticsearch_data`. Basically what we are doing here is that we are changing the ownership to all the files and subdirecotries recusrsicely to a new user and group which in our case is both 1000, this is referring to the user in the `elasticsearch` container.

> Note: Please make sure to have the container running before proceeding.

**Step 2: Data Loading and Preprocessing**

In this step, we will load the `documents.json` file and preprocess it to flatten the hierarchy, making it suitable for Elasticsearch. The `documents.json` file contains a list of courses, each with a list of documents. We will extract each document and add a `course` field, indicating which course it belongs to.

**Step 3: Embeddings - Sentence Transformers**

To perform a semantic search, we need to convert our documents into dense vectors (embeddings) that capture the semantic meaning of the text. We will use a pre-trained model from the Sentence Transformers library to generate these embeddings. These embeddings are then indexed into Elasticsearch. These embeddings enable us to perform semantic search, where the goal is to find text that is contextually similar to a given query.

The `text` and `question` fields are the actual data fields containing the primary information, whereas other fields like `section` and `course` are more categorical and less informative for the purpose of creating meaningful embeddings.

* Install the` sentence_transformers` library.
* Load the pre-trained model and use it to generate embeddings for our documents.

```python
# Load a pretrained sentence transformer model

model = SentenceTransformer("all-mpnet-base-v2") # best pretrained model in their library

documents = [doc.update({'text_vector': model.encode(doc['text']).tolist()}) or doc for doc in documents]
```
Pretty much what we did here was to convert the `text` field into an embedding and creating a new key called `text_vector` for each `doc`

**Step 4: Connecting to ElasticSearch**

In this step, we will set up a connection to an Elasticsearch instance. Make sure you have Elasticsearch running.

```python
# establishing the connection with ElasticSearch

es_client = Elasticsearch("http://localhost:9200")

es_client.info()
```
**Step 5: Create Mappings and Index**

We will define the mappings and create the index in Elasticsearch, where the generated embeddings will also be stored.

Mapping is specifying how documents and their fields are structured and indexed in Elasticsearch. Each document is composed of various fields, each assigned a specific data type.

Similar to a database schema, mapping outlines the structure of documents, detailing the fields, their data types (e.g., string, integer, or date), and how these fields should be indexed and stored.

By defining documents and indices, we ensure that an index acts like a table of contents in a book, facilitating efficient searches.

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
            "course": {"type": "keyword"},
            "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}

index_name = "course-questions"

# Delete the index if it exists
es_client.indices.delete(index=index_name, ignore_unavailable=True)

# Create the index
es_client.indices.create(index=index_name, body=index_settings)
```

**Step 6: Adding the documents to the Index**

We then add the preprocessed documents along with their embeddings to the Elasticsearch index. This allows Elasticsearch to store and manage the documents efficiently, enabling fast and accurate search queries.

I used the `bulk` method instead of the conventional `index` method, just for exploratory purposes:
```python
# lastly to populate the index with our documents list using the bulk method instead of the conventional create method

index = {"index": {
    "_index":index_name
}}

operations =  [item for doc in documents for item in (index, doc)]

resp = es_client.bulk(operations = operations, timeout="120s")
```
**Step 7: Performing Semantic Search with Filter**

Based on our workflow diagram at the start of the section, the other side of the coin is the user query. This too needs to undergo a transformation process to be converted into a vector embedding, followed by defining the parameters of the query before running the `search` method.

1. Let's transform our query into an embedding.
```python
# Here we will use the search term that was used in the course - again we need to convert our search term into an embedding

search_term = 'Windows or Mac?'

vector_search_term = model.encode(search_term)
```
2. Define our query parameters.
```python
# we need to define the parameters of our query, that includes our search term vector as well

knn_query = {
    "field" : "text_vector",  # the field in which the search term should be queried
    "query_vector" : vector_search_term,  # the embedding of our search term
    "k" : 5,  # the number of nearest documents to be retrieved that matches the search term 
    "num_candidates" : 10000 # group of documents the search is going to look into
}
```
3. Running a search query using a filter.
```python
# running our semantic search with a filter in place - `match` is used as a filter field

response = es_client.search(
    index=index_name,
    query={
        "match" : {"section": "General course-related questions"},
    },
    knn=knn_query,
    size=5
)
```

For the full notebook of the example we looked through, please click [here](https://github.com/peterchettiar/LLMzoomcamp_2024/blob/main/Module-3-vector-databases/semantic_search_example.ipynb).
