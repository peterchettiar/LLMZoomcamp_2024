# Week 3 notes

### Table of contents

- [3.1 Introduction to Vector Databases](#31-introduction-to-vector-databases)
  - [Vector Search](#vector-search)
  - [Vector Embeddings and Indexing](#vector-embeddings-and-indexing)
  - [Approximate Nearest Neighbour (ANN)](#approximate-nearest-neighbour-ann)
  - [Vector Search Data and Workflow](#vector-search-data-workflow)

## 3.1 Introduction to Vector Databases

In the evolving landscape of data management, vector databases have emerged as a critical solution for handling vast and diverse datasets (examples of unstructured data that make up to more that 80% of the data being generated today - social media posts, images, videos, audio). Unlike traditional databases, which are limited to structured data, vector databases excel in managing unstructured data and providing relevant results based on context.

> Note: A vector database indexes and stores vector embeddings for fast retrieval and similarity search.

Let's take an image of a cat as an example of handling unstructured data. Based on pixel values alone we cannot search for similar images. And since we cannot store unstructured data in relational databases, the only way to find similar cat images in said database is to assign tags or attributes to the image, often manually, to perform such searches. Again this is not ideal.

![image](https://github.com/user-attachments/assets/a060dbf3-4e8f-47d3-8cf0-5cc4595e6aa6)

Therefore, there was a need to come up with a more viable solution to represent unstructured data, the solution being vector search and vector embeddings!

> Please note that the terms vector search and vector database are related concepts in the field of data management and information retrieval, but they have distinct meanings.

### Vector Search

Vector search is a method of finding similar items in a dataset by comparing their vector representations (a.k.a `vector embeddings`, which will be discussed in the next section). Unlike traditional keyword-based search, which relies on exact matches, vector search uses mathematical representations of data to find items that are similar in meaning or context. This is a high-level summary and we will look a little deeper into this topic but at this stage I think it would be prudent to make a comparison between `vector search` and `vector database`. Essentially they refer to the same thing, a process for converting unstructured data into `vector embeddings` and storing them as well as indexing the numeric representations for fast retrieval, but I guess the context in which the terms are used could be different. Hence, please find the following differences:

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

The idea behind the `vector search` concept is to basically to convert our unstructured data like text documents or images into a numerical representation (your vector embedding) and subsequently be stored in a multi-dimensional vector space. This way its easy for the machine to learn and understand, as well as yield more relevant results when performing semantic searches.

Using the same cat example as before, if you provide a cat image, this would be converted to a vector embedding and `vector search` would return the vector embedding closest to our query vector embedding based on the euclidean distance (i.e. straight line distance between two vectors in a multidimensional space) or cosine similarity (i.e. cosine of the angle between two vectors - range from -1 to 1 with 1 being an identical vector) in our vector database. And because we have a `index` structure that often includes a distance metric, the execution time is much shorter for the search process as opposed to having to calculate the distance for each vector embedding in our vector database.

So you maybe wondering what is the purpose of all this, its simply to enable the following use cases:

1. Long-term memory for LLMs
2. Semantic search; search based on the meaning or context
3. Similarity search for text, images, audio, or video data
4. Recommendation engine

### Vector Embeddings and Indexing

At this point we should already have a working knowledge of `vector embeddings` but the officical definition by [elastic](https://www.elastic.co/what-is/vector-embedding) is:

_They are a way to convert words and sentences and other data into numbers that capture their meanings and relationships. They represent different data types as points in a multidimensional space, where similar data points are clustered closer together. These numerical representations help machines understand and process this data more effectively._

So the way to convert unstructured data to a `vector embedding` is through the use of ML Models, depending on the type of data you are working with. Following are a few examples of the type of embeddings:

| Type of Embedding   | Description                                                                                                                                                  | Examples/Techniques                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Word embeddings     | Represent individual words as vectors, capturing semantic relationships and contextual information from large text corpora.                                  | Word2Vec, GloVe, FastText                                                        |
| Sentence embeddings | Represent entire sentences as vectors, capturing the overall meaning and context of the sentences.                                                           | Universal Sentence Encoder (USE), SkipThought                                    |
| Document embeddings | Represent documents (anything from newspaper articles to academic papers) as vectors, capturing the semantic information and context of the entire document. | Doc2Vec, Paragraph Vectors                                                       |
| Image embeddings    | Represent images as vectors by capturing different visual features.                                                                                          | Convolutional neural networks (CNNs), ResNet, VGG                                |
| User embeddings     | Represent users in a system or platform as vectors, capturing user preferences, behaviors, and characteristics.                                              | Used in recommendation systems, personalized marketing, user segmentation        |
| Product embeddings  | Represent products in ecommerce or recommendation systems as vectors, capturing a product's attributes, features, and other semantic information.            | Used to compare, recommend, and analyze products based on vector representations |

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
3. Tree-Based indexing - use a tree-like structure to partition the vector database thereby eliminating large portions of data during search
4. Graph-Based indexing - constructs a graph like structure where each node represents a vector, and edges connect nodes based on proximity (similarity)

### Approximate Nearest Neighbour (ANN)

Since we are using `elasticsearch` as our choice of search engine, we can take a deeper look into their method for indexing - ANN algoritms.

Approximate Nearest Neighbour (ANN) is an algorithm that finds a data point in a data set that is very close to the query point, but not necessarily the absolute closest one. This is an upgrade from traditional NN algorthms that searches through all the data to find the perfect match, which can be time consuming as well as computationally expensive given that data sources gets larger each year. Hence, ANNs are game changers as they use intelligent shortcuts and data structures to efficiently navigate the search space. So instead of taking up huge amounts of time and resources, it can identify data points with much less effort that are close enough to be useful in most practical scenarios.

Now that we know what ANNs are as well as their purpose of building vector indexes, we can proceed to understand how they work. Generally how these algorithms work is firstly a **dimensionality reduction** technique being deployed followed by a **defined metric** to calculate the similarity between the query vector and all other vectors in the table.

There are types of ANNs, to name a few:

1. KD-trees
2. Local-sensitivity hashing (LSH)
3. Annoy
4. Linear scan algorithm
5. Inverted file (IVF) indexes
6. Hierarchical Navigational Small Worlds (HNSW)

Let's take a closer look into LSH to get a deeper understanding of how ANNs work. LSH builds the index in the vector database by using a hashing function. Vector embeddings that are nearby each other are hashed to the same bucket. We can then store all these similar vectors in a single table or bucket. When a query vector is provided, its nearest neighbours can be found by hashing the query vector, and then computing the similartiy metric for all the vectors in the table for all other vectors that hashed to the same value. This indexing strategy optimizes for speed and finding.

### Vector Search Data Workflow

To summarise what we have discussed, the below diagram visually describes the end-to-end workflow of `vector search`

![image](https://github.com/user-attachments/assets/5ec81fcd-8361-4db0-a4f7-6103ffca15fc)

So starting from the left-hand side of the image, we have the unstructured data sources where data is being pulled and converted into `vector embeddings` using ML models. Again, the data type determines the ML model being deployed for ths transformation. For example, to convert word to word embeddings we use Word2Vec.

After the transformation, these `vector embeddings` undergo an indexing process using Approximate Nearest Neighbours (ANNs) such as Local-Sensitivity Hashing (LSH) where `vector embedding` are grouped with other `vector embeddings` with high similarity scores.

On the other side the query goes through a similar process where the query is converted into an embedding as well as undergoing an indexing process. Naturally, the query index will enable the search of similar vector embedding indices based on the similarity score with the query index and finally providing the results.

## 3.2 Semantic Search Engine with ElasticSearch
