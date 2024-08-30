# Week 3 notes

### Table of contents

- [3.1 Introduction to Vector Databases](#31-introduction-to-vector-databases)
  - [Vector Search](#vector-search)
  - [Vector Embeddings and Indexing](#vector-embeddings-and-indexing)
  - [Vector Search Data and Workflow](#vector-search-data-workflow)

## 3.1 Introduction to Vector Databases

In the evolving landscape of data management, vector databases have emerged as a critical solution for handling vast and diverse datasets (examples of unstructured data that make up to more that 80% of the data being generated today - social media posts, images, videos, audio). Unlike traditional databases, which are limited to structured data, vector databases excel in managing unstructured data and providing relevant results based on context.

> Note: A vector database indexes and stores vector embeddings for fast retrieval and similarity search.

Let's take an image of a cat as an example of handling unstructured data. Based on pixel values alone we cannot search for similar images. And since we cannot store unstructured data in relational databases, the only way to find similar cat images in said database is to assign tags or attributes to the image, often manually, to perform such searches. Again this is not ideal.

![image](https://github.com/user-attachments/assets/a060dbf3-4e8f-47d3-8cf0-5cc4595e6aa6)

Therefore, there was a need to come up with a more viable solution to represent unstructured data, the solution being vector search and vector embeddings!

> Please note that the terms vector search and vector database are related concepts in the field of data management and information retrieval, but they have distinct meanings.

### Vector Search

Vector search is a method of finding similar items in a dataset by comparing their vector representations (a.k.a vector embeddings, which will be discussed in the next section). Unlike traditional keyword-based search, which relies on exact matches, vector search uses mathematical representations of data to find items that are similar in meaning or context. This is a high-level summary and we will look a little deeper into this topic but at this stage I think it would be prudent to make a comparison between `vector search` and `vector database`. Essentially they refer to the same thing, a process for converting unstructured data into `vector embeddings` and storing them as well as indexing the numeric representations for fast retrieval, but I guess the context in which the terms are used could be different. Hence, please find the following differences:

| Aspect | Vector Search | Vector Database |
|--------|---------------|-----------------|
| Definition | A technique to find similar items based on vector representations | A specialized database system for storing, managing, and querying vector data |
| Primary Function | Searching for similar vectors | Storing and managing vector data, including search capabilities |
| Storage | Does not inherently involve storage | Provides persistent storage for vector data |
| Implementation | Can be implemented on various data structures | Purpose-built system for vector data |
| Scope | A method or operation | A complete data management system |
| Optimization | Focuses on search algorithms | Optimized for vector operations, indexing, and scaling |
| Features | Primarily search functionality | Includes data management, indexing, and querying capabilities |
| Use Cases | Can be part of larger systems | Standalone system for vector-based applications |
| Examples | Cosine similarity, Euclidean distance, ANN algorithms | Pinecone, Milvus, Faiss, Weaviate |
| Scalability | Depends on implementation | Often designed for large-scale operations |
| Performance | Varies based on implementation | Generally optimized for high-performance vector operations |

The idea behind the `vector search` concept is to basically to convert our unstructured data like text documents or images into a numerical representation (your vector embedding) and subsequently be stored in a multi-dimensional vector space. This way its easy for the machine to learn and understand, as well as yield more relevant results when performing semantic searches.

Using the same cat example as before, if you provide a cat image, this would be converted to a vector embedding and `vector search` would return the vector embedding closest to our query vector embedding based on the euclidean distance in our vector database. And because we have a `index` structure that often includes a distance metric, the execution time is much shorter as opposed to having to calculate the distance for each vector embedding in our vector database.

### Vector Embeddings and Indexing

### Vector Search Data Workflow

