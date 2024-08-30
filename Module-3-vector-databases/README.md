# Week 3 notes

### Table of contents

- [3.1 Introduction to Vector Databases](#31-introduction-to-vector-databases)
  - [What is Vector Search?]()
  - [Why use Vector Search?]()
  - [How does Vector Search work?]()

## 3.1 Introduction to Vector Databases

In the evolving landscape of data management, vector databases have emerged as a critical solution for handling vast and diverse datasets (examples of unstructured data that make up to more that 80% of the data being generated today - social media posts, images, videos, audio). Unlike traditional databases, which are limited to structured data, vector databases excel in managing unstructured data and providing relevant results based on context.

> Note: A vector database indexes and stores vector embeddings for fast retrieval and similarity search.

Let's take an image of a cat as an example of handling unstructured data. Based on pixel values alone we cannot search for similar images. And since we cannot store unstructured data in relational databases, the only way to find similar cat images is assign tags or texts to the image in that database, often manually, to perform such searches.

### What is Vector Search?

Vector search is a method of finding similar items in a dataset by comparing their vector representations. Unlike traditional keyword-based search, which relies on exact matches, vector search uses mathematical representations of data to find items that are similar in meaning or context.

### Why use vector search?
