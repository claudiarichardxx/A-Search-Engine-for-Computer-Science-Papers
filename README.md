# A Computer Science Paper Search Engine

A simple search engine for computer science papers with two modes of retrieval:  
- **Syntactic Search**: Traditional keyword-based matching  
- **Semantic Search**: Embedding-based similarity search for contextual results

🎥 [Watch Demo](https://drive.google.com/file/d/138yKlAD_liaoGztbctpOrMsLbIMtKM61/view?usp=drive_link)

---

## 🔍 Overview

This project demonstrates a dual-search interface over a corpus of academic papers:
- **Syntactic Search** uses traditional information retrieval methods.
- **Semantic Search** leverages embeddings from pre-trained language models to retrieve contextually relevant documents, even when exact keywords don't match.

Use cases include:
- Quickly finding relevant research papers
- Exploring related work through vector-based semantic similarity
- Comparing traditional vs modern retrieval techniques


🔍 Built with:
- React + Vite frontend (JS, CSS, HTML)
- Python backend with Elasticsearch
- Data sourced from [arXiv.org](https://arxiv.org/)

---

## 🚀 Features

- **Syntactic Search** using Elasticsearch’s default retrieval (BM25).
- **Semantic Search** using BERT embeddings for contextual similarity.
- Custom **clustering-based indexing** of papers for more relevant retrieval.
- ArXiv **scraper pipeline** to fetch and update research papers.
- Fast semantic retrieval via **merge sort over embedding distances**.

---

## 🧱 Tech Stack

### Frontend
- React
- Vite
- JavaScript, CSS, HTML

### Backend
- Python
- Elasticsearch
- Sentence-BERT (for embeddings)
- Requests, BeautifulSoup (for scraping)

---

## 🗃️ How It Works

1. **Scraping**: A Python scraper collects metadata and abstracts from arXiv.
2. **Clustering & Indexing**: Papers are clustered and indexed into Elasticsearch.
3. **Syntactic Search**: Elasticsearch returns results based on keyword matches.
4. **Semantic Search**:
   - Query is embedded using BERT
   - Embedding compared to paper vectors
   - Merge sort used for efficient top-K selection

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- Node.js


