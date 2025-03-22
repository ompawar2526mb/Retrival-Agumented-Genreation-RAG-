# Retrieval-Augmented Generation (RAG) with OpenAI

## Overview
This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline using **OpenAI's LLM** for text generation and **OpenAI's embedding model** for vector-based search. The system enhances the language model's responses by retrieving relevant information from an external knowledge base before generating responses.

## Features
- **OpenAI's LLM** for natural language generation.
- **OpenAI Embeddings** for semantic search.
- **Efficient Retrieval Pipeline** using chroma DB vector databases.
- **Context-Enhanced Responses** for more accurate and informed outputs.

## Installation
Clone the repository and install dependencies:

```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

## Configuration
Set up your OpenAI API key as an environment variable:

```sh
export OPENAI_API_KEY="your-api-key"
```

Or create a `.env` file:

```
OPENAI_API_KEY=your-api-key
```

## Usage
Run the main script to start the RAG pipeline:

```sh
python main.py --query "Your question here"
```

## Workflow
1. **Embedding Creation**: Text documents are converted into vector embeddings using OpenAI's embedding model.
2. **Storage & Retrieval**: Embeddings are stored in a vector database (e.g., FAISS, Pinecone, Chroma) for efficient similarity search.
3. **Query Processing**: When a query is received, the most relevant documents are retrieved based on semantic similarity.
4. **Generation**: The retrieved context is provided to OpenAI's LLM, enhancing its response quality.

## Dependencies
- `openai`
- `faiss` / `pinecone` / `chromadb` (for vector search)
- `python-dotenv`
- `numpy`
- `tqdm`

## Example
```python
from openai import OpenAI
import faiss
import numpy as np

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

# Convert query to embeddings
query = "What is retrieval-augmented generation?"
query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002").data

# Retrieve relevant docs (example with FAISS)
index = faiss.read_index("vector_store.index")
D, I = index.search(np.array([query_embedding]), k=5)

# Generate response using retrieved context
context = "\n".join(retrieved_docs)
prompt = f"Context: {context}\n\nAnswer the following question:\n{query}"
response = client.completions.create(model="gpt-4", prompt=prompt, max_tokens=200)

print(response["choices"][0]["text"].strip())
```

## Contributing
Pull requests are welcome! Please follow best practices and open an issue before making major changes.

## License
This project is licensed under the MIT License.

---

**Author:** Your Name | [GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)

