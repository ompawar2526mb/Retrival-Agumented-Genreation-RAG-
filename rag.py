import os
import sys
import argparse
from typing import List, Dict, Any, Optional
import tempfile
import logging

# Environment and configuration
import dotenv
from dotenv import load_dotenv

# Document processing
import pandas as pd
import docx
import PyPDF2
from pathlib import Path
import glob

# Vector database
import chromadb
from chromadb.utils import embedding_functions

# LLM integration
import openai
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents of various formats into text chunks."""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Extract text from PDF files."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    
    @staticmethod
    def read_docx(file_path: str) -> str:
        """Extract text from DOCX files."""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    @staticmethod
    def read_txt(file_path: str) -> str:
        """Extract text from TXT files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @classmethod
    def process_document(cls, file_path: str) -> str:
        """Process a document based on its file extension."""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return cls.read_pdf(file_path)
            elif file_extension == '.docx':
                return cls.read_docx(file_path)
            elif file_extension == '.txt':
                return cls.read_txt(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_extension} for file {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks of specified size with overlap."""
        if not text:
            return []
            
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks


class VectorStore:
    """Handle interactions with ChromaDB vector store."""
    
    def __init__(self, collection_name: str = "document_collection"):
        """Initialize the vector store with OpenAI embeddings."""
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Set up OpenAI embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name="text-embedding-3-small"
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Collection '{collection_name}' loaded")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Collection '{collection_name}' created")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> None:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents to add to vector store")
                return
                
            # Generate IDs based on document content
            ids = [f"doc_{i}_{hash(doc) % 10000}" for i, doc in enumerate(documents)]
            
            # Add metadatas if not provided
            if not metadatas:
                metadatas = [{"source": "document", "index": i} for i in range(len(documents))]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store for relevant documents."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


class RAGSystem:
    """Main RAG system that combines document processing, vector store, and LLM generation."""
    
    def __init__(self):
        """Initialize the RAG system."""
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # System prompts for different functions
        self.system_prompts = {
            "default": """You are a helpful AI assistant that answers questions based on the provided context.
If the context doesn't contain relevant information, acknowledge this and provide a general response.
Always be truthful and admit when you don't know something.
Base your answers only on the provided context and your general knowledge.""",
            
            "with_context": """You are a helpful AI assistant answering questions based on specific context provided below.
Use the context information to provide accurate, relevant answers.
If the answer cannot be found in the context, acknowledge this and provide your best response based on general knowledge.
Do not make up information that is not supported by the context.

Context:
{context}

Answer the user's question based on this context."""
        }
    
    def ingest_documents(self, file_paths: List[str]) -> bool:
        """Process and ingest documents into the vector store."""
        all_chunks = []
        all_metadatas = []
        
        # Process each file
        for file_path in file_paths:
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    continue
                
                # Process document
                document_text = self.document_processor.process_document(file_path)
                if not document_text:
                    continue
                
                # Chunk document
                chunks = self.document_processor.chunk_text(document_text)
                
                # Create metadata for each chunk
                metadatas = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]
                
                # Add to lists
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                
                logger.info(f"Processed {file_path} into {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Add to vector store
        if all_chunks:
            self.vector_store.add_documents(all_chunks, all_metadatas)
            return True
        else:
            logger.warning("No content was extracted from the provided files")
            return False
    
    def query_llm(self, prompt: str, context: str = None, temperature: float = 0.3) -> str:
        """Query the LLM with a prompt and optional context."""
        try:
            # Select system prompt based on context availability
            if context:
                system_prompt = self.system_prompts["with_context"].format(context=context)
            else:
                system_prompt = self.system_prompts["default"]
            
            # Make API call to OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",  # Can be configured from .env in a more advanced version
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the RAG pipeline."""
        # Query vector store for relevant documents
        results = self.vector_store.query(question)
        
        # Extract documents and their sources
        documents = results["documents"][0] if results["documents"] and results["documents"][0] else []
        metadatas = results["metadatas"][0] if results["metadatas"] and results["metadatas"][0] else []
        
        # Format context from retrieved documents
        if documents:
            context_parts = []
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                source = meta.get("source", "Unknown")
                context_parts.append(f"[Document {i+1} from {source}]\n{doc}\n")
            
            context = "\n".join(context_parts)
        else:
            context = None
            return self.query_llm(question)  # If no context, just query the LLM directly
        
        # Query LLM with context
        return self.query_llm(question, context)


def main():
    """Main function to run the RAG application."""
    parser = argparse.ArgumentParser(description="RAG Application with ChromaDB and OpenAI")
    parser.add_argument("--ingest", nargs="+", help="Ingest documents into the vector store")
    parser.add_argument("--query", type=str, help="Query the RAG system")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("Error: .env file not found. Please create a .env file with your OPENAI_API_KEY.")
        return
    
    # Initialize RAG system
    try:
        rag_system = RAGSystem()
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Process arguments
    if args.ingest:
        print(f"Ingesting {len(args.ingest)} documents...")
        success = rag_system.ingest_documents(args.ingest)
        if success:
            print("Documents ingested successfully.")
        else:
            print("Failed to ingest documents.")
    
    if args.query:
        print(f"Question: {args.query}")
        answer = rag_system.answer_question(args.query)
        print(f"Answer: {answer}")
    
    if args.interactive:
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            user_input = input("\nQuestion: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            answer = rag_system.answer_question(user_input)
            print(f"\nAnswer: {answer}")
    
    # If no arguments provided, show help
    if not (args.ingest or args.query or args.interactive):
        parser.print_help()


if __name__ == "__main__":
    main()