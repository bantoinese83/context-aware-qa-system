# Context-Aware Q&A System with Knowledge Base & AI-Driven Generation

This project demonstrates how to build a context-aware Q&A system that extracts relevant information from a knowledge base and uses generative AI to provide context-aware responses to user queries.

## Features

- **Dynamic Knowledge Base**: Upload documents like PDFs, text files, or CSVs and create a fully searchable knowledge base.
- **AI-Augmented Responses**: Combines traditional information retrieval with AI-generated responses for human-like answers.
- **Persistent Storage**: Stores question-answer pairs in a SQLite database for future analysis.

## Technologies Used

- **FastAPI**: Backend framework for building APIs.
    - **FAISS**: Vector search engine for efficient document retrieval.
- **LangChain**: Text splitting and document handling.
- **Google Generative AI (Gemini)**: AI-driven text generation for augmented answers.
    - **SQLAlchemy**: ORM for database interaction.
    - **HuggingFace Embeddings**: Pre-trained embeddings for text search.
    - **Python-dotenv**: For loading environment variables from `.env` files.
