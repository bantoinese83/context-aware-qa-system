import logging
import os
from contextlib import contextmanager

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import GenerationConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

knowledge_base_path = "knowledge_base.json"
database_url = "sqlite:///question_answers.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class QuestionAnswer(Base):
    __tablename__ = "question_answers"
    id = Column(Integer, primary_key=True)
    question = Column(String)
    answer = Column(String)


def initialize_database():
    """
    Initializes the database and creates tables if they don't exist.
    """
    try:
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        logger.info("Database initialized and tables created if they didn't exist.")
    except OperationalError as e:
        logger.error(f"Error initializing the database: {e}")
    except Exception as e:
        logger.error(f"Unexpected error initializing the database: {e}")


@contextmanager
def get_session():
    """
    Context manager for database session.
    """
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


# Indexing Pipeline
def create_knowledge_base(source_path):
    """
    Creates a knowledge base from a source document.

    Args:
        source_path (str): Path to the source document.
    """
    try:
        # Determine the loader based on a file extension
        if source_path.endswith('.txt'):
            loader = TextLoader(source_path)
        elif source_path.endswith('.pdf'):
            loader = PyPDFLoader(source_path)
        elif source_path.endswith('.csv'):
            loader = CSVLoader(source_path)
        else:
            raise ValueError("Unsupported file type")

        # Load document
        documents = loader.load()

        # Split into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Embed and store in a vector database
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(knowledge_base_path)
        logger.info(f"Knowledge base created at {knowledge_base_path}")

    except Exception as e:
        logger.error(f"Error creating knowledge base: {e}")


# Generation Pipeline
def answer_question(question):
    """
    Generates a context-aware response to a question.

    Args:
        question (str): The user's question.

    Returns:
        str: The generated response.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(knowledge_base_path, embeddings, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever()
        retrieved_information = retriever.invoke(question)
        logger.debug(f"Retrieved Information: {retrieved_information}")

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        generation_config = GenerationConfig(
            temperature=1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        chat_session = model.start_chat(history=[])
        augmented_query = f"Question: {question}. Info: {retrieved_information}"
        response = chat_session.send_message(augmented_query)
        logger.info(f"Question: {question}, Answer: {response.text}")

        store_question_answer(question, response.text)

        return response.text

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return "Sorry, I couldn't understand your question. Please try again."


def store_question_answer(question, answer):
    """
    Stores the question-answer pair in the database.
    """
    try:
        with get_session() as session:
            session.add(QuestionAnswer(question=question, answer=answer))
            session.commit()
            logger.info(f"Question-answer pair stored in database.")
    except OperationalError as e:
        logger.error(f"Error connecting to the database: {e}")
    except Exception as e:
        logger.error(f"Error storing question-answer pair: {e}")


# Example Usage
if __name__ == "__main__":
    initialize_database()

    create_knowledge_base("fastapi_tutorial.pdf")

    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break
        answer = answer_question(question)
        print(f"Answer: {answer}")
