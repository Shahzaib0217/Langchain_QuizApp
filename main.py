import streamlit as st
import os
import json
import asyncio
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_transformers import DoctranQATransformer
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers.openai_functions import create_metadata_tagger
from dotenv import load_dotenv

load_dotenv()

# Setting environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_MODEL"] = os.getenv("OPENAI_API_MODEL")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


# Function to read documents from a directory
def load_docs(directory):
    """
    Load PDF documents from a directory.

    Args:
        directory (str): Directory path containing PDF files.

    Returns:
        list: List of loaded documents.
    """
    print("Starting Documents loading ....")
    try:
        loader = PyPDFDirectoryLoader(directory)
        documents = loader.load()
        print("Documents Loaded Successfully!")
        return documents
    except Exception as e:
        print("Some Error occurred during Document loading", e)


# Function to split documents into snippets of text
def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks of text.

    Args:
        documents (list): List of documents.
        chunk_size (int): Size of each chunk in tokens.
        chunk_overlap (int): Overlap between chunks in tokens.

    Returns:
        list: List of split documents.
    """
    try:
        print("Starting Documents splitting ....")
        text_splitter = RecursiveCharacterTextSplitter(
            length_function=len,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(documents)
        print("Documents splitted successfully!")
        return split_docs
    except Exception as e:
        print("Some Error occurred during Document splitting", e)


# Function to store document chunks to Supabase
def store_doc_to_supabase(doc_splits):
    try:
        print("Storing Documents to SupaBase")
        vector_store.add_documents(doc_splits)
    except Exception as e:
        pass


# Initializing OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initializing Supabase vector store
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Initializing QA transformer
qa_transformer = DoctranQATransformer()


# Function to create questions and answers using GPT
async def QA_maker(doc_splits):
    try:
        print("Starting making Questions ....")
        custom_prompt = """
        Please Add questions about India
        """
        transformed_document = await qa_transformer.atransform_documents(doc_splits[:4], prompt=custom_prompt)
        metadata_list = []
        for doc in transformed_document:
            metadata_list.extend(doc.metadata["questions_and_answers"])
        output_file = "Questions.json"
        with open(output_file, "w") as f:
            json.dump(metadata_list, f, indent=2)
        print(f"Questions saved to {output_file}")
    except Exception as e:
        print("Some error occurred during making questions", e)


# Function to create MCQs based on questions and answers
def MCQs_Maker(QAs):
    try:
        print("\nStarting Making MCQs .... ")
        # Schema for MCQs
        schema = {
            "properties": {
                "Question_Statement": {
                    "type": "string",
                    "description": "The Question Statement for MCQ",
                },
                "Correct_Option": {
                    "type": "string",
                    "description": "The correct option based on the context.",
                },
                "Incorrect_Option1": {
                    "type": "string",
                    "description": "Incorrect Option",
                },
                "Incorrect_Option2": {
                    "type": "string",
                    "description": "Incorrect Option",
                },
                "Incorrect_Option3": {
                    "type": "string",
                    "description": "Incorrect Option",
                }
            },
            "required": ["Question_Statement", "Correct_Option", "Incorrect_Option1", "Incorrect_Option2",
                         "Incorrect_Option3"],
        }

        original_documents = []
        for qa in QAs:
            original_documents.append(Document(page_content=
                                               f"Context\nQuestion:{qa['question']}\nAnswer:{qa['answer']}\n"
                                               f"You have to make MCQ question and its options using this context\n"
                                               f"Make sure there should be one Correct option and three incorrect options"
                                               ))

        # Initializing ChatOpenAI model
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
        enhanced_documents = document_transformer.transform_documents(original_documents)
        print("\n\nTransformer outputs:")
        print(
            *[json.dumps(d.metadata) for d in enhanced_documents],
            sep="\n\n---------------\n\n"
        )
    except Exception as e:
        print("Some error occurred during making questions", e)


if __name__ == '__main__':
    directory = 'Docs/'
    loaded_documents = load_docs(directory)
    print("Docs loaded: ", len(loaded_documents))
    # print(*loaded_documents, sep="\n----------------------------------\n")

    doc_splits = split_docs(loaded_documents)
    # print("Splits: ",len(doc_splits))
    # for s in doc_splits:
    #     print(s.page_content,"\n\n")

    # store_doc_to_supabase(doc_splits)
    asyncio.run(QA_maker(doc_splits))

    # Making Mcqs from questions
    with open('Questions.json', 'r') as json_file:
        QAs = json.load(json_file)
    MCQs_Maker(QAs[:4])
