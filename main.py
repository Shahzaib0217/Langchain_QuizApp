import streamlit as st
import os
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers.openai_functions import create_metadata_tagger
from dotenv import load_dotenv

load_dotenv()


# Function to read documents from a directory
def load_docs(pdfs):
    """
    Load PDF documents from a directory.

    Args:
        directory (str): Directory path containing PDF files.

    Returns:
        list: List of loaded documents.
    """
    with st.spinner("Loading Documents... Please wait."):
        print("Starting Documents loading ....")
        try:
            documents=[]
            for pdf in pdfs:
                reader = PdfReader(pdf)
                i = 1
                for page in reader.pages:
                    documents.append(Document(page_content=page.extract_text(), metadata={'page': i,"source":pdf.name}))
                    i += 1
            print("Documents Loaded Successfully!")
            st.success("Documents loading completed!")
            return documents
        except Exception as e:
            st.error(f"Documents loading Failed! {e}")
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
    #     with st.spinner("Loading... Please wait."):
    #         # Simulate a time-consuming task
    #         time.sleep(3)
    #     # Once the task is complete, clear the loading spinner
    #     st.success("Task completed!")
    with st.spinner("Splitting Documents... Please wait."):
        try:
            print("Starting Documents splitting ....")
            text_splitter = RecursiveCharacterTextSplitter(
                length_function=len,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_documents(documents)
            print("Documents splitted successfully!")
            st.success("Documents splitting completed!")
            return split_docs
        except Exception as e:
            st.error(f"Documents splitting Failed! {e}")
            print("Some Error occurred during Document splitting", e)


# Function to store document chunks to Supabase
def store_doc_to_supabase(doc_splits,vector_store):
    with st.spinner("Storing Documents to Database... Please wait."):
        try:
            print("Storing Documents to SupaBase")
            vector_store.add_documents(doc_splits)
        except Exception as e:
            pass







def store_qa_to_supabase(doc_splits,vector_store_for_qa):
    with st.spinner("Storing Q/As in database... Please wait."):
        try:
            print("Storing Questions Answers to SupaBase")
            vector_store_for_qa.add_documents(doc_splits)
        except Exception as e:
            pass

def store_mcqs_to_supabase(doc_splits,vector_store_for_mcqs):
    with st.spinner("Storing MCQs in database... Please wait."):
        try:
            print("Storing MCQS to SupaBase")
            vector_store_for_mcqs.add_documents(doc_splits)
        except Exception as e:
            pass
        st.success("MCQs have been saved in database")




# Function to create questions and answers using GPT
def QA_maker(doc_splits,custom_prompt="None"):
    with st.spinner("Making Q/As... Please wait."):
        try:
            print("Starting making Questions Answer ....")
            schema = {
                "properties": {
                    "questions_and_answers":{
                        "type": "array",
                        "description": "A List in of JSON of All the Question Answers that can be from given content",
                            "items": {
                                "type": "object",
                                "properties":{
                                    "question":{
                                        "type":"string",
                                        "description":"The statement of the generated Question."
                                    },
                                    "answer":{
                                        "type":"string",
                                        "description":"The answer of the generated question."
                                    }
                                },
                                "required":["question","answer"]
                            },

                        }

                },
                "required": ["questions_and_answers"]
            }


            original_documents = []
            for doc_it in doc_splits:
                original_documents.append(Document(page_content=
                                                   f"Context\n:```{doc_it.page_content}```\n"
                                                   f"You have to make all the Questions and Answers from this context\n"
                                                   f"(These Questions Answers will be convert in the MCQs in future but not now)\n"
                                                   f"Please strictly follow the instructions given below for making Questions and Answers(which questions you will make and which question you will skip)\n"
                                                   f"INSTRUCTIONS:```{custom_prompt}```"
                                                   ))

            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
            document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
            enhanced_documents = document_transformer.transform_documents(original_documents[:4])
            qa_list=[]
            qa_documents=[]
            for index, doc in enumerate(enhanced_documents):
                metadata_qas=doc.metadata["questions_and_answers"]
                qa_list.extend(metadata_qas)
                if len(metadata_qas)>0:
                    for qa in metadata_qas:
                        qa_documents.append(Document(page_content=json.dumps(qa),metadata=original_documents[index].metadata))
            output_file = "Questions.json"
            with open(output_file, "w") as f:
                json.dump(qa_list, f, indent=2)
            print(f"Questions saved to {output_file}")
            print("Question Answer have been created!")
            st.success("Q/As have been created!")
            return qa_documents,qa_list
        except Exception as e:
            print("Some error occurred during making questions",e)
            st.error(f"Q/As making failed! {e}")
            return [],[]

# Function to create MCQs based on questions and answers
def MCQs_Maker(QAs,custom_prompt="None"):
    with st.spinner("Making MCQs... Please wait."):
        try:
            print("Starting Making MCQS")
            schema = {
                "properties": {
                    "Question_Statement": {
                        "type":"string",
                        "description": "The Question Statement for MCQ",
                                    },
                    "Correct_Option": {
                        "type": "string",
                        "description":"The correct option based on the context."

                    },
                    "Incorrect_Option1": {
                        "type": "string",
                        "description": "Incorrect Option"

                    },
                    "Explain_Incorrect1": {
                        "type": "string",
                        "description": "Why Incorrect_Option1 is wrong choice for this question"
                    },
                    "Incorrect_Option2": {
                        "type": "string",
                        "description": "Incorrect Option"

                    },
                    "Explain_Incorrect2": {
                        "type": "string",
                        "description": "Why Incorrect_Option2 is wrong choice for this question."
                    },
                    "Incorrect_Option3": {
                        "type": "string",
                        "description": "Incorrect Option"
                    },
                    "Explain_Incorrect3":{
                        "type":"string",
                        "description":"Why Incorrect_Option3 is wrong choice for this question."
                    }
                },
                "required": ["Question_Statement","Correct_Option","Incorrect_Option1","Incorrect_Option2",
                             "Incorrect_Option3","Explain_Incorrect1","Explain_Incorrect2","Explain_Incorrect3"],
            }
            original_documents=[]
            for qa in QAs:
                original_documents.append(Document(page_content=
                f"Context\nQuestion:{qa['question']}\nAnswer:{qa['answer']}\n"
                f"You have to make MCQ question and its option using this context\n"
                f"Make sure there should be one Correct option and three incorrect options\n"
                f"The incorrect options should not be confusing mean not that much relevant to right answer\n"
                f"Strictly follow the instructions given below for making mcqs\n"
                f"Instructions:```{custom_prompt}```"
                ))
            # Must be an OpenAI model that supports functions
            llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo-0613")
            document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
            enhanced_documents = document_transformer.transform_documents(original_documents)
            MCQ_list=[]
            for doc in enhanced_documents:
                MCQ_list.append(Document(page_content=doc.metadata["Question_Statement"],metadata=doc.metadata))
            st.success("MCQs have been created!")
            return MCQ_list
        except Exception as e:
            st.error(f"MCQs creation Failed! {e}")
            print("Some Error occurred during Making MCQS",e)



if __name__ == '__main__':
    # ---- Streamlit code --------
    # Side Bar
    with st.sidebar:
        st.title("📝 PDF MCQs Generator App")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
            value=os.environ.get("OPENAI_API_KEY", None) or st.session_state.get("OPENAI_API_KEY", ""),
        )

        supabase_url = st.text_input(
            "SUPABASE URL",
            type="password",
            placeholder="Paste your SUPABASE URL here",
            help="Vist, https://supabase.com/",
            value=os.environ.get("SUPABASE_URL", "")
        )

        supabase_service_key = st.text_input(
            "SUPABASE SERVICE KEY",
            type="password",
            placeholder="Paste your SUPABASE SERVICE KEY here",
            value=os.environ.get("SUPABASE_SERVICE_KEY", "")
        )

        st.markdown("---")
        st.markdown(
            "## How to use\n"
            "1. Enter the required keys 🔑\n"  # noqa: E501
            "2. Upload a file 📄\n"
            "3. Enter a prompt to customize quiz 💬\n"
        )

        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"



    #  Main page
    st.title("📝 PDF MCQs Generator App")

    if not (openai_api_key and supabase_url and supabase_service_key):
        st.warning(
            "Enter your API keys in the sidebar."
        )
    else:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["SUPABASE_URL"] = supabase_url
        os.environ["SUPABASE_SERVICE_KEY"] = supabase_service_key

    # File input
    uploaded_files = None
    custom_prompt1 = "Questions statements should be clear."
    custom_prompt2 = "Give logical explaination of incorrect options."

    if os.environ.get("OPENAI_API_KEY", None) and os.environ.get("SUPABASE_URL", None) and os.environ.get("SUPABASE_SERVICE_KEY", None):
        # Initializing OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)

        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )

        vector_store_for_qa = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="question_answers",
            query_name="match_question_answers"
        )

        vector_store_for_mcqs = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="mcqs",
            query_name="match_mcqs"
        )
        uploaded_files = st.file_uploader('Choose your .pdf files', type="pdf", accept_multiple_files=True)
        if len(uploaded_files) > 0:
            custom_prompt1 = st.text_input(
                "Enter a prompt for Making Question/Answers",
                value=custom_prompt1,
                placeholder="What Kind of QA's you want to make",
            )
            custom_prompt2 = st.text_input(
                "Enter a prompt for Making MCQs",
                value=custom_prompt2,
                placeholder="What Kind of MCQs you want to make",
            )
            if st.button("Start Making Quiz"):
                loaded_documents = load_docs(uploaded_files)
                if loaded_documents:
                    doc_splits = split_docs(loaded_documents)
                    if doc_splits:
                        print("Splits: ", len(doc_splits))
                        store_doc_to_supabase(doc_splits,vector_store)
                        QAs_docs, QAs_List = QA_maker(doc_splits, custom_prompt1)
                        if QAs_docs:
                            store_qa_to_supabase(QAs_docs,vector_store_for_qa)
                            if QAs_List:
                                mcqs = MCQs_Maker(QAs_List[:4],custom_prompt2)
                                store_mcqs_to_supabase(mcqs,vector_store_for_mcqs)