import streamlit as st
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_transformers.openai_functions import create_metadata_tagger
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

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
def QA_maker(api_key,doc_splits,custom_prompt="None",gptmodel="gpt-4"):
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
                                                   f"You have to make all the Questions and Answers from this context which follows the given Instructions\n"
                                                   f"(These Questions Answers will be convert in the MCQs in future but not now)\n"
                                                   f"Please strictly follow the instructions given below for making Questions and Answers(which questions you will make and which question you will skip)\n"
                                                   f"INSTRUCTIONS:```"
                                                   f"Quality of Question and Answer should be high do make simple and childish Q/A's\n"
                                                   f"{custom_prompt}```"
                                                   ))

            llm = ChatOpenAI(temperature=1,openai_api_key= api_key,model=gptmodel)
            document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
            enhanced_documents = document_transformer.transform_documents(original_documents)
            qa_list=[]
            qa_documents=[]
            for index, doc in enumerate(enhanced_documents):
                metadata_qas=doc.metadata["questions_and_answers"]
                qa_list.extend(metadata_qas)
                if len(metadata_qas)>0:
                    for qa in metadata_qas:
                        qa_documents.append(Document(page_content=json.dumps(qa),metadata=original_documents[index].metadata))
            print("Question Answer have been created!")
            st.success("Q/As have been created!")
            return qa_documents,qa_list
        except Exception as e:
            print("Some error occurred during making questions",e)
            st.error(f"Q/As making failed! {e}")
            return [],[]

# Function to create MCQs based on questions and answers
def MCQs_Maker(api_key,QAs,custom_prompt="None",gptmodel="gpt-4"):
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
            llm = ChatOpenAI(temperature=1,openai_api_key=api_key, model=gptmodel)
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

    with st.sidebar:
        st.title("📝 PDF MCQs Generator App")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",
            value=""
        )
        gptmodel=st.selectbox("Select a model:", ["gpt-4", "gpt-3.5-turbo-0613"])
        supabase_url = st.text_input(
            "SUPABASE URL",
            type="password",
            placeholder="Paste your SUPABASE URL here",
            help="Vist, https://supabase.com/",
            value=""
        )

        supabase_service_key = st.text_input(
            "SUPABASE SERVICE KEY",
            type="password",
            placeholder="Paste your SUPABASE SERVICE KEY here",
            value=""
        )

        st.markdown("---")
        st.markdown(
            "## How to use\n"
            "1. Enter the required keys 🔑\n"  # noqa: E501
            "2. Upload a file 📄\n"
            "3. Enter a prompt to customize quiz 💬\n"
        )

        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"



    #  Main page
    st.title("📝 PDF MCQs Generator App")

    if not (api_key and supabase_url and supabase_service_key):
        st.warning(
            "Enter your API keys in the sidebar."
        )
    # else:
    #     os.environ["OPENAI_API_KEY"] = openai_api_key
    #     os.environ["SUPABASE_URL"] = supabase_url
    #     os.environ["SUPABASE_SERVICE_KEY"] = supabase_service_key

    # File input
    uploaded_files = None
    custom_prompt1 = "Questions statements should be clear."
    custom_prompt2 = "Give logical explaination of incorrect options."

    if api_key and supabase_url and supabase_service_key:
        # Initializing OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        supabase: Client = create_client(supabase_url, supabase_service_key)

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
        if "Mcqs" not in st.session_state and "Related" not in st.session_state and "AskQuestion" not in st.session_state:
            uploaded_files = st.file_uploader('Choose your .pdf files', type="pdf", accept_multiple_files=True)
            if len(uploaded_files) > 0 and "Mcqs" not in st.session_state:
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
                            QAs_docs, QAs_List = QA_maker(api_key,doc_splits, custom_prompt1,gptmodel)
                            if QAs_docs:
                                store_qa_to_supabase(QAs_docs,vector_store_for_qa)
                                if QAs_List:
                                    mcqs = MCQs_Maker(api_key,QAs_List,custom_prompt2,gptmodel)
                                    store_mcqs_to_supabase(mcqs,vector_store_for_mcqs)
                                    if "Mcqs" in st.session_state:
                                        st.session_state["Mcqs"].extend(mcqs)
                                        st.experimental_rerun()
                                    else:
                                        st.session_state["Mcqs"] = mcqs
                                        st.experimental_rerun()
        if "AskQuestion" in st.session_state:
            st.write("---")
            st.header("Ask Questions")
            st.write("---")
            st.subheader("Question")
            st.write(st.session_state["AskQuestion"]["question"])
            st.write("---")
            st.subheader("Answer")
            st.write(st.session_state["AskQuestion"]["answer"])
            st.write("---")
            st.subheader("Query")
            prompt=""

            question=st.text_input(
                "Ask about the seleted Question",
                value=prompt,
                placeholder="Query",
            )
            answer=""
            if question:
                llm = ChatOpenAI(model_name=gptmodel,openai_api_key=api_key, temperature=0.7)
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=vector_store.as_retriever()
                )
                result = qa_chain({"query":
                                    f"'''Question:{st.session_state['AskQuestion']['question']}\n"
                                    f"Answer:{st.session_state['AskQuestion']['answer']}'''\n"
                                   f"This question answer was made from the documents\n"
                                   f"On the base of above question and answer retrieve documents form retriever to make a context\n"
                                   f"I want to ask some some query about this created question"
                                   f"My query is '''{question}'''"
                                   f"Please give answer of this query"
                                   })
                answer=result["result"]
            if answer:
                st.write(answer)

            if st.button("Back"):
                del st.session_state["AskQuestion"]
                st.experimental_rerun()

        else:
            if "Related" in st.session_state:
                st.write("---")
                st.subheader("Related Questions")
                colms = st.columns((2, 2, 1, 1))
                fields = ["Question", 'Answer', "", ""]
                for col, field_name in zip(colms, fields):
                    # header
                    col.header(field_name)
                st.write("---")

                for x, m in enumerate(st.session_state["Related"]):
                    mcq = m.metadata
                    col1, col2, col3, col4 = st.columns((2, 2, 1, 1))
                    col1.write(mcq["Question_Statement"])
                    col2.write(mcq["Correct_Option"])
                    button_type = "Related"
                    button_phold = col3.empty()
                    button_phold1 = col4.empty()
                    if button_phold.button(button_type,key=x, type="primary"):
                        st.session_state["Related"]=vector_store_for_mcqs.similarity_search(m.page_content,5)
                        st.experimental_rerun()
                    if button_phold1.button("Ask Question", key=str(x) + "_", type="primary"):
                        st.session_state["AskQuestion"] = {"question": mcq["Question_Statement"],
                                                           "answer": mcq["Correct_Option"]}
                        st.experimental_rerun()
                    st.write("---")
                if st.button("Show All"):
                    del st.session_state["Related"]
                    st.experimental_rerun()
            else:
                if "Mcqs" in st.session_state and "Related" not in st.session_state :
                    st.write("---")
                    st.subheader("All Questions")
                    colms = st.columns((2, 2,1,1))
                    fields = ["Question", 'Answer',"",""]
                    for col, field_name in zip(colms, fields):
                        # header
                        col.header(field_name)
                    st.write("---")
                    for x, m in enumerate(st.session_state["Mcqs"]):
                        mcq=m.metadata
                        col1, col2,col3,col4 = st.columns((2, 2, 1,1))
                        col1.write(mcq["Question_Statement"])
                        col2.write(mcq["Correct_Option"])
                        button_type = "Related"
                        button_phold = col3.empty()
                        button_phold1 = col4.empty()
                        if button_phold.button(button_type,key=x,type="primary"):
                            st.session_state["Related"] = vector_store_for_mcqs.similarity_search(m.page_content, 5)
                            st.experimental_rerun()
                        if button_phold1.button("Ask Question",key=str(x)+"_",type="primary"):
                            st.session_state["AskQuestion"]={"question":mcq["Question_Statement"],"answer":mcq["Correct_Option"]}
                            st.experimental_rerun()
                        st.write("---")
