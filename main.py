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

# Function to load environment variables
load_dotenv()


# Function to read documents from a directory
def load_docs(pdfs):
    """
    Load PDF documents from a list of file paths.

    Args:
        pdfs (list): List of file paths to PDF documents.

    Returns:
        list: List of loaded documents, each represented as a 'Document' object.
    """
    with st.spinner("Loading Documents... Please wait."):
        print("Starting Documents loading ....")
        try:

            # a list where we will store pages of pdf in document format
            documents=[]
            for pdf in pdfs:
                reader = PdfReader(pdf)

                # to take record of page number will use in metadata
                i = 1
                for page in reader.pages:
                    # making document of pdf pages and appending it in the document list
                    documents.append(Document(page_content=page.extract_text(), metadata={'page': i,"source":pdf.name}))
                    i += 1

            # After loading all the document
            print("Documents Loaded Successfully!")
            st.success("Documents loading completed!")

            # at last return the documents list
            return documents
        except Exception as e:

            # in case of exception
            st.error(f"Documents loading Failed! {e}")
            print("Some Error occurred during Document loading", e)


# Function to split documents into snippets of text
def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split a list of documents into smaller text chunks.

    Args:
        documents (list): List of documents, each represented as a 'Document' object.
        chunk_size (int): Size of each text chunk in tokens.
        chunk_overlap (int): Overlap between consecutive text chunks in tokens.

    Returns:
        list: List of split documents, where each document is a list of text chunks.
    """

    with st.spinner("Splitting Documents... Please wait."):
        try:
            print("Starting Documents splitting ....")
            #Initializing text splitter to split the given documents based on chunk size and chunk overlap
            text_splitter = RecursiveCharacterTextSplitter(
                length_function=len,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap)

            #Calling function to split documents (we will give the documents and it return splits of it)
            split_docs = text_splitter.split_documents(documents)
            print("Documents splitted successfully!")
            st.success("Documents splitting completed!")

            # returning the splitted documents
            return split_docs
        except Exception as e:
            # in case of exception
            st.error(f"Documents splitting Failed! {e}")
            print("Some Error occurred during Document splitting", e)


# Function to store document chunks to Supabase
def store_doc_to_supabase(doc_splits,vector_store):
    """
    Store document splits to a SupaBase database.

    Args:
        doc_splits (list): List of split documents.
        vector_store (SupaBaseVectorStore): A SupaBase vector store object for storing documents.

    Returns:
        None
    """
    with st.spinner("Storing Documents to Database... Please wait."):
        try:
            print("Storing Documents to SupaBase")
            # Take documents and add documents to supabase by calling add_documents function
            vector_store.add_documents(doc_splits)
        except Exception as e:
            pass






# function to store Q/As in supabase
def store_qa_to_supabase(doc_splits,vector_store_for_qa):
    """
    Store question-answer pairs to a SupaBase database.

    Args:
        doc_splits (list): List of split question-answer pairs.
        vector_store_for_qa (SupaBaseVectorStore): A SupaBase vector store object for storing question-answer pairs.

    Returns:
        None
    """
    with st.spinner("Storing Q/As in database... Please wait."):
        try:
            print("Storing Questions Answers to SupaBase")
            # calling function to add Q/As documents
            vector_store_for_qa.add_documents(doc_splits)
        except Exception as e:
            pass

# function to add mcqs documents in supabase
def store_mcqs_to_supabase(doc_splits,vector_store_for_mcqs):
    """
    Store multiple-choice questions (MCQs) to a SupaBase database.

    Args:
        doc_splits (list): List of split MCQs.
        vector_store_for_mcqs (SupaBaseVectorStore): A SupaBase vector store object for storing MCQs.

    Returns:
        None
    """
    with st.spinner("Storing MCQs in database... Please wait."):
        try:
            print("Storing MCQS to SupaBase")
            # calling function to add mcqs documents in supabase
            vector_store_for_mcqs.add_documents(doc_splits)
        except Exception as e:
            pass
        st.success("MCQs have been saved in database")




# Function to create questions and answers using GPT
def QA_maker(api_key,doc_splits,custom_prompt="None",gptmodel="gpt-4"):
    """
    Generate Questions and Answers (Q/As) from a list of split documents.

    Args:
        api_key (str): OpenAI API key for language model generation.
        doc_splits (list): List of split documents.
        custom_prompt (str): Custom prompt for Q/A generation (default is "None").
        gptmodel (str): OpenAI GPT model to use for generation (default is "gpt-4").

    Returns:
        tuple: A tuple containing two lists - (qa_documents, qa_list).
               qa_documents: List of Q/A documents.
               qa_list: List of generated Q/As as dictionaries.
    """
    with st.spinner("Making Q/As... Please wait."):
        try:
            print("Starting making Questions Answer ....")
            # Intilizing schema for transformer to create question answers
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
                # defining required properties of the schema
                "required": ["questions_and_answers"]
            }

            # array in which splits context is placed including prompt
            original_documents = []

            # iteration throught the splits
            for doc_it in doc_splits:
                # appending query documents in the list
                original_documents.append(Document(page_content=
                                                   f"Context\n:```{doc_it.page_content}```\n"
                                                   f"You have to make all the Questions and Answers from this context which follows the given Instructions\n"
                                                   f"(These Questions Answers will be convert in the MCQs in future but not now)\n"
                                                   f"Please strictly follow the instructions given below for making Questions and Answers(which questions you will make and which question you will skip)\n"
                                                   f"INSTRUCTIONS:```"
                                                   f"Quality of Question and Answer should be high do make simple and childish Q/A's\n"
                                                   f"{custom_prompt}```"
                                                   ))

            # initializing llm
            llm = ChatOpenAI(temperature=1,openai_api_key= api_key,model=gptmodel)
            # initializing transformer with metadata tagger
            document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
            # Getting required properties from the orignal documents using transformer
            enhanced_documents = document_transformer.transform_documents(original_documents)


            qa_list=[]
            qa_documents=[]

            for index, doc in enumerate(enhanced_documents):
                # extracting question answer list from the enhanced_documents
                metadata_qas=doc.metadata["questions_and_answers"]
                # appending the extracted list in the qa list
                qa_list.extend(metadata_qas)
                # if a list some question in it
                if len(metadata_qas)>0:
                    # then iterate through the list
                    for qa in metadata_qas:
                        # makes its document and append it in the qa_documents list
                        qa_documents.append(Document(page_content=json.dumps(qa),metadata=original_documents[index].metadata))
            print("Question Answer have been created!")
            st.success("Q/As have been created!")
            # returninn the qa documents and qa list
            return qa_documents,qa_list
        except Exception as e:
            # if some exception occur
            print("Some error occurred during making questions",e)
            st.error(f"Q/As making failed! {e}")
            return [],[]

# Function to create MCQs based on questions and answers
def MCQs_Maker(api_key,QAs,custom_prompt="None",gptmodel="gpt-4"):
    """
    Generate Multiple-Choice Questions (MCQs) from a list of Questions and Answers (Q/As).

    Args:
        api_key (str): OpenAI API key for language model generation.
        QAs (list): List of generated Questions and Answers (Q/As).
        custom_prompt (str): Custom prompt for MCQ generation (default is "None").
        gptmodel (str): OpenAI GPT model to use for generation (default is "gpt-4").

    Returns:
        list: List of generated MCQ documents.
    """
    with st.spinner("Making MCQs... Please wait."):
        try:
            print("Starting Making MCQS")
            # defining schema for mcqs
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
                # defining required field
                "required": ["Question_Statement","Correct_Option","Incorrect_Option1","Incorrect_Option2",
                             "Incorrect_Option3","Explain_Incorrect1","Explain_Incorrect2","Explain_Incorrect3"],
            }
            # array in which qa will placed including prompt to make mcqs form it
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
            # defining llm with high temperature to increase creativity
            llm = ChatOpenAI(temperature=1,openai_api_key=api_key, model=gptmodel)
            # defining transformer with metadata tagger
            document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
            # Getting required properties from the orignal documents using transformer
            enhanced_documents = document_transformer.transform_documents(original_documents)
            MCQ_list=[]
            for doc in enhanced_documents:
                # making mcqs documents which have mcqs in metadata and appending it in the list
                MCQ_list.append(Document(page_content=doc.metadata["Question_Statement"],metadata=doc.metadata))
            st.success("MCQs have been created!")
            # returning required data
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

    # if api_key not given then warn
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

    # default prompts for q/as and mcqs
    custom_prompt1 = "Questions statements should be clear."
    custom_prompt2 = "Give logical explaination of incorrect options."

    # if all the api_keys given
    if api_key and supabase_url and supabase_service_key:
        # Initializing OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # defining supabase client
        supabase: Client = create_client(supabase_url, supabase_service_key)

        # defining vector store to store documents splits
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )

        # defining vector store to Q/As
        vector_store_for_qa = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="question_answers",
            query_name="match_question_answers"
        )

        # defining vector store to store mcqs
        vector_store_for_mcqs = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="mcqs",
            query_name="match_mcqs"
        )
        # checking if there is any data in the session
        if "Mcqs" not in st.session_state and "Related" not in st.session_state and "AskQuestion" not in st.session_state:

            # To input the pdfs
            uploaded_files = st.file_uploader('Choose your .pdf files', type="pdf", accept_multiple_files=True)

            # if some pdfs are uploaded
            if len(uploaded_files) > 0 and "Mcqs" not in st.session_state:
                # custom prompts with default values
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
                    # first load all the pages of pdf in documents from the uploaded pdf
                    loaded_documents = load_docs(uploaded_files)
                    if loaded_documents:
                        # creating splits of the loaded documents
                        doc_splits = split_docs(loaded_documents)
                        if doc_splits:
                            print("Splits: ", len(doc_splits))
                            # storing documents in the supabase by passing vertor store
                            store_doc_to_supabase(doc_splits,vector_store)
                            # funtion to create QAs form the documents splits
                            QAs_docs, QAs_List = QA_maker(api_key,doc_splits, custom_prompt1,gptmodel)
                            if QAs_docs:
                                # storing QAs documents in the supabase
                                store_qa_to_supabase(QAs_docs,vector_store_for_qa)
                                if QAs_List:
                                    # calling function to create mcqs from the questions
                                    mcqs = MCQs_Maker(api_key,QAs_List,custom_prompt2,gptmodel)
                                    # storing mcqs in the supabse
                                    store_mcqs_to_supabase(mcqs,vector_store_for_mcqs)
                                    if "Mcqs" in st.session_state:
                                        # preserving state data in the session
                                        st.session_state["Mcqs"].extend(mcqs)
                                        st.experimental_rerun()
                                    else:
                                        # preserving state data in the session
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

            # input box to take query of user
            question=st.text_input(
                "Ask about the seleted Question",
                value=prompt,
                placeholder="Query",
            )
            answer=""
            if question:
                # defining llm with creative temperature
                llm = ChatOpenAI(model_name=gptmodel,openai_api_key=api_key, temperature=0.7)
                # making qa chain pass passing supabase retriever to find similar splits
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=vector_store.as_retriever()
                )
                # passing prompt with the given question and the similar splits
                result = qa_chain({"query":
                                    f"'''Question:{st.session_state['AskQuestion']['question']}\n"
                                    f"Answer:{st.session_state['AskQuestion']['answer']}'''\n"
                                   f"This question answer was made from the documents\n"
                                   f"On the base of above question and answer retrieve documents form retriever to make a context\n"
                                   f"I want to ask some some query about this created question"
                                   f"My query is '''{question}'''"
                                   f"Please give answer of this query"
                                   })
                # getting answer
                answer=result["result"]
            if answer:
                # printing answer on the screen
                st.write(answer)

            if st.button("Back"):
                # to delete the current state from the session
                del st.session_state["AskQuestion"]
                st.experimental_rerun()

        else:
            # if the current state is of related page
            if "Related" in st.session_state:
                st.write("---")
                st.subheader("Related Questions")
                # making grids
                colms = st.columns((2, 2, 1, 1))
                fields = ["Question", 'Answer', "", ""]
                for col, field_name in zip(colms, fields):
                    # header
                    col.header(field_name)
                st.write("---")
                # extracting QAs from the session
                for x, m in enumerate(st.session_state["Related"]):
                    mcq = m.metadata
                    # showing the question in the grid with two buttons one is related and other is ask question
                    col1, col2, col3, col4 = st.columns((2, 2, 1, 1))
                    col1.write(mcq["Question_Statement"])
                    col2.write(mcq["Correct_Option"])
                    button_type = "Related"
                    button_phold = col3.empty()
                    button_phold1 = col4.empty()
                    # if related button pressed
                    if button_phold.button(button_type,key=x, type="primary"):
                        # finding related question from the supabase and storing them in the session and rerun the app
                        st.session_state["Related"]=vector_store_for_mcqs.similarity_search(m.page_content,5)
                        st.experimental_rerun()
                    if button_phold1.button("Ask Question", key=str(x) + "_", type="primary"):
                        # storing current question in the session and rerun the app
                        st.session_state["AskQuestion"] = {"question": mcq["Question_Statement"],
                                                           "answer": mcq["Correct_Option"]}
                        st.experimental_rerun()
                    st.write("---")
                if st.button("Show All"):
                    # to delete related session
                    del st.session_state["Related"]
                    st.experimental_rerun()
            else:
                if "Mcqs" in st.session_state and "Related" not in st.session_state :
                    st.write("---")
                    st.subheader("All Questions")
                    # making grid
                    colms = st.columns((2, 2,1,1))

                    fields = ["Question", 'Answer',"",""]
                    for col, field_name in zip(colms, fields):
                        # header
                        col.header(field_name)
                    st.write("---")
                    for x, m in enumerate(st.session_state["Mcqs"]):
                        # extracting mcqs from the list
                        mcq=m.metadata
                        col1, col2,col3,col4 = st.columns((2, 2, 1,1))
                        col1.write(mcq["Question_Statement"])
                        col2.write(mcq["Correct_Option"])
                        button_type = "Related"
                        button_phold = col3.empty()
                        button_phold1 = col4.empty()
                        if button_phold.button(button_type,key=x,type="primary"):
                            # finding related question from the supabase and storing them in the session and rerun the app
                            st.session_state["Related"] = vector_store_for_mcqs.similarity_search(m.page_content, 5)
                            st.experimental_rerun()
                        if button_phold1.button("Ask Question",key=str(x)+"_",type="primary"):
                            # storing current question in the session and rerun the app
                            st.session_state["AskQuestion"]={"question":mcq["Question_Statement"],"answer":mcq["Correct_Option"]}
                            st.experimental_rerun()
                        st.write("---")
