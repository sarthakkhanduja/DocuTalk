# import streamlit as st
# import tempfile
# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders import PyPDFLoader
# from dotenv import load_dotenv
# from langchain_openai import OpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS

# # Function to load and process file content using LangChain loaders
# def load_file(uploaded_file):
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(uploaded_file.getvalue())
#         temp_file_path = temp_file.name
    
#     # Process the file using appropriate LangChain loader
#     if uploaded_file.type == "application/pdf":
#         loader = PyPDFLoader(temp_file_path)
#         docs = loader.load()
#     elif uploaded_file.type == 'text/csv':
#         loader = CSVLoader(temp_file_path)
#         docs = loader.load()
#     else:
#         st.write(f"Cannot process the contents of {uploaded_file.name} (unsupported file type)")
#         return []
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
#     documents = text_splitter.split_documents(docs)
    
#     return documents

# # Load the ENV variables
# load_dotenv()

# # Check and load the API key
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# # Set the layout to wide mode
# st.set_page_config(layout="wide", page_title="DocuTalk - Say hello to your files")

# # Sidebar for file/folder upload and API Key input
# with st.sidebar:
#     with st.form(key="file_uploader"):
#         st.header("Upload Files")
#         uploaded_files = st.file_uploader("Choose files", type=None, accept_multiple_files=True)
#         upload_files_btn = st.form_submit_button("Upload files")

#     # Adding space between containers
#     st.markdown("<br><br>", unsafe_allow_html=True)

#     with st.form(key="openai_api"):
#         st.header("OpenAI API Key")
#         openai_api_key = st.text_input("OpenAI API Key", type="password")
#         api_key_submitted = st.form_submit_button("Submit API Key")

# # Main content area with a central title
# st.markdown(
#     """
#     <div style='text-align: center; margin-top: 50px;'>
#         <h1 style='font-size: 72px;'>DocuTalk</h1>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# # Ensure the API key is set
# if api_key_submitted:
#     os.environ["OPENAI_API_KEY"] = openai_api_key
# else:
#     st.write("No Open AI API key has been provided. You need to provide this for the application to work.")


# # Declaring the embeddings and LLM
# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)

# # Creating the prompt
# prompt = ChatPromptTemplate.from_template("""
#     You are my Research Assistant. You will be given context via embeddings regarding a lot of research papers in my vector database.
#     Answer the following question based only on the provided context. 
#     Provide a detailed and comprehensive answer. Ensure your response is thorough and informative.
#     <context>
#     {context}
#     </context>
#     Question: {input}""")

# # Creating the document chain
# document_chain = create_stuff_documents_chain(llm, prompt)

# # Display the contents of the uploaded files
# if upload_files_btn:
#     all_documents = []

#     for uploaded_file in uploaded_files:
#         documents = load_file(uploaded_file)
#         if documents:
#             all_documents.extend(documents)
    
#     if all_documents:
#         # Initialize the FAISS vector store
#         vectorstore = FAISS.from_documents(all_documents, embeddings)
        
#         # Initializing the retriever
#         retriever = vectorstore.as_retriever()

#         # Creating the retriever chain
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         # Provide an input box for the user to ask questions
#         user_query = st.text_input("Ask a question about your documents:")

#         if user_query:
#             response = retrieval_chain.invoke({"input": user_query})
#             st.write(response["answer"])
#     else:
#         st.write("No documents to process.")
# else:
#     st.write("You have not uploaded any files yet.")


import streamlit as st
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Function to load and process file content using LangChain loaders
def load_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    # Process the file using appropriate LangChain loader
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
    elif uploaded_file.type == 'text/csv':
        loader = CSVLoader(temp_file_path)
        docs = loader.load()
    else:
        st.write(f"Cannot process the contents of {uploaded_file.name} (unsupported file type)")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)
    
    return documents

# Load the ENV variables
load_dotenv()

# Set the layout to wide mode
st.set_page_config(layout="wide", page_title="DocuTalk - Say hello to your files")

# Initialize session state variables
if 'api_key_submitted' not in st.session_state:
    st.session_state['api_key_submitted'] = False
if 'files_uploaded' not in st.session_state:
    st.session_state['files_uploaded'] = False
if 'all_documents' not in st.session_state:
    st.session_state['all_documents'] = []

# Sidebar for file/folder upload and API Key input
with st.sidebar:
    st.header("OpenAI API Key")
    with st.form(key="openai_api"):
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        api_key_submitted = st.form_submit_button("Submit API Key")
        if api_key_submitted and openai_api_key:
            st.session_state['api_key_submitted'] = True
            st.success("OpenAI API key submitted successfully.")
        elif api_key_submitted and not openai_api_key:
            st.error("No OpenAI API key has been provided. You need to provide this for the application to work.")

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.header("Upload Files")
    with st.form(key="file_uploader"):
        uploaded_files = st.file_uploader("Choose files", type=None, accept_multiple_files=True)
        upload_files_btn = st.form_submit_button("Upload files")
        if upload_files_btn and uploaded_files:
            st.session_state['files_uploaded'] = True
            st.success("Files uploaded successfully.")
            all_documents = []
            for uploaded_file in uploaded_files:
                documents = load_file(uploaded_file)
                if documents:
                    all_documents.extend(documents)
            st.session_state['all_documents'] = all_documents
        elif upload_files_btn and not uploaded_files:
            st.error("You have not uploaded any files yet.")

# Main content area with a central title
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px;'>
        <h1 style='font-size: 72px;'>DocuTalk</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Ensure the API key is set
if st.session_state['api_key_submitted']:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Declaring the embeddings and LLM
if st.session_state['api_key_submitted'] and st.session_state['files_uploaded']:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)

    # Creating the prompt
    prompt = ChatPromptTemplate.from_template("""
        You are my Research Assistant. You will be given context via embeddings regarding a lot of research papers in my vector database.
        Answer the following question based only on the provided context. 
        Provide a detailed and comprehensive answer. Ensure your response is thorough and informative.
        <context>
        {context}
        </context>
        Question: {input}""")

    # Creating the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Check if there are documents to process
    if st.session_state['all_documents']:
        # Initialize the FAISS vector store
        vectorstore = FAISS.from_documents(st.session_state['all_documents'], embeddings)
        
        # Initializing the retriever
        retriever = vectorstore.as_retriever()

        # Creating the retriever chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Provide an input box for the user to ask questions
        user_query = st.text_input("Ask a question about your documents:")
        query_submitted = st.button("Submit Question")

        if query_submitted:
            if user_query:
                response = retrieval_chain.invoke({"input": user_query})
                # Add a divider
                st.markdown("<hr>", unsafe_allow_html=True)
                
                st.write(response["answer"])
            else:
                st.error("Please enter a question to get a response.")
    else:
        st.error("No documents to process.")
elif st.session_state['api_key_submitted']:
    st.error("Please upload your files.")
elif st.session_state['files_uploaded']:
    st.error("Please provide your OpenAI API key.")
