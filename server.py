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
        # print("PDF encountered")
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

# Check and load the API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Set the layout to wide mode
st.set_page_config(layout="wide")

# Sidebar for file/folder upload and API Key input
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader("Choose files", type=None, accept_multiple_files=True)
    openai_api_key = st.text_input("OpenAI API Key", type="password")

# Ensure the API key is set
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Declaring the embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = OpenAI(model="gpt-3.5-turbo-instruct")

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


# Main content area with a central title
st.markdown(
    """
    <div style='text-align: center; margin-top: 50px;'>
        <h1 style='font-size: 72px;'>DocuTalk</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display the contents of the uploaded files
if uploaded_files:
    all_documents = []
    
    for uploaded_file in uploaded_files:
        documents = load_file(uploaded_file)
        # print(len(documents))
        if documents:
            all_documents.extend(documents)
            # print(len(all_documents))
    
    if all_documents:
        # Initialize the FAISS vector store
        vectorstore = FAISS.from_documents(all_documents, embeddings)
        
        # Initializing the retriever
        retriever = vectorstore.as_retriever()

        # Creating the retriever chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Provide an input box for the user to ask questions
        user_query = st.text_input("Ask a question about your documents:")

        if user_query:
            response = retrieval_chain.invoke({"input": user_query})
            st.write(response["answer"])
    else:
        st.write("No documents to process.")
else:
    st.write("You have not uploaded any files yet.")
