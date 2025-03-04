import streamlit as st
import os
from dotenv import load_dotenv
import openai
from PyPDF2 import PdfReader, errors
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ✅ Load API Key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("API Key not found! Please set OPENAI_API_KEY in your .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# ✅ Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except errors.PdfReadError:
            st.error(f"Error: Could not process file {pdf.name}. It may be corrupted.")
        except Exception as e:
            st.error(f"Unexpected error while processing {pdf.name}: {e}")
    return text


# ✅ Split text into chunks for FAISS
def get_text_chunks(text):
    if not text.strip():
        st.error("No text extracted from the PDF. Please check the file.")
        st.stop()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


# ✅ Create & save FAISS vector store
def get_vector_store(text_chunks):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error while creating vector store: {e}")


# ✅ Load GPT-4 conversational model
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "answer is not available in the context."
    
    Context:\n {context}\n
    Question: {question}\n
    Answer:
    """
    model = ChatOpenAI(model="gpt-4", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# ✅ Process user queries
def user_input(user_question):
    try:
        if not os.path.exists("faiss_index"):
            st.error("Vector store not found. Please upload and process a PDF first.")
            return
        
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])

    except Exception as e:
        st.error(f"Error during question processing: {e}")


# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF 💁")

    # User input field
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    # Sidebar for PDF Upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file before processing.")
                return
            
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)


                
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")


if __name__ == "__main__":
    main()
