import streamlit as st
from PyPDF2 import PdfReader, errors
import openai
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

# Function to extract text from PDF
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
            st.error(f"Error: Could not process file {pdf.name}. It may be corrupted or invalid.")
        except Exception as e:
            st.error(f"Unexpected error while processing {pdf.name}: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    if not text.strip():
        st.error("No text to split.")
        st.stop()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create and store vector embeddings
def get_vector_store(text_chunks):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_texts(text_chunks, embeddings=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error while creating vector store: {e}")

        import re

def get_extraction_prompt(content: str):
    """
    Determines the type of assessment (case study or written assessment) and returns the structured extraction prompt.
    """

    is_case_study = bool(re.search(r'\bcase\s*study\b', content, re.IGNORECASE)) or "case study context" in content.lower()

    assessment_type = "case_study" if is_case_study else "written_assessment"

    if is_case_study:
        return f"""
        You are an assistant that extracts structured information from text. 
        Extract the case study context only once if the document is for a case study assessment. 
        Then extract all the questions and suggested answers in the specified format.

        Extract the case study context only once if the document is for a case study assessment. Then extract all the questions and suggested answers, and format them as a JSON array. Each item should have the following structure:
        -'Total Duration,Duration,time-flaot-(Give only number , do not add any unit name with the number example:- 30 ,60,120 etc)'
        -'instructions to Candidate'
        - 'question_number'
        - 'question'
        - 'question_instruction (These instruction come after question number. It is present in ())'
        - 'suggested_answer' (give all points which are present in the suggested answer, as an array of points, ensuring all points of a question in the suggested answer are fully captured, including multi-paragraph content if applicable)
        -'comparison_count -flaot- (This count will come after the Suggested answer Handing if it is present. If comparison_count is not present, then try to find it from the question. If it is not present in both places, then return null)'
        -'comparison_instruction (If not present, then send null. This instruction will come after the Suggested answer Handing (any 1, any 2, any 3, any one, any two, any three). If not present, then send null)'
        
        - 'case_study_context' (if applicable)

        **Important:**  
        - Ensure that the **suggested_answer** field contains all the suggested points, irrespective of comparison_count.  
        - Ensure that the **suggested_answer** field contains the full, detailed answer.  
        - Extract all relevant answer points, including any subpoints or explanations.  
        - If the answer is split across multiple paragraphs or bullet points, include them all in the array.  
        - Do not summarize or truncate answers; keep the complete answer structure.  
        
        Example output format:
        {{
            "assessment_type": "case_study",
            "duration":<Duration>,
            "assessment_instruction":[<instructions to Candidate_point_1>, <instructions to Candidate_point_2>, ...],
            "case_study_context": "<case study content>",
            "questions_and_answers": [
                {{
                    "question_number": <question_number>,
                    "question": "<question_text>",
                    "question_instruction": "<question_instruction>",
                    "suggested_answer": [<answer_point_1>, <answer_point_2>, ...],
                    "comparison_count":<comparison_count>,
                    "comparison_instruction":<comparison_instruction>
                }}
            ]
        }}

        Document content:
        {content}
        """
    else:
        return f"""
        You are an assistant that extracts structured information from text. 
        Extract all the questions and suggested answers in the specified format.

        Extract all the questions and suggested answers, and format them as a JSON array. Each item should have the following structure:
        -'Total Duration,Duration,time -flaot-(Give only number , do not add any unit name with the number example:- 30 ,60,120 etc)'
        -'instructions to Candidate'
        - 'question_number'
        - 'question'
        - 'question_instruction (These instruction come after question number. It is present in ())'
        -'suggested_answer' (give all points which are present in the suggested answer, ensuring all points of a question in the suggested answer are fully captured, as an array of points, including multi-paragraph content if applicable)
        -'comparison_count -flaot- (This count will come after the Suggested answer Handing if it is present. If comparison_count is not present, then try to find it from the question. If it is not present in both places, then return null)'
        -'comparison_instruction (If not present, then send null. This instruction will come after the Suggested answer Handing (any 1, any 2, any 3, any one, any two, any three). If not present, then send null)'
        
        **Important:**  
        - Ensure that the **suggested_answer** field contains all the suggested points, irrespective of comparison_count.  
        - Ensure that the **suggested_answer** field contains the full, detailed answer.  
        - Extract all relevant answer points, including any subpoints or explanations.  
        - If the answer is split across multiple paragraphs or bullet points, include them all in the array.  
        - Do not summarize or truncate answers; keep the complete answer structure.  
        
        Example output format:
        {{
            "assessment_type": "written_assessment",
            "duration":<Duration>,
            "assessment_instruction":[<instructions to Candidate_point_1>, <instructions to Candidate_point_2>, ...],
            "case_study_context": "",
            "questions_and_answers": [
                {{
                    "question_number": <question_number>,
                    "question": "<question_text>",
                    "question_instruction": "<question_instruction>",
                    "suggested_answer": [<answer_point_1>, <answer_point_2>, ...],
                    "comparison_count":<comparison_count>,
                    "comparison_instruction":<comparison_instruction>
                }}
            ]
        }}

        Document content:
        {content}
        """



# Function to call OpenAI API for structured extraction
def extract_structured_data(chunks):
    results = []
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    for idx, chunk in enumerate(chunks):
        st.write(f"Processing chunk {idx + 1}/{len(chunks)}...")
        
        prompt = get_extraction_prompt(chunk)
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3
            )
            results.append(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            return None

    return results

# Main function to run the app
def main():
    st.set_page_config(page_title="Chat PDF Extractor")
    st.header("Chat with PDF and Extract Structured Data 💁")

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    if st.button("Process PDF"):
        if pdf_docs:
            with st.spinner("Extracting text from PDF..."):
                raw_text = get_pdf_text(pdf_docs)
            
            if raw_text:
                st.write(f"Total words in document: {len(raw_text.split())}")
                
                with st.spinner("Splitting into chunks..."):
                    text_chunks = get_text_chunks(raw_text)
                
                st.write(f"Processing {len(text_chunks)} chunks...")
                
                with st.spinner("Extracting structured data..."):
                    structured_output = extract_structured_data(text_chunks)
                    
                    if structured_output:
                        st.json(structured_output)
                        st.success("Extraction Completed Successfully!")
        else:
            st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
