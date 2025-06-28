Thanks, Priya! Based on your second project (the **Streamlit-based PDF Q\&A Extractor using OpenAI**), here's a clean, professional `README.md` file designed for GitHub:

---

```markdown
# 📄 PDF Q&A Extractor - AI-powered Assessment Parser

This project is a **Streamlit** application that reads uploaded **PDFs**, extracts **structured question-answer data**, and formats it for automated LLM-based evaluations. It's ideal for educational institutions, assessment developers, or AI grading tools.

---

## 🚀 Features

- 📚 Upload one or more **PDFs**
- 🤖 Automatically detects **Case Study** vs **Written Assessment**
- 🧠 Uses **OpenAI GPT-4 Turbo** for structured extraction
- 📦 Extracts:
  - Total Duration
  - Instructions to Candidate
  - Case Study Context (if present)
  - Questions & Suggested Answers
  - Comparison count & instruction
- 🧩 Saves **FAISS vector index** for potential retrieval tasks

---

## 📁 Project Structure

```

.
├── faiss\_index/                 # Stored vector index
├── main.py                      # Streamlit app entry
├── app.py                       # Optional alt entry (if any)
├── test.py                      # Optional testing file
├── requirements.txt             # Python dependencies
├── .env                         # OpenAI API key
└── README.md                    # You're here!

````

---

## ⚙️ Setup Instructions

### ✅ Prerequisites
- Python 3.10 or later
- OpenAI API key

### 🛠 Installation

```bash
git clone https://github.com/Priya-Rathor/AI_INNOV_JAM.git
cd AI_INNOV_JAM
pip install -r requirements.txt
````

### 🔐 Environment Setup

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🧪 How to Use

### 1. Run the App

```bash
streamlit run main.py
```

### 2. In the UI:

* Upload PDF(s) containing assessment content.
* Click **“Process PDF”**.
* View structured JSON output with all extracted questions, suggested answers, and metadata.

---

## 🧠 Under the Hood

### ✨ Core Components

* **`get_pdf_text()`**: Extracts clean text from PDF pages
* **`get_text_chunks()`**: Splits long text into chunks for LLM input
* **`get_extraction_prompt()`**: Dynamically generates prompts based on assessment type (case study or written)
* **`extract_structured_data()`**: Sends prompts to GPT-4-Turbo to get clean structured JSON

### ⚙️ Uses:

* **LangChain** for chunking & FAISS indexing
* **OpenAI GPT-4-Turbo** for extraction

---

## 📄 Example Output

```json
{
  "assessment_type": "case_study",
  "duration": 60,
  "assessment_instruction": ["Read all questions carefully.", "Answer in your own words."],
  "case_study_context": "ABC Corp is expanding...",
  "questions_and_answers": [
    {
      "question_number": 1,
      "question": "What challenges did ABC Corp face?",
      "question_instruction": "(10 marks)",
      "suggested_answer": [
        "High operational cost",
        "Regulatory issues"
      ],
      "comparison_count": 2,
      "comparison_instruction": "Any 2"
    }
  ]
}
```

---

## 📌 To-Do

* [ ] Add CSV/Excel export of extracted data
* [ ] Add API wrapper to send to evaluation system
* [ ] Enable persistent FAISS retrieval + RAG-style chat

---

## 👩‍💻 Author

**Priya Rathor**
🔗 [GitHub](https://github.com/Priya-Rathor)

---

## 📚 Tech Stack

* Streamlit
* PyPDF2
* LangChain
* FAISS
* OpenAI (GPT-4 Turbo)
* Python 3.11+

---

## 🏁 License

This project is under the **MIT License**.

```

