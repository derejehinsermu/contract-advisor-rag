# Contract Q&A Bot

This project is a Contract Q&A Bot built using Streamlit, LangChain, and Pinecone. The bot can answer questions about contract documents by leveraging a retrieval-augmented generation (RAG) approach. The document is loaded, processed into chunks, and stored in a Pinecone vector database, from which the bot retrieves relevant information to generate answers.

## Project Structure

```
project-root/
├.
├── app.py
├── data
│   ├── long_contract
│   │   ├── Raptor_Contract.docx
│   │   └── Raptor Q&A2.docx
│   └── short_contracts
│       ├── Robinson_Advisory.docx
│       └── Robinson_Q&A.docx
├── notebooks
│   ├── evaluation_dataset.json
│   ├── RAG_evaluation_pipeline.ipynb
│   ├── simple_RAG_pipeline.ipynb
│   ├── Visualize_qa_evaluate_advance_rag_pipeline.ipynb
│   ├── Visualize_qa_evaluation_meduim_rag_pipeline.ipynb
│   └── Visualize_simple_rag_pipeline.ipynb
├── README.md
├── requirements.txt
├── scripts
│   ├── advance_rag_pipeline
│   │   ├── evaluation2.py
│   │   ├── evaluator.py
│   │   ├── generator.py
│   │   ├── retrieval.py
│   │   └── visualize.py
│   ├── load_split_data
│   │   └── load_split.py
│   ├── medium_rag__pipeline
│   │   ├── evaluator.py
│   │   ├── generator.py
│   │   └── retrieval.py
│   ├── simple_pipeline
│   │   ├── evaluator.py
│   │   ├── generator.py
│   │   └── retrieval.py
│   ├── visualize_advance_rag_pipeline
│   │   ├── generator.py
│   │   └── retrieval.py
│   ├── visualize_medium_rag_pipe_line
│   │   ├── generator.py
│   │   └── retrieval.py
│   └── visualize_simple_rag_pipe_line
│       ├── generator.py
│       └── retrieval.py
└── utils
    ├── generator.py
    ├── __init__.py
    ├── loader.py
    ├── retriever.py
    └── splitter.py
```
## Getting Started

### Prerequisites

### Installation

1. **Clone the repository:**
   ```bash
    git@github.com/derejehinsermu/LizzyAI_Contract_Advisor_Bot.git

2. **Create and Activate a Virtual Environment**
    Navigate to the root directory of the project and create a virtual environment named 'venv', then activate it:
    ```sh
    cd LizzyAI_Contract_Advisor_Bot.git
    python -m venv venv  | virtualenv venv
    source venv/bin/activate

3 **Install Requirements**
    While inside the virtual environment, install the project requirements:
    
    ```sh    
    pip install -r requirements.txt


4 **Set up environment variables:**
Create a .env file in the root directory and add your OpenAI and Pinecone API keys:


