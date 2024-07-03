# Contract Q&A Bot

This project is a Contract Q&A Bot built using Streamlit, LangChain, and Pinecone. The bot can answer questions about contract documents by leveraging a retrieval-augmented generation (RAG) approach. The document is loaded, processed into chunks, and stored in a Pinecone vector database, from which the bot retrieves relevant information to generate answers.

## Project Structure

project-root/
.
├── app.py
├── data
│   ├── long_contrcacts
│   │   ├── Raptor Contract.docx
│   │   └── Raptor Q&A2.docx
│   └── short_contracts
│       ├── Robinson_Advisory.docx
│       └── Robinson_Q&A.docx
├── evaluation
│   ├── evaluator.py
│   ├── __init__.py
│   └── metrics.py
├── notebooks
│   ├── evaluation.ipynb
│   ├── generator_training.ipynb
│   ├── RAG_evaluation_pipeline.ipynb
│   ├── retriever_training.ipynb
│   └── simple_RAG_pipeline.ipynb
├── README.md
├── requirements.txt
└── utils
    ├── generator.py
    ├── __init__.py
    ├── loader.py
    ├── retriever.py
    └── splitter.py


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


Set up environment variables:
Create a .env file in the root directory and add your OpenAI and Pinecone API keys:

makefile

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_API_ENV=your_pinecone_environment

