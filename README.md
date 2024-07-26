# Contract Q&A Retrieval-Augmented Generation (RAG) System

## Overview

The Contract Q&A Retrieval-Augmented Generation (RAG) system is designed to provide users with an interactive way to inquire about contract documents. Inspired by Lizzy AI, this project aims to develop a system that efficiently answers questions about contracts using advanced natural language processing and retrieval-augmented generation techniques.

## Technologies Used

- **LangChain**: A framework for building applications with large language models (LLMs).
- **OpenAI API**: Provides access to GPT-3.5-Turbo for language processing.
- **GPT-4**: Utilized for advanced language understanding and generation.
- **MongoDB Atlas**: A vector database for storing and retrieving contract data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project builds a Contract Q&A Bot that allows users to interact with and ask questions about contract documents. It processes the contracts by breaking them into chunks and storing them in a MongoDB Atlas vector database. The bot then retrieves relevant information from this database to provide accurate answers, leveraging technologies such as LangChain, Chroma, and GPT models.

## Features

- **Interactive Q&A**: Users can ask detailed questions about contract documents.
- **Efficient Data Retrieval**: Utilizes MongoDB Atlas for fast data storage and retrieval.
- **Advanced Language Processing**: Leverages GPT-3.5-Turbo and GPT-4 for accurate responses.
- **User-Friendly Interface**: Built with Streamlit for an easy-to-use web interface.

## Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/derejehinsermu/contract-advisor-rag.git

2. **Create and Activate a Virtual Environment**
   
    Navigate to the root directory of the project and create a virtual environment named 'venv', then activate it:
    ```sh
    cd contract-advisor-rag.git
    python -m venv venv  | virtualenv venv
    source venv/bin/activate

4. **Install Requirements**
    While inside the virtual environment, install the project requirements:
    
    pip install -r requirements.txt

## Usage

To run the Contract Q&A Bot:

1. **Start the Streamlit App:**
   ```bash
   cd app
   streamlit run stream_app.py


2. **Open your browser and navigate to http://localhost:8501 to interact with the bot.**
   
![image](https://github.com/user-attachments/assets/3f4b9499-03a6-4549-b0ff-5194414ff963)

## Contributing

Contributions are welcome! 

3. For any questions or support, please contact derejehinsermu2@gmail.com.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
