# TIET Assistant Chatbot

The **TIET Assistant Chatbot** is an interactive, AI-powered companion application built tailored to help students at TIET. It features a beautiful, clean ChatGPT-like user interface to converse with Large Language Models and provides functionality to track and store student details.

## Features

- **Interactive AI Chat:** Engage in natural conversatons powered by advanced Large Language Models.
- **Modern User Interface:** A clean, responsive design with custom CSS built on top of Streamlit layout.
- **Student Profiling:** Allows users to add and manage their details (Name, Roll No., Branch) conveniently through the chat interface.
- **Secure Configuration:** Employs dotenv for secure API key management along with `.env` files.

## Project Structure

- `streamlit_app.py`: The main frontend application running on Streamlit that handles UI and LLM interactions.
- `LLM_main.py`: Core logic for language model processing and interactions.
- `llm_main_prompt.txt`: System-level prompts configuration for guiding the behavior of the chatbot.
- `.env`: Environment variables configuration file (Should NOT be committed to version control).
- `final.py` & `test.py`: Additional backend integration and test scripts.

## Prerequisites

- [Python](https://www.python.org/) 3.9+ installed on your system.
- Basic knowledge of working with python virtual environments.

## Setup & Installation

**1. Clone the repository**
(Or simply navigate to your project directory):
```bash
git clone <your-repository-url>
cd Software_Eng
```

**2. Create and activate a Virtual Environment**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

**3. Install Dependencies**
Install the required Python packages (such as `streamlit`, `openai`, `python-dotenv`). If you have a `requirements.txt` file, you can install them directly:
```bash
pip install -r requirements.txt
```
Otherwise, manually install them:
```bash
pip install streamlit openai python-dotenv
```

**4. Set up Environment Variables**
Create a `.env` file in the root directory (using the `.env.example` if available) and add your API keys:
```env
HUGGINGFACE_API_KEY=your_actual_api_key_here
```
*(Make sure `.env` is listed in your `.gitignore` to prevent leaking your API keys.)*

**5. Run the Application**
Launch the chatbot using Streamlit:
```bash
streamlit run streamlit_app.py
```

The application will start, and you can view it in your browser, typically at `http://localhost:8501`.

## Usage

- Once the application is running, use the chat input located at the bottom to send queries to the TIET Assistant.
- To manage student information, click on the "**➕ Add Details**" button positioned in the top right, fill in your details (Name, Roll No., Branch) and save.

## Contribution

Feel free to fork the repository and submit pull requests if you have any enhancements, visual updates, or optimizations!
