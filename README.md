# CeeMyne App

This is a chatbot built using **Python**, **Dash**, and **LangChain**, integrated with **Google Generative AI** and **LangGraph** to enable stateful, context-aware conversations.



## API Key Setup
If you don’t have an API key, you can get one from [Chat | Google AI Studio](https://aistudio.google.com/prompts/new_chat).  
In your app directory, create a file named `.env`, paste the following, and save:
GOOGLE_API_KEY="your-api-key-here"

Installation
git clone https://github.com/Ceenuggets/dash-chatbot.git

cd ceenuggets
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

Run the App
python langchain_gemni_chatbot.py
