from dotenv import load_dotenv
import os
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import openai
from langchain_community.tools import Tool, DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Set your Azure OpenAI API key and endpoint
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_MODEL_DEFAULT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
TEXT_EMBEDDING = os.getenv("TEXT_EMBEDDING")

# Configure Azure OpenAI Service API
# openai.api_type = "azure"
# openai.api_version = OPENAI_API_VERSION
# openai.api_base = AZURE_OPENAI_ENDPOINT
# openai.api_key = AZURE_OPENAI_API_KEY


def load_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        texts = []
        pages = []
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()
            if text:
                texts.append(text)
                pages.append(f"BonBon FAQ.pdf (page {page_num + 1})")
    return texts, pages


def split_documents_with_pages(texts, pages, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators="\n")
    chunked_docs = []
    chunked_pages = []

    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        chunked_docs.extend(chunks)
        chunked_pages.extend([pages[i]] * len(chunks))

    return chunked_docs, chunked_pages


def initialize_embeddings():
    embeddings = AzureOpenAIEmbeddings(
        model=TEXT_EMBEDDING
    )
    return embeddings


def store_embeddings_in_chroma(chunked_docs, pages, embeddings, db_name="bonbon_faq"):
    metadatas = [{"source": source} for source in pages]
    vector_store = Chroma.from_texts(chunked_docs, embeddings, collection_name=db_name, metadatas=metadatas)
    return vector_store


def index_pdf_to_chroma(pdf_path):
    texts, pages = load_pdf(pdf_path)
    chunked_docs, chunked_pages = split_documents_with_pages(texts, pages)
    embeddings = initialize_embeddings()
    vector_store = store_embeddings_in_chroma(chunked_docs, chunked_pages, embeddings)
    return vector_store


def create_tools(vector_store):
    internet_search_tool = Tool(
        name="Internet Search",
        func=DuckDuckGoSearchRun().run,
        description="Tìm kiếm thông tin trên internet bằng DuckDuckGo."
    )

    knowledge_base_tool = Tool(
        name="Knowledge Base Search",
        func=vector_store.as_retriever().get_relevant_documents,
        description="Tra cứu thông tin từ tài liệu BonBon FAQ."
    )

    return [internet_search_tool, knowledge_base_tool]


def create_agent_with_tools(vector_store):
    client = AzureChatOpenAI(deployment_name=DEPLOYMENT_NAME, temperature=0, openai_api_version=OPENAI_API_VERSION)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=2)
    tools = create_tools(vector_store)
    agent = initialize_agent(
        tools=tools,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        llm=client,
        memory=memory,
        verbose=True
    )

    return agent


def chatbot_conversation(agent, chat_history=None):
    if chat_history is None:
        chat_history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("End conversation.")
            break

        result = agent.invoke(user_input)

        chat_history.append({"user": user_input, "bot": result})


vector_store = index_pdf_to_chroma("./data/BonBonFAQ.pdf")
agent = create_agent_with_tools(vector_store)
chatbot_conversation(agent)
