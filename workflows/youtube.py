from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from pymongo import MongoClient
from typing_extensions import TypedDict, List, Union
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langgraph.types import interrupt, Command
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from pymongo import MongoClient


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")  

client = MongoClient("mongodb://localhost:27017/")
db = client["resume_db"]
users_col = db["users"]


class AppState(TypedDict):
    question: str
    answer: str
    search_results: str
    youtube_url: str
    video_transcript: str
    vector_store_path: str
    chatbot: RetrievalQA
    confidence: float
    messages:Annotated[List[dict], add_messages]

@tool(description="Searches the web using Tavily API. Input should be a search query.")
def travily_search(query: str) -> str:
    search = TavilySearch(max_results=5)
    return search.invoke(
        {query: query}
    )



llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
# llm2 =ChatGroq(
#     model="meta-llama/llama-4-scout-17b-16e-instruct"
# )

agent = create_react_agent(llm, tools=[travily_search])

def input_youtube_url(state: AppState) -> AppState:
    """Prompts the user to input a YouTube video URL."""
    youtube_url = interrupt("Enter the YouTube video id: ")
    state["youtube_url"] = youtube_url
    return state

def extract_youtube_transcript(state: AppState) -> AppState:
    """Extracts the transcript from a YouTube video URL."""
    loader =  YouTubeTranscriptApi()
    documents = loader.fetch(state["youtube_url"])
    full_transcript = " ".join([snippet.text for snippet in documents.snippets])
    state["video_transcript"] = full_transcript
    print(state["video_transcript"][:500])  # Print first 500 characters for debugging
    return state

def store_transcript_in_vector_db(state: AppState) -> AppState:
    """Stores the extracted transcript in a vector database."""
    persist_path = "./chroma_store/youtube_transcript" 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(state["video_transcript"])

    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_path,
        collection_name="youtube_transcript_mxbai"
      )

    state["vector_store_path"] = persist_path
    # state["vector_store"] = vector_store 
    print(f"Stored {len(chunks)} chunks in vector database.")
    return state
    
def chatbot(state: AppState) -> AppState:
    """Creates a chatbot using the extracted transcript and vector database."""
    llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    vector_store = Chroma(
        persist_directory=state["vector_store_path"],
        embedding_function=embeddings,
        collection_name="youtube_transcript_mxbai"
    )
    question = state["question"]
    retrieved_docs = vector_store.similarity_search_with_score(question)
    # average_score = sum([s for _, s in retrieved_docs]) / len(retrieved_docs)
    # state["confidence"] =  average_score if average_score > 0 else 0
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided YouTube video transcript. Use the information from the transcript to answer the user's questions. If you don't know the answer, say 'I don't know'."),
        ("human", "Question: {question}\nContext: {context}\nAnswer:")
    ])
    
    answer_chain = chat_prompt | llm
    print("retrived_docs", retrieved_docs)
    ai_message = answer_chain.invoke({"question": question, "context": retrieved_docs})

    state["answer"] = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)
    print(f"Answer: {state['answer']}")  
    return state

def ask_chatbot(state: AppState) -> AppState:
    """Allows the user to ask questions to the chatbot."""
    question = interrupt("Ask a question about the video (or type 'exit' to quit): ")
    if question and question.lower() != "exit":
        state["question"] = question
        return state
    else:
        return state

def create_agent(state: AppState) -> AppState:
    """Creates a React agent for the chatbot."""
    # search_result = agent.invoke(state["question"])
    messages = [{"role": "user", "content": state["question"]}]
    search_result = agent.invoke({"messages": messages})
    state["search_results"] = search_result
    return state

def get_confidence(state: AppState) -> float:
    """Returns the confidence score of the chatbot's answer."""
    llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
    embeddings = OllamaEmbeddings(model="llama3.1:8b", base_url="http://localhost:11434")
    vector_store = Chroma(
        persist_directory=state["vector_store_path"],
        embedding_function=embeddings,
        collection_name="youtube_transcript"
    )
    question = state["question"]
    retrieved_docs = vector_store.similarity_search_with_score(question)
    average_score = sum([s for _, s in retrieved_docs]) / len(retrieved_docs)
    state["confidence"] = 1 - average_score if average_score > 0 else 0
    return state.get("confidence", 0.0)

def should_continue(state: AppState) -> str:
    """Determines whether to continue asking questions or end."""
    if "question" not in state or state.get("question", "").lower() == "exit":
        return "end"
    return "continue"

builder = StateGraph(AppState)

builder.add_node("input_youtube_url", input_youtube_url)
builder.add_node("extract_youtube_transcript", extract_youtube_transcript)
builder.add_node("store_transcript_in_vector_db", store_transcript_in_vector_db)
builder.add_node("chatbot", chatbot)
builder.add_node("ask_chatbot", ask_chatbot)
builder.add_node("agent", create_agent)

builder.set_entry_point("input_youtube_url")
builder.add_edge("input_youtube_url", "extract_youtube_transcript")
builder.add_edge("extract_youtube_transcript", "store_transcript_in_vector_db")
builder.add_edge("store_transcript_in_vector_db", "ask_chatbot")
# builder.add_edge("ask_chatbot", "chatbot")
builder.add_conditional_edges(
    "ask_chatbot",
    should_continue,
    {
        "continue": "chatbot",
        "end": END
    }
)
builder.add_edge("chatbot", "ask_chatbot")