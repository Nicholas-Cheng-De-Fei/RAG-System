from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# If using the pro version you can uncomment the below import to have multi modal inputs
# from langchain.messages import HumanMessage
import time
from utils.logger import log
from utils.utils import get_envvar

ENV_GOOGLE_AI_MODEL = "GOOGLE_AI_MODEL"
ENV_GOOGLE_APIKEY = "GOOGLE_APIKEY"

SYSTEM_PROMPT = """
    You are an intelligent and patient study assistant designed to help students understand their course notes.

    IMPORTANT, Your goals:
    1. Use the **retrieved notes** from the vector database as your primary knowledge source.
    2. Provide **accurate, clear, and well-structured explanations** based strictly on that context.
    3. If the answer is **not found in the notes** or if the context says "No relevant documents found," then you can answer based on what you understand, **politely noting that the information is not in the notes** and encourage the student to add more materials.
    4. When explaining, be **educational** — break down complex ideas, define key terms, and use examples when possible.

    IMPORTANT, Input format:
    - You will always receive a dictionary with two keys:
        - `context`: The retrieved notes or documents, or the string "No relevant documents found" if nothing was retrieved.
        - `query`: The student's question as a string.
    - Use the `context` to answer the `query` as accurately as possible.

    IMPORTANT, Response format:
    - **Answer:** Give the main explanation clearly.
    - **Key Points/Summary:** List 2–4 concise bullet points summarizing the answer.
    - **Extra Tip (if relevant):** Add a short clarification, example, or analogy to help understanding.

    Tone:
    Friendly, approachable, and encouraging — like a helpful tutor.
"""

TEST_PROMPT = """
    You are an intelligent and patient study assistant designed to help students understand their course notes.
    IMPORTANT: do not use any RAG tools such as online search. Answer the given query with only your pre-trained data and knowledge.
    
        IMPORTANT, Response format:
    - **Answer:** Give the main explanation clearly.
    - **Key Points/Summary:** List 2–4 concise bullet points summarizing the answer.
    - **Extra Tip (if relevant):** Add a short clarification, example, or analogy to help understanding.
    
    Tone:
    Friendly, approachable, and encouraging — like a helpful tutor.
"""

query_transformation_prompt = '''
    You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
'''

def connect_to_google_ai() -> ChatGoogleGenerativeAI:
    """
    Sets up the google ai model.
    """
    model = get_envvar(ENV_GOOGLE_AI_MODEL)
    api_key = get_envvar(ENV_GOOGLE_APIKEY)

    try: 
        google_ai  = ChatGoogleGenerativeAI(
            model = model,
            google_api_key = api_key
        )
        log.info(f"Model set: {model}")
        return google_ai
    except Exception as err:
        log.error(f"Could not create google AI model due to, {err}")

def query_google_ai(query: str, google_ai: ChatGoogleGenerativeAI) -> dict:
    """
    Invokes the google ai model with the query and returns its response.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]

    log.info("Awaiting response from AI model")
    start = time.perf_counter()
    
    response = google_ai.invoke(messages)

    end = time.perf_counter()
    log.info(f"Response received, took {end - start:.4f} seconds")

    return {"query" : query, "response": response}

def query_google_ai_test(query: str, google_ai: ChatGoogleGenerativeAI) -> dict:
    messages = [
        SystemMessage(content=TEST_PROMPT),
        HumanMessage(content=query)
    ]
    log.info("Awaiting response from AI model")
    start = time.perf_counter()
    
    response = google_ai.invoke(messages)

    end = time.perf_counter()
    log.info(f"Response received, took {end - start:.4f} seconds")

    return {"query" : query, "response": response}
    
def query_transformation(query:str, google_ai: ChatGoogleGenerativeAI) -> list:
    messages = [
        SystemMessage(content = query_transformation_prompt),
        HumanMessage(content=query)
    ]
    
    log.info("Awaiting query transformation from AI model")
    start = time.perf_counter()
    
    response = google_ai.invoke(messages)
    
    end = time.perf_counter()
    log.info(f"Response received, took {end - start:.4f} seconds")
        
    return response.content.split('\n')