from cohere.client_v2 import ClientV2
from utils.logger import log
from utils.utils  import get_envvar

def connect_to_cohere_reranker() -> ClientV2:
    """
    Connect to cohere client using API key.
    """
    cohere_client = ClientV2(api_key=get_envvar("COHERE_API_KEY"))
    return cohere_client

def rerank(query: str, documents: list, k: int = 10) -> list:
    """
    Reranks the top-k candidates from Chroma using Cohere's Rerank model.
    """
    cohere_client = connect_to_cohere_reranker()
    log.info("INFO: Starting re-ranking process")

    rerank_response = cohere_client.rerank(
        model="rerank-v3.5",  # Can try rerank-multilingual-v3.0
        query=query,
        documents=documents,
        top_n=min(k, len(documents)),
    )

    # It is already sort by relevance and extract top results
    reranked_docs = [
        documents[result.index] for result in rerank_response.results
    ]
    log.info("INFO: Re-ranking process completed")

    return reranked_docs