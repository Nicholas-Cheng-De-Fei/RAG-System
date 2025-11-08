from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import numpy as np
from pathlib import Path
import re
from sklearn.metrics.pairwise import cosine_similarity
import time
from utils.logger import log
from utils.utils import get_envvar

ENV_HUGGINGFACE_EMBEDDING_MODEL = "HUGGINGFACE_EMBEDDING_MODEL"

def read_pdf_document(document_path: str) -> str:
    """
    Reads the PDF document based on the document path.

    Returns
    -------
    A list of langchain document objects.
    """
    path = Path(document_path)

    # If the path is not absolute, make it absolute
    if (not path.is_absolute()):
        path = path.resolve()
    try:
        loader = PyPDFLoader(str(path))
        raw_docs = loader.load()
        # Clean the text for each document/page
        cleaned_docs = []
        for doc in raw_docs:
            cleaned_content = clean_text(doc.page_content)
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            cleaned_docs.append(cleaned_doc)

        return cleaned_docs
    except ValueError as value_err:
        log.error(f"File path {document_path} is invalid or the file cannot be found")
        raise HTTPException(status_code=400, detail="Invalid file path or file cannot be found")
    
def read_pdf_document_into_markdown(document_path: str) -> str:
    """
    Reads the PDF document based on the document path and converts it into markdown format.

    Returns
    -------
    A string containing the markdown representation of the PDF.
    """
    path = Path(document_path)

    # If the path is not absolute, make it absolute
    if (not path.is_absolute()):
        path = path.resolve()
    try:
        loader = PyMuPDF4LLMLoader(str(path))
        return loader.load()
    except ValueError as value_err:
        log.error(f"File path {document_path} is invalid or the file cannot be found")
        raise HTTPException(status_code=400, detail="Invalid file path or file cannot be found")

def clean_text(text: str) -> str:
    """
    Light text cleaning for PDF content.
    Removes HTML, URLs, extra whitespace, and non-printable characters.
    """
    text = re.sub(r"<[^>]+>", " ", text)              # remove HTML
    text = re.sub(r"http\S+|www\.\S+", "", text)      # remove URLs
    text = re.sub(r"\[[0-9]*\]", "", text)            # remove citation marks [1], [2]
    text = re.sub(r"\s+", " ", text)                  # collapse multiple spaces/newlines
    text = text.replace("\u2212", "-")
    text = re.sub(r"[^\x20-\x7E]", "", text)          # remove non-ASCII characters
    return text.strip()

def native_chunking(documents: list) -> list:
    """
    Spilts the text into multiple chunks based on the natural structure such as breakpoints, new lines etc.

    Returns
    -------
    Returns a list of documents but are smaller in text length than the original document list.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", "."],
    )
    log.info("Chunking process has begun")
    start = time.perf_counter()

    base_chunks = splitter.split_documents(documents)

    end = time.perf_counter()
    log.info(f"Chunking process completed, took {end - start:.4f} seconds")
    
    return base_chunks

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model.
    """
    huggingface_embedding_model = get_envvar(ENV_HUGGINGFACE_EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_embedding_model)
    return embeddings


#################################
### SEMANTIC CHUNKING STUFF
#################################

# Assuming you have a function to get your embedding model
# from your_embedding_module import get_embedding_model

def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['embedding']
        embedding_next = sentences[i + 1]['embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences

def semantic_chunking(documents: list, max_sentences_per_chunk: int = 6) -> list:
    """
    Spilts the text into multiple chunks based on semantic meaning and a maximum sentence length.

    Returns
    -------
    Returns a list of documents but are smaller in text length than the original document list.
    """        
    combined_text = "\n".join(getattr(document, "page_content", str(document))
                               for document in documents)
    if not combined_text.strip():
        # Handle cases with no text content
        return []

    single_sentences_list = re.split(r'(?<=[.?!])\s+', combined_text)
    print(f"{len(single_sentences_list)} sentences were found")
    
    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list) if x]
    
    if not sentences:
        return []

    sentences = combine_sentences(sentences, buffer_size=1)

    # This part is a placeholder. You should have your own embedding model logic.
    # For demonstration, we'll create dummy embeddings.
    # In a real scenario, you would use a proper embedding model.
    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_documents(
        [sentence['combined_sentence'] for sentence in sentences])

    for i, sentence in enumerate(sentences):
        sentence['embedding'] = embeddings[i]
        
    distances, sentences = calculate_cosine_distances(sentences)

    # We need to get the distance threshold that we'll consider an outlier
    # We'll use numpy .percentile() for this
    breakpoint_percentile_threshold = 60
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff

    # Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

    # Initialize the start index
    start_index = 0

    # Create a list to hold the grouped sentences
    chunks = []
    
    # Find all unique breakpoint indices and sort them.
    breakpoint_indices = sorted(list(set(indices_above_thresh)))

    # Iterate through the breakpoints to slice the sentences
    for i, index in enumerate(breakpoint_indices):
        # The end index is the current breakpoint
        end_index = index

        # Slice the sentence_dicts from the current start index to the end index
        group = sentences[start_index:end_index + 1]
        
        # Further split the group if it's too long
        for j in range(0, len(group), max_sentences_per_chunk):
            sub_group = group[j:j + max_sentences_per_chunk]
            combined_text = ' '.join([d['sentence'] for d in sub_group])
            chunks.append(combined_text)
        
        # Update the start index for the next group
        start_index = index + 1

    # The last group, if any sentences remain
    if start_index < len(sentences):
        remaining_group = sentences[start_index:]
        for j in range(0, len(remaining_group), max_sentences_per_chunk):
            sub_group = remaining_group[j:j+max_sentences_per_chunk]
            combined_text = ' '.join([d['sentence'] for d in sub_group])
            chunks.append(combined_text)

    # grouped_sentences now contains the chunked sentences
    
    for i, chunk in enumerate(chunks[:10]):
        buffer = 200
        
        print (f"Chunk #{i}")
        print (chunk.strip())
        print ("\n")
        
    return [Document(page_content=chunk) for chunk in chunks]

## LAYOUT CHUNKING

def layout_chunking(documents: list) -> list:
    """
    Spilts the text into multiple chunks based on layout structure.

    Returns
    -------
    Returns a list of documents but are smaller in text length than the original document list.
    """
    # Placeholder for layout-based chunking logic
    # This would typically involve analyzing the document's layout elements
    combined_text = "\n".join(getattr(document, "page_content", str(document))
                               for document in documents)
    if not combined_text.strip():
        # Handle cases with no text content
        return []

    # --- REGEX CONVERSIONS ---
    # NOTE: The order is important. Go from most specific to most general.

    # 1. Handles formats like **1.2**Title and also ** 1.2 ** Title
    processed_text = re.sub(r'\n\*\*\s*(\d\.\d)\s*\*\*\s*([^\n]+)', r'\n## \1 \2', combined_text)

    # 2. Handles formats like **1**Title and also ** 1 ** Title
    processed_text = re.sub(r'\n\*\*\s*(\d)\s*\*\*\s*([^\n]+)', r'\n# \1 \2', processed_text)

    # 3. Handle any other bolded titles without numbers like **Introduction**
    processed_text = re.sub(r'\n\*\*\s*(.+?)\s*\*\*', r'\n# \1', processed_text)

    processed_text = processed_text.encode("utf-8", errors="replace").decode()

    # log.info(f"Document page content preview: {combined_text[:500]}")
    # log.info("\n\n\n")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    log.info("Chunking based on layout structure has begun")
    
    start = time.perf_counter()
    
    final_structured_documents = markdown_splitter.split_text(processed_text)
    
    end = time.perf_counter()
    log.info(f"Chunking process completed, took {end - start:.4f} seconds")
    
    # Uncomment this to see all the chunks
    # for document in final_structured_documents:
    #     log.info(f"Layout-based chunk preview: {document.page_content}")
    log.info(f"Number of layout-based chunks created: {len(final_structured_documents)}")
    
    return final_structured_documents
