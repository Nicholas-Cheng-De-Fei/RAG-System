from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from services.query_service import query_google_ai, query_transformation
from services.chroma_db_service import connect_to_chroma_db, disconnect_chroma_db, retrieve, multi_retrieve, rerank_retrieve
from services.reranking import rerank
from evaluation.questions import len_questions, get_question
from evaluation.evaluations import test_correctness, test_faithfulness
from models.app_models import DocumentProcessRequest
from controllers.app_controller import chunk_document, chunk_document_with_layout
from utils.logger import log

import json

### For base RAG Model
def base_chunk_base_retrieve(google_ai: ChatGoogleGenerativeAI, groq_ai: ChatGroq):
    result = []
    model = "base"
    evaluation_data = []
    
    for i in range(len_questions):
        file, file_path, question, ground_truth = get_question(i)
        log.info(f"{model} model: evaluating question {i+1}") 
        
        if isinstance(file, list):
            db_name = "normal-combi"
        else:
            db_name = f"normal-{file}"
        db = connect_to_chroma_db(db_name)
        context = retrieve(question, db)
        query_with_context = f"Context:\n{context['context']}\n\nQuestion:\n{question}"
        response = query_google_ai(query_with_context, google_ai)
        generated_answer = response["response"].content
        
        correctness_score = test_correctness(question, ground_truth, generated_answer, groq_ai)
        faithfulness_score = test_faithfulness(question, context["context"], generated_answer, groq_ai)
        result.append((correctness_score, faithfulness_score))
        evaluation_data.append({
            "question_id": i + 1,
            "document_path": file_path,
            "question": question,
            "ground_truth": ground_truth,
            "retrieved_context": context,
            "generated_answer": generated_answer,
            "correctness_score": correctness_score,
            "faithfulness_score": faithfulness_score,
            "model_tested": model
        })
        
    log.info(f"{model} model results: {result}")
    evaluation_data.insert(0, result)
    output_filename = f"evaluation/results/evaluation_results_{model}.json"
    with open(output_filename, 'w') as f:
        json.dump(evaluation_data, f, indent=4)
        
    return result

### For query transformation model
def base_chunk_multi_retrieve(google_ai: ChatGoogleGenerativeAI, groq_ai: ChatGroq):
    db: Chroma
    result = []
    model = "Query Transformation"
    evaluation_data = []
    
    for i in range(len_questions):
        file, file_path, question, ground_truth = get_question(i)
        log.info(f"{model} model: evaluating question {i+1}") 
        
        if isinstance(file, list):
            db_name = "normal-combi"
        else:
            db_name = f"normal-{file}"
        db = connect_to_chroma_db(db_name)
        queries = query_transformation(question, google_ai)
        context = multi_retrieve(queries, db)
        
        query_with_context = f"Context:\n{context}\n\nQuestion:\n{question}"
        response = query_google_ai(query_with_context, google_ai)
        generated_answer = response["response"].content
        
        correctness_score = test_correctness(question, ground_truth, generated_answer, groq_ai)
        faithfulness_score = test_faithfulness(question, context, generated_answer, groq_ai)
        
        result.append((correctness_score, faithfulness_score))
        evaluation_data.append({
            "question_id": i + 1,
            "document_path": file_path,
            "question": question,
            "ground_truth": ground_truth,
            "retrieved_context": context,
            "generated_answer": generated_answer,
            "correctness_score": correctness_score,
            "faithfulness_score": faithfulness_score,
            "model_tested": model
        })
    
    log.info(f"{model} model results: {result}")
    evaluation_data.insert(0, result)
    output_filename = f"evaluation/results/evaluation_results_{model}.json"
    with open(output_filename, 'w') as f:
        json.dump(evaluation_data, f, indent=4)
    return result

### For reranking model
def base_chunk_rerank(google_ai: ChatGoogleGenerativeAI, groq_ai: ChatGroq):
    db: Chroma
    result = []
    model = "rerank"
    evaluation_data = []
    
    for i in range(len_questions):
        file, file_path, question, ground_truth = get_question(i)        
        log.info(f"{model} model: evaluating question {i+1}") 
                  
        if isinstance(file, list):
            db_name = "normal-combi"
        else:
            db_name = f"normal-{file}"
        db = connect_to_chroma_db(db_name)
        
        retrieved_documents = rerank_retrieve(question, db)
        if retrieved_documents:
            # Extract the string content from the list of Document objects
            documents_to_rerank = [getattr(doc, "page_content", str(doc)) for doc in retrieved_documents] 
            
            # Rerank the list of strings
            reranked_context = rerank(question, documents_to_rerank)
        else:
            reranked_context = ["No relevant documents retrieved"]
            
        context = "\n\n".join(reranked_context)
        query_with_context = f"Context:\n{context}\n\nQuestion:\n{question}"

        response = query_google_ai(query_with_context, google_ai)
        generated_answer = response["response"].content
        
        correctness_score = test_correctness(question, ground_truth, generated_answer, groq_ai)
        faithfulness_score = test_faithfulness(question, context, generated_answer, groq_ai)
        
        result.append((correctness_score, faithfulness_score))
        evaluation_data.append({
            "question_id": i + 1,
            "document_path": file_path,
            "question": question,
            "ground_truth": ground_truth,
            "retrieved_context": context,
            "generated_answer": generated_answer,
            "correctness_score": correctness_score,
            "faithfulness_score": faithfulness_score,
            "model_tested": model
        })
    
    log.info(f"{model} model results: {result}")
    evaluation_data.insert(0, result)
    output_filename = f"evaluation/results/evaluation_results_{model}.json"
    with open(output_filename, 'w') as f:
        json.dump(evaluation_data, f, indent=4)
    return result

### For layout chunking model
def layout_chunk_base_retrieve(google_ai: ChatGoogleGenerativeAI, groq_ai: ChatGroq):
    db: Chroma
    result = []
    model = "Layout"
    evaluation_data = []
    
    for i in range(len_questions):
        i = 3
        file, file_path, question, ground_truth = get_question(i)        
        log.info(f"{model} model: evaluating question {i+1}") 
                
        try: 
            if isinstance(file, list):
                db_name = "layout-combi"
            else:
                db_name = f"layout-{file}"
            db = connect_to_chroma_db(db_name)
            
            context = retrieve(question, db, k = 4)
            query_with_context = f"Context:\n{context['context']}\n\nQuestion:\n{question}"
            
            response = query_google_ai(query_with_context, google_ai)
            generated_answer = response["response"].content
            
            correctness_score = test_correctness(question, ground_truth, generated_answer, groq_ai)
            faithfulness_score = test_faithfulness(question, context["context"][:-3100], generated_answer, groq_ai)
            
            result.append((correctness_score, faithfulness_score))
            evaluation_data.append({
                "question_id": i + 1,
                "document_path": file_path,
                "question": question,
                "ground_truth": ground_truth,
                "retrieved_context": context,
                "generated_answer": generated_answer,
                "correctness_score": correctness_score,
                "faithfulness_score": faithfulness_score,
                "model_tested": model
            })
        except Exception as e:
            log.error(f"Failed to process question {i+1}: {e}")
            evaluation_data.append({
                "question_id": i + 1,
                "failed": True,
                "context length": len(context["context"]),
                "context": context
            })
        finally:
            break
        

    log.info(f"{model} model results: {result}")
    evaluation_data.insert(0, result)
    output_filename = f"evaluation/results/evaluation_results_{model}.json"
    with open(output_filename, 'w') as f:
        json.dump(evaluation_data, f, indent=4)
    return result

### For model with everything
def layout_chunk_multi_retrieve(google_ai: ChatGoogleGenerativeAI, groq_ai: ChatGroq):
    db: Chroma
    result = []
    model = "everything"
    evaluation_data = []
    
    for i in range(len_questions):
        i = 5
        file, file_path, question, ground_truth = get_question(i)        
        log.info(f"{model} model: evaluating question {i+1}") 
        
        try: 
            if isinstance(file, list):
                db_name = "layout-combi"
            else:
                db_name = f"layout-{file}"
            db = connect_to_chroma_db(db_name)
            queries = query_transformation(question, google_ai)
            context = multi_retrieve(queries, db, k=10)
            
            if context: # Only proceed if the context list is not empty
                # Find the longest string (document) in the context list
                longest_context_string = max(context, key=len)
                
                # Remove the longest string from the context list
                try:
                    context.remove(longest_context_string)
                    log.info(f"Removed longest context string of length: {len(longest_context_string)}")
                except ValueError:
                    # This handles the unlikely case where .remove() fails
                    log.warning("Longest context string was not found for removal.")
                    
                longest_context_string = max(context, key=len)
                # Remove the longest string from the context list
                try:
                    context.remove(longest_context_string)
                    log.info(f"Removed longest context string of length: {len(longest_context_string)}")
                except ValueError:
                    # This handles the unlikely case where .remove() fails
                    log.warning("Longest context string was not found for removal.")
                    
                
            
            query_with_context = f"Context:\n{context}\n\nQuestion:\n{question}"
            response = query_google_ai(query_with_context, google_ai)
            generated_answer = response["response"].content
            
            correctness_score = test_correctness(question, ground_truth, generated_answer, groq_ai)
            faithfulness_score = test_faithfulness(question, context, generated_answer, groq_ai)
            
            result.append((correctness_score, faithfulness_score))
            evaluation_data.append({
                "question_id": i + 1,
                "document_path": file_path,
                "question": question,
                "ground_truth": ground_truth,
                "retrieved_context": context,
                "generated_answer": generated_answer,
                "correctness_score": correctness_score,
                "faithfulness_score": faithfulness_score,
                "model_tested": model
            })
        except Exception as e:
            log.error(f"Failed to process question {i+1}: {e}")
            evaluation_data.append({
                "question_id": i + 1,
                "failed": True,
                "context length": len(context),
                "context": context
            })
        finally:
            break
    
    log.info(f"{model} model results: {result}")
    evaluation_data.insert(0, result)
    output_filename = f"evaluation/results/evaluation_results_{model}.json"
    with open(output_filename, 'w') as f:
        json.dump(evaluation_data, f, indent=4)
    return result
