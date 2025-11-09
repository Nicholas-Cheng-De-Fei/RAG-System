from utils.utils import get_envvar
from utils.logger import log
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr, BaseModel, Field

ENV_GROQ_API_KEY="GROQ_API_KEY"
ENV_GROQ_MODEL="GROQ_MODEL"

correctness_prompt = PromptTemplate(
    input_variables=["question", "ground_truth", "generated_answer"],
    template="""
    You are an expert evaluator. Your sole task is to determine the correctness score.
    
    Question: {question}
    Ground Truth: {ground_truth}
    Generated Answer: {generated_answer}

    Based on the comparison, you **MUST** provide your final assessment by scoring from 0.0 to 1.0, 
    where 1.0 is perfectly correct and 0.0 is completely incorrect.

    **You MUST output the score by calling the provided function (tool) named 'ResultScore' and nothing else.**
    """
)

faithfulness_prompt = PromptTemplate(
    input_variables=["question","context", "generated_answer"],
    template="""
    Question: {question}
    Context: {context}
    Generated Answer: {generated_answer}

    Evaluate if the generated answer to the question can be deduced from the context.
    Score of 0 or 1, where 1 is perfectly faithful *AND CAN BE DERIVED FROM THE CONTEXT* and 0 otherwise.
    You don't mind if the answer is correct; all you care about is if the answer can be deduced from the context.

    Example:
    Question: What is 2+2?
    Context: 4.
    Generated Answer: 4.
    In this case, the context states '4', but it does not provide information to deduce the answer to 'What is 2+2?', so the score should be 0.
    
    **You MUST output the score by calling the provided function (tool) named 'ResultScore' and nothing else.**
    """
    
)

class ResultScore(BaseModel):
    score: float = Field(..., description="The score of the result, ranging from 0 to 1 where 1 is the best possible score.")

def connect_to_groq_ai() -> ChatGroq:
    api_key = get_envvar(ENV_GROQ_API_KEY)
    model = get_envvar(ENV_GROQ_MODEL)
    try:
        groq_ai= ChatGroq(
            model=model,
            api_key = SecretStr(api_key),
        )
        log.info(f"Open AI model set: {model}")
        return groq_ai
    except Exception as err:
        log.error(f"Could not create claude AI model due to, {err}")
        
def test_correctness(question, ground_truth, generated_answer, groq_ai: ChatGroq) -> float:
    correctness_chain = correctness_prompt | groq_ai.with_structured_output(ResultScore)

    result = correctness_chain.invoke({
        "question": question, 
        "ground_truth": ground_truth, 
        "generated_answer": generated_answer
    })
    return result.score

def test_faithfulness(question, context, generated_answer, groq_ai: ChatGroq) -> float:
    faithfulness_chain = faithfulness_prompt | groq_ai.with_structured_output(ResultScore)
    result = faithfulness_chain.invoke({
        "question": question, 
        "context": context, 
        "generated_answer": generated_answer
    })
    return result.score

