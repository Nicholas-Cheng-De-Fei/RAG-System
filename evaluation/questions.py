questions = [
    {
        "file" : "Computational_cost_of_semantic_chunking.pdf",
        "file_path" : "static/Computational cost of semantic chunking.pdf",
        "question" : "What is the defining characteristic of 'semantic chunking' in Retrieval-Augmented Generation (RAG) systems?",
        "ground_truth" : "Semantic chunking is a strategy that aims to improve retrieval performance by dividing documents into semantically coherent segments. This approach segments documents based on semantic similarity or detecting semantic distance thresholds between consecutive sentences to maintain coherence"
    },
    {
        "file" : "Computational_cost_of_semantic_chunking.pdf",
        "file_path" : "static/Computational cost of semantic chunking.pdf",
        "question" : "What three primary proxy tasks were designed to indirectly evaluate the quality of chunking strategies in the study presented in the sources?",
        "ground_truth": "The study systematically evaluated the effectiveness of chunking strategies using three proxy tasks: document retrieval, evidence retrieval, and retrieval-based answer generation"
    },
    {
        "file" : "Reconstructing_Context.pdf",
        "file_path" : "static/Reconstructing Context.pdf",
        "question" : "What are two advanced chunking techniques, besides traditional early chunking, aimed at preserving global context within RAG systems?",
        "ground_truth" : "Two advanced techniques introduced to preserve global context and mitigate context fragmentation are late chunking and contextual retrieval. Late chunking involves embedding the entire document first before segmentation to retain global context, potentially leading to superior results across various retrieval tasks."
    },
    {
        "file" : "Accelerating_LLM_Inference.pdf",
        "file_path" : "static/Accelerating LLM Inference.pdf",
        "question" : "In the context of long-context LLMs, what is the key phenomenon called where chunks attended to by tokens within a generated chunk exhibit substantial consistency, which ChunkLLM exploits to enhance inference efficiency?",
        "ground_truth" : "This phenomenon is called the Intra-Chunk Attention Consistency (ICAC) pattern. ChunkLLM exploits ICAC by updating chunk selection only when the currently decoded token is identified as a chunk boundary."
    },
    {
        "file" : "Beyond_Long_Context.pdf",
        "file_path" : "static/Beyond Long Context.pdf",
        "question" : "In the clinical domain, what is the methodology that uses entity-aware retrieval strategies to achieve improved semantic accuracy and computational efficiency when processing Electronic Health Record (EHR) notes?",
        "ground_truth" : "The methodology is Clinical Entity Augmented Retrieval (CLEAR). CLEAR addresses the limitations of traditional chunk-based RAG by employing entity-aware, entity-centered retrieval strategies and demonstrated a 78%"+" reduction in token usage compared to wide-context processing in evaluations"
    },
    {
        "file" : "Accelerating_LLM_Inference.pdf",
        "file_path" : "static/Accelerating_LLM_Inference.pdf",
        "question": "Contrast ChunkLLM and Dynamic Hierarchical Sparse Attention (DHSA) regarding their dynamic attention management mechanisms, specifically addressing how they derive chunk representations and utilize them to achieve efficiency gains while preserving performance in long-context models.",
        "ground_truth": "ChunkLLM and Dynamic Hierarchical Sparse Attention (DHSA) both propose mechanisms for efficient long-context modeling by dynamically managing attention sparsity, but they employ different architectural additions and chunk representation strategies. ChunkLLM introduces two pluggable components: the QK Adapter (Q-Adapter and K-Adapter) and the Chunk Adapter. The Chunk Adapter is a one-layer feed-forward neural network (FNN) classifier that detects if a token is a chunk boundary using contextual semantic information. The QK Adapter fulfills feature compression and generates chunk attention scores, trained using an attention distillation approach where the Kullback-Leibler (KL) divergence between chunk attention scores and full attention scores guides optimization to enhance the recall rate of key chunks. ChunkLLM leverages the Intra-Chunk Attention Consistency (ICAC) pattern, triggering chunk selection updates exclusively when the current token is identified as a chunk boundary, substantially enhancing inference efficiency. ChunkLLM maintains 98.64%" + " of the vanilla model's performance on long-context benchmarks and achieves a maximum speedup of 4.48x when processing 120K long texts. DHSA, conversely, is a plug-in module that dynamically predicts attention sparsity during prefill and decode stages without retraining the base model. DHSA employs a **Dynamic Hierarchical Sparsity Prediction approach. It first uses a boundary prediction function to adaptively segment input sequences into variable-length chunks. Chunk representations ($q_c$ and $k_c$) are derived by aggregating token queries and keys using a **length-normalized aggregation strategy, which involves scaling the sum of embeddings by the square root of the chunk size ($\sqrt{|C|}$) to mitigate sensitivity to variable chunk lengths. It estimates chunk-level similarity ($S_c$) and then upsamples it to obtain the token-level similarity matrix ($S_t$), applying TOPK selection to generate the sparsity mask. DHSA reports matching dense attention in accuracy, while reducing prefill latency by 20-60%"+ " and peak memory usage by 35%."
    },
    {
        "file" : ["Reconstructing_Context.pdf", "Computational_cost_of_semantic_chunking.pdf", "Beyond_Long_Context.pdf"],
        "file_path" : "multiple files",
        "question": "Explain the observed trade-offs between computational efficiency and semantic integrity across various chunking and retrieval strategies—including Fixed-size/Semantic Chunking, Late Chunking, Contextual Retrieval, and Clinical Entity Augmented Retrieval (CLEAR)—and identify which approach demonstrated superior scalability advantages in high-complexity clinical documents.",
        "ground_truth": "The sources reveal significant trade-offs among various chunking and retrieval strategies concerning computational cost and the preservation of semantic integrity. 1. Fixed-size vs. Semantic Chunking (RAG Baseline): Traditional fixed-size chunking is computationally simple and efficient. However, its simplicity risks fragmenting semantically related content, leading to suboptimal retrieval. Semantic chunking, which aims for semantically coherent segments, involves additional computational costs that the sources found were often not justified by consistent performance gains on standard document structures. Overall, fixed-size chunking was suggested as a more efficient and reliable choice for practical RAG applications on non-synthetic datasets. 2. Late Chunking vs. Contextual Retrieval: Late Chunking defers segmentation until after the entire document is embedded, preserving full contextual information for efficiency. Late Chunking offers higher efficiency but may sacrifice relevance and completeness. In contrast, Contextual Retrieval enhances chunks by prompting an LLM to generate additional context for each chunk, improving contextual integrity. This context preservation, particularly when combined with Rank Fusion (ContextualRankFusion), yields better overall results in retrieval evaluation than Late Chunking but incurs greater computational resources, potentially requiring up to 20GB of VRAM for chunk contextualization in long documents. 3. Clinical Entity Augmented Retrieval (CLEAR): This entity-aware method achieves a balance by selectively centering clinically relevant spans around identified entities, overcoming the positional bias ('lost in the middle' problem) associated with processing entire long documents. CLEAR achieved a 78.4% token savings compared to Wide Context processing while maintaining the highest average semantic similarity (0.878), demonstrating an optimal balance between accuracy and computational cost. The approach that demonstrated superior scalability advantages in high-complexity documents was CLEAR. In evaluations involving large clinical notes (exceeding 65,000 tokens), CLEAR achieved a 75% win rate, confirming that its entity-aware retrieval advantages grow as document complexity and document size increase, making it highly suitable for large EHR document processing"
    },
]

len_questions = len(questions)

def get_question(index:int):
    file = questions[index]["file"]
    file_path = questions[index]["file_path"]
    question = questions[index]["question"]
    ground_truth = questions[index]["ground_truth"]
    return file, file_path, question, ground_truth

