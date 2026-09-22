"""Corpus Generation Script for QASubjectBot.
Generates 14 realistic, in-depth technical documents on Large Language Models,
Transformers, and RAG Systems under the /documents directory.
- 12 multi-page PDF documents with running headers, footers, and page numbers
- 2 comprehensive Plain Text documents
"""

import html
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render running headers and total page numbers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))

        # Running Header (on pages > 1)
        if self._pageNumber > 1:
            header_text = getattr(self, "doc_title", "QASubjectBot Technical Knowledge Series")
            self.drawString(54, 750, header_text)
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer (on all pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_str)
        self.drawString(54, 40, "Confidential & Proprietary - AI Research Lab Reference Corpus")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)

        self.restoreState()


def escape_rl(text):
    """Escapes HTML/XML entities so ReportLab paraparser doesn't crash on <, >, &."""
    escaped = html.escape(text, quote=False)
    return escaped.replace("\n", "<br/>")


DOCUMENTS_DATA = [
    {
        "filename": "01_neural_network_foundations.pdf",
        "title": "Chapter 1: Foundations of Deep Neural Networks and Backpropagation",
        "pages": [
            [
                ("Section 1.1: Biological Inspiration and Artificial Neurons", 
                 "Artificial Neural Networks (ANNs) are computational models inspired by biological neural networks in animal brains. The elementary unit of an artificial neural network is the perceptron, first introduced by Frank Rosenblatt in 1958. In a modern feedforward neural network, each artificial neuron computes a weighted sum of its inputs, adds a scalar bias term, and passes the resulting pre-activation through a non-linear activation function.\n\n"
                 "Mathematically, for an input vector x in R^d, weights W in R^{m x d}, and bias b in R^m, the activation output y is computed as y = f(W * x + b), where f is an element-wise non-linear activation function. Without non-linear activation functions, any multilayer neural network collapses into a single linear transformation, regardless of its depth, rendering deep architectures incapable of approximating complex continuous functions as guaranteed by the Universal Approximation Theorem."),
                ("Section 1.2: Activation Functions",
                 "Key activation functions historically shaped deep learning convergence and representational capacity:\n"
                 "- Sigmoid: defined as sigma(z) = 1 / (1 + exp(-z)), mapping real values to the interval (0, 1). While useful for binary classification probabilities, sigmoid is susceptible to vanishing gradients at extreme values because its derivative sigma'(z) = sigma(z) * (1 - sigma(z)) peaks at only 0.25.\n"
                 "- Hyperbolic Tangent (tanh): defined as tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)), mapping real values to (-1, 1), providing zero-centered activations that stabilize early optimization.\n"
                 "- Rectified Linear Unit (ReLU): defined as f(x) = max(0, x). ReLU alleviated vanishing gradients in positive regimes, accelerating convergence during forward and backward passes across deep convolutional and dense architectures.\n"
                 "- Gaussian Error Linear Unit (GELU): defined as x * Phi(x), where Phi(x) is the standard Gaussian cumulative distribution function. GELU weights inputs by their value rather than gating strictly by their sign. GELU is the standard activation function across modern transformer architectures including BERT, GPT-3, and LLaMA."),
            ],
            [
                ("Section 1.3: Loss Functions and Backpropagation",
                 "Training a deep neural network requires measuring the discrepancy between model predictions and true ground-truth targets using a scalar loss function L(theta). For multi-class classification tasks over K classes, Categorical Cross-Entropy Loss is standard: L = - sum_{k=1}^K y_k * log(p_k), where p_k is the predicted softmax probability for class k, defined as p_k = exp(z_k) / sum_{j=1}^K exp(z_j).\n\n"
                 "The Backpropagation algorithm, popularized by Rumelhart, Hinton, and Williams in 1986, computes the exact gradient of the scalar loss with respect to every parameter in the network through the recursive application of the multivariate chain rule from differential calculus. Reverse-mode automatic differentiation enables efficient computation of all parameter gradients in computational time proportional to a single forward inference pass."),
                ("Section 1.4: Optimization Algorithms and Regularization",
                 "Stochastic Gradient Descent (SGD) updates parameters theta along the negative gradient direction: theta_{t+1} = theta_t - lr * grad(L). Modern architectures rely on adaptive learning rate optimizers:\n"
                 "- Momentum: accelerates SGD in consistent gradient directions and dampens oscillations: v_t = gamma * v_{t-1} + lr * grad(L).\n"
                 "- Adam (Adaptive Moment Estimation): maintains exponentially decaying moving averages of past gradients (first moment m_t with beta1=0.9) and squared gradients (second moment v_t with beta2=0.999), applying bias correction before updating parameters.\n"
                 "- AdamW (Loshchilov & Hutter, 2017): decouples weight decay regularization from the gradient update step, preventing L2 penalty distortion in adaptive optimizers. AdamW is ubiquitous in modern large language model training with default beta1=0.9, beta2=0.95, and weight decay=0.1."),
            ],
            [
                ("Section 1.5: Weight Initialization and Gradient Stability",
                 "Proper weight initialization is critical for preventing exploding or vanishing gradients during the initial optimization steps of deep models. Xavier (Glorot) initialization draws weights from a uniform distribution U(-sqrt(6/(d_in + d_out)), sqrt(6/(d_in + d_out))) to preserve activation variance across layers with symmetric activations like tanh. He (Kaiming) initialization scales variance as 2 / d_in, specifically tailored to avoid signal degradation in ReLU networks.\n\n"
                 "Modern deep transformer models incorporate Residual Connections (He et al., 2016), which add identity skip connections around non-linear sub-layers: x_{l+1} = x_l + F(x_l). Residual connections provide an uninterrupted gradient highway back to early layers, enabling models with hundreds of layers to train stably without gradient attenuation."),
            ],
        ],
    },
    {
        "filename": "02_transformer_architecture_and_attention.pdf",
        "title": "Chapter 2: The Transformer Architecture and Self-Attention Mechanisms",
        "pages": [
            [
                ("Section 2.1: The Paradigm Shift from RNNs to Transformers",
                 "Prior to 2017, sequence transduction models relied primarily on recurrent neural networks (RNNs) and Long Short-Term Memory networks (LSTMs). RNNs process tokens sequentially step-by-step, creating an unavoidable temporal dependency bottleneck that prohibits parallelization across sequence positions during training. Furthermore, gradients suffer from exponential decay across long context windows, leading to catastrophic forgetting of distant information.\n\n"
                 "In 2017, Vaswani et al. published 'Attention Is All You Need', introducing the Transformer architecture. The Transformer dispenses entirely with recurrence and convolutions, relying solely on self-attention mechanisms to model global dependencies between input and output tokens in constant O(1) sequential operations, unleashing massive parallelization on modern GPU accelerators."),
                ("Section 2.2: Scaled Dot-Product Attention",
                 "Scaled Dot-Product Attention operates on three matrices: Queries (Q), Keys (K), and Values (V), each projected into dimension d_k. The attention weights are computed by taking the matrix dot product of Q with K transpose, dividing by the scaling factor sqrt(d_k), and applying the softmax function along sequence rows:\n\n"
                 "Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V\n\n"
                 "The scaling factor 1 / sqrt(d_k) is mathematically essential: for large projection dimensions d_k, dot products grow large in magnitude, pushing the softmax function into regions with near-zero gradients. Scaling ensures numerical stability and maintains healthy gradient flow during backpropagation."),
            ],
            [
                ("Section 2.3: Multi-Head Attention and Modern Variants",
                 "Instead of performing a single attention function with d_model-dimensional queries, keys, and values, Multi-Head Attention (MHA) linearly projects Q, K, and V h times with learned parameter matrices to dimensions d_k, d_k, and d_v respectively. Each head attends to information from different representation subspaces at different positions:\n\n"
                 "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O, where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V).\n\n"
                 "Modern architectures have evolved beyond standard MHA to optimize inference memory bandwidth:\n"
                 "- Multi-Query Attention (MQA, Shazeer 2019): shares a single Key and Value head across all Query heads, slashing KV cache memory footprint by 8x-16x.\n"
                 "- Grouped-Query Attention (GQA, Ainslie et al. 2023): partitions query heads into G groups, with each group sharing a single Key and Value head. GQA achieves nearly the same quality as full MHA with the inference speed and memory savings of MQA (used in LLaMA 2/3, Mistral, and Gemma)."),
                ("Section 2.4: Positional Encodings and Normalization Layers",
                 "Because self-attention is permutation-invariant, positional information must be injected into token representations. Sinusoidal positional encodings use fixed trigonometric functions of varying frequencies. Modern architectures utilize Rotary Position Embeddings (RoPE, Su et al. 2021), which encode absolute positions via rotation matrices in the complex plane, naturally preserving relative positional distances under dot products and enabling context length extrapolation.\n\n"
                 "Layer Normalization is applied either before (Pre-LN) or after (Post-LN) the sub-layers. Modern transformers utilize Pre-LN with Root Mean Square Normalization (RMSNorm, Zhang & Sennrich 2019) to stabilize deep residual streams during high-throughput training while reducing computational overhead by 10-50%."),
            ],
            [
                ("Section 2.5: Feed-Forward Networks and SwiGLU",
                 "Each transformer layer contains a position-wise Feed-Forward Network (FFN) following the attention sub-layer. In the original transformer, the FFN consisted of two linear transformations with a ReLU activation: FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2, with an inner dimension d_ff typically equal to 4 * d_model.\n\n"
                 "Modern state-of-the-art architectures (LLaMA, PaLM, Mistral) adopt the SwiGLU activation variant (Shazeer, 2020), which uses a Swish-gated linear unit: SwiGLU(x) = (Swish_1(x * W_gate) * (x * W_up)) * W_down, where Swish_1(z) = z * sigmoid(z). SwiGLU consistently yields superior downstream perplexity and reasoning performance compared to standard ReLU or GELU feed-forward networks."),
            ],
        ],
    },
    {
        "filename": "03_llm_pretraining_and_scaling_laws.pdf",
        "title": "Chapter 3: Large Language Model Pretraining, Datasets, and Scaling Laws",
        "pages": [
            [
                ("Section 3.1: Pretraining Objectives and Auto-Regressive Modeling",
                 "Modern Large Language Models (LLMs) such as GPT-4, Claude 3, and LLaMA 3 are pretrained on massive web-scale text corpora using the Causal Language Modeling (CLM) objective. Given an arbitrary token sequence x = (x_1, x_2, ..., x_T), the model is trained to maximize the log-likelihood of predicting the next token x_t conditioned on all preceding tokens x_{prev}:\n\n"
                 "L_CLM = sum_{t=1}^T log P(x_t | x_1, ..., x_{t-1}; theta)\n\n"
                 "This autoregressive next-token prediction task forces the model to compress vast amounts of syntactic, semantic, factual, reasoning, and world knowledge into its billions of learned parameters."),
                ("Section 3.2: Data Curation, Filtering, and Deduplication",
                 "The quality of pretraining data fundamentally dictates downstream model capability. Modern pretraining datasets (e.g. Common Crawl, FineWeb, RedPajama) undergo strict data processing pipelines:\n"
                 "1. Text Extraction and Formatting: Stripping HTML boilerplate, script tags, and metadata.\n"
                 "2. Heuristic Quality Filtering: Removal of short documents, low-diversity texts, repetitive n-grams, and high punctuation ratios.\n"
                 "3. Deduplication: Exact URL matching, MinHash LSH (Locality-Sensitive Hashing) for near-duplicate document removal, and suffix-array based substring deduplication.\n"
                 "4. Safety & PII Redaction: Filtering hate speech, toxic content, and redacting personal identifiable information (emails, phone numbers, addresses)."),
            ],
            [
                ("Section 3.3: Neural Scaling Laws and the Chinchilla Optimal Ratio",
                 "In 2020, Kaplan et al. from OpenAI observed that cross-entropy loss scales as a power-law with respect to compute budget C, dataset size D (in tokens), and model parameter count N: L(N, D) ~ (N_c / N)^alpha_N + (D_c / D)^alpha_D.\n\n"
                 "In 2022, Hoffmann et al. from DeepMind revised these findings in the seminal 'Chinchilla' study. They demonstrated that for compute-optimal training, model size N and training tokens D should be scaled in equal proportion (approximately 20 tokens per model parameter). Consequently, earlier models like GPT-3 (175B parameters trained on 300B tokens) were severely undertrained compared to Chinchilla (70B parameters trained on 1.4T tokens). Modern models now train on 15T+ tokens to maximize downstream inference efficiency."),
                ("Section 3.4: Distributed Training Infrastructure",
                 "Training hundreds of billions of parameters requires distributed parallelism paradigms across clusters of thousands of GPUs:\n"
                 "- Tensor Parallelism (TP, e.g. Megatron-LM): splits individual weight matrices across GPUs within a single NVLink node.\n"
                 "- Pipeline Parallelism (PP): partitions consecutive transformer layers across separate nodes.\n"
                 "- Fully Sharded Data Parallelism (FSDP / ZeRO-3): shards model weights, gradients, and optimizer states across all compute ranks, eliminating memory redundancies."),
            ],
            [
                ("Section 3.5: Tokenization Paradigms: Byte-Pair Encoding and WordPiece",
                 "Text is converted into numerical discrete tokens before entering neural networks. Modern LLMs use subword tokenization algorithms:\n"
                 "- Byte-Pair Encoding (BPE, Sennrich et al. 2016): iteratively merges the most frequent pairs of bytes or characters in the corpus into single tokens. Tiktoken (cl100k_base, o200k_base) uses byte-level BPE with vocabulary sizes ranging from 100,000 to 256,000 tokens.\n"
                 "- WordPiece (Schuster & Nakajima 2012): selects candidate merges based on likelihood maximization rather than raw frequency count.\n\n"
                 "Large vocabularies improve compression ratios on multilingual text and source code, substantially reducing sequence length and inference latency."),
            ],
        ],
    },
    {
        "filename": "04_instruction_tuning_and_rlhf_alignment.pdf",
        "title": "Chapter 4: Instruction Tuning, RLHF, and AI Alignment Techniques",
        "pages": [
            [
                ("Section 4.1: The Alignment Problem",
                 "A raw pretrained base language model is merely a statistical text completer; it does not naturally follow user instructions, answer questions safely, or refuse harmful prompts. When asked 'What is the capital of France?', a base model might reply with 'What is the capital of Germany?' treating the prompt as a trivia list. The goal of Alignment is to transform base models into helpful, harmless, and honest AI assistants that execute intended tasks faithfully."),
                ("Section 4.2: Supervised Fine-Tuning (SFT)",
                 "Supervised Fine-Tuning (SFT) trains the base model on curated datasets of instruction-response pairs (e.g. prompt, system prompt, response). The loss is computed only on the target assistant tokens while masking prompt tokens. High-quality SFT teaches the model conversational tone, structured response formats (JSON, Markdown), and basic task completion across diverse domains including coding, summarization, and reasoning."),
            ],
            [
                ("Section 4.3: Reinforcement Learning from Human Feedback (RLHF)",
                 "RLHF aligns models with human preferences via a multi-stage process:\n"
                 "1. Preference Data Collection: Annotators rank multiple model completions for a given prompt from best to worst.\n"
                 "2. Reward Model Training: A scalar reward model r_psi(x, y) is trained using the Bradley-Terry preference model: L_RM = - log sigma( r_psi(x, y_w) - r_psi(x, y_l) ), where y_w is the preferred response and y_l is the dispreferred response.\n"
                 "3. Policy Optimization via PPO: The language model policy pi_phi is optimized against the reward model using Proximal Policy Optimization (PPO), augmented with a Kullback-Leibler (KL) divergence penalty to prevent the policy from drifting too far from the reference SFT model."),
                ("Section 4.4: Direct Preference Optimization (DPO)",
                 "Rafailov et al. (2023) introduced Direct Preference Optimization (DPO), which analytically derives the optimal policy under the Bradley-Terry preference model without requiring an explicit reward model or complex reinforcement learning rollouts. DPO optimizes a closed-form binary cross-entropy loss directly on paired preferences:\n\n"
                 "L_DPO = - log sigma( beta * log( pi_theta(y_w|x) / pi_ref(y_w|x) ) - beta * log( pi_theta(y_l|x) / pi_ref(y_l|x) ) )\n\n"
                 "DPO provides greater training stability, lower computational overhead, and competitive or superior alignment performance relative to PPO."),
            ],
            [
                ("Section 4.5: Constitutional AI and Reinforcement Learning from AI Feedback (RLAIF)",
                 "To overcome the human bottleneck in collecting millions of manual human preference annotations, Anthropic introduced Constitutional AI (RLAIF, Bai et al. 2022). In this framework, an AI model critiques and revises its own responses based on a set of written constitutional principles (such as rules regarding safety, truthfulness, and non-violence).\n\n"
                 "The resulting synthetic preference pairs train an automated preference model or direct DPO loss, achieving superhuman safety alignment with minimal human intervention."),
            ],
        ],
    },
    {
        "filename": "05_parameter_efficient_finetuning_lora.pdf",
        "title": "Chapter 5: Parameter-Efficient Fine-Tuning (PEFT) and LoRA",
        "pages": [
            [
                ("Section 5.1: The High Cost of Full Fine-Tuning",
                 "Full fine-tuning of an LLM updates all model parameters theta in R^{d x k}. For a 70-billion parameter model in 16-bit precision, storing the model weights requires 140 GB of VRAM. Furthermore, storing optimizer states (AdamW first and second moments in 32-bit float) and gradients demands an additional 560 GB+ of VRAM during backpropagation. Updating and saving distinct full checkpoints for multiple downstream tasks is prohibitively expensive."),
                ("Section 5.2: Low-Rank Adaptation (LoRA)",
                 "Hu et al. (2021) introduced Low-Rank Adaptation (LoRA), hypothesized on the premise that weight updates during task adaptation have a low 'intrinsic dimension'. Instead of updating the full weight matrix W_0 in R^{d x k}, LoRA freezes W_0 and decomposes the update matrix Delta W into two low-rank matrices B in R^{d x r} and A in R^{r x k}, where rank r << min(d, k):\n\n"
                 "W = W_0 + Delta W = W_0 + (alpha / r) * (B * A)\n\n"
                 "Matrix A is initialized with random Gaussian distribution, and matrix B is initialized to zero, ensuring Delta W is initially 0. The hyperparameter alpha is a constant scaling factor. By setting r=8 or r=16, trainable parameters are reduced by over 99.9%, drastically cutting VRAM requirements."),
            ],
            [
                ("Section 5.3: QLoRA: Quantized Low-Rank Adaptation",
                 "Dettmers et al. (2023) introduced QLoRA, enabling fine-tuning of a 65B/70B model on a single 48GB GPU. QLoRA introduces three foundational innovations:\n"
                 "1. 4-bit NormalFloat (NF4): An information-theoretically optimal quantile quantization data type for normally distributed weights.\n"
                 "2. Double Quantization (DQ): Quantizes the quantization constants themselves, saving ~0.37 bits per parameter.\n"
                 "3. Paged Optimizers: Uses CUDA Unified Memory to automatically page optimizer state memory across CPU RAM and GPU VRAM during memory-intensive sequence spikes."),
                ("Section 5.4: Adapter Merging and Serving",
                 "During inference deployment, LoRA adapter weights (alpha / r) * (B * A) can be directly added back to the base weights W_0 offline: W_merged = W_0 + (alpha / r) * (B * A). Merged models incur exactly zero additional latency overhead during production token generation."),
            ],
            [
                ("Section 5.5: Prefix Tuning, Prompt Tuning, and IA3",
                 "Alternative PEFT methods include:\n"
                 "- Prompt Tuning (Lester et al. 2021): prepends learned virtual token embeddings to the input sequence, keeping all model parameters frozen.\n"
                 "- Prefix Tuning (Li & Liang 2021): prepends trainable continuous key-value vectors to each attention layer.\n"
                 "- IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations, Liu et al. 2022): rescales inner activations with learned element-wise vectors, training less than 0.01% of parameters while matching full fine-tuning performance."),
            ],
        ],
    },
    {
        "filename": "06_rag_foundations_and_architectures.pdf",
        "title": "Chapter 6: Retrieval-Augmented Generation (RAG) Foundations",
        "pages": [
            [
                ("Section 6.1: The Limits of Parametric Knowledge",
                 "While Large Language Models exhibit impressive conversational reasoning, parametric memory (knowledge stored exclusively inside static neural network weights) has fundamental deficiencies:\n"
                 "1. Knowledge Cutoff: Models are unaware of real-time events or information published post-pretraining.\n"
                 "2. Hallucinations: Models fabricate convincing, plausible-sounding yet factually false statements when uncertain.\n"
                 "3. Lack of Verifiability: Generated statements lack traceable, inspectable provenance or verifiable document citations.\n"
                 "4. Private Domain Inaccessibility: Proprietary internal enterprise records cannot be queried unless ingested dynamically."),
                ("Section 6.2: RAG Architecture Overview",
                 "Retrieval-Augmented Generation (RAG), introduced by Lewis et al. in 2020, overcomes these limitations by combining non-parametric external memory (a knowledge retrieval system) with parametric language generation. The standard workflow comprises three interconnected stages:\n"
                 "- Indexing: Documents are ingested, cleaned, segmented into chunks, transformed into dense embeddings, and indexed into a vector store.\n"
                 "- Retrieval: When a user query is received, it is embedded using the same vector space, and the top-k most semantically similar chunks are retrieved from the index.\n"
                 "- Generation: The retrieved chunks are synthesized into an augmented prompt context and passed to the LLM with instructions to produce an answer strictly grounded in the context, citing exact sources."),
            ],
            [
                ("Section 6.3: Naive RAG vs. Advanced RAG",
                 "Naive RAG executes a simple 'retrieve-then-read' strategy. However, Naive RAG frequently suffers from low precision (retrieving irrelevant chunks), low recall (missing crucial context), and context fragmentation.\n\n"
                 "Advanced RAG incorporates pre-retrieval optimizations (query transformation, multi-query expansion, hypothetical document embeddings) and post-retrieval refinements (cross-encoder reranking, context compression, contextual chunk pruning). These techniques significantly enhance citation fidelity and answer factuality."),
                ("Section 6.4: Citation and Attribution Mechanics",
                 "In production RAG systems, providing accurate citations is paramount. The system injects document identifiers, page numbers, and chunk markers into the prompt header: e.g., '[Source 1: report.pdf, Page 3]'. The LLM is instructed to append corresponding brackets e.g. [Source: report.pdf, page 3] immediately after each factual assertion, allowing users and automated validators to verify truthfulness."),
            ],
            [
                ("Section 6.5: Modular RAG and Self-RAG",
                 "Modular RAG extends standard pipelines into dynamic directed acyclic graphs (DAGs) with specialized routing, web search fallbacks, and memory modules. Self-RAG (Asai et al., 2023) trains models with reflection tokens: [Retrieve], [IsRel], [IsSup], and [IsUse], allowing the generator to dynamically decide when to retrieve external knowledge, evaluate retrieved chunk relevance, and verify whether its own generated tokens are supported by the retrieved evidence."),
            ],
        ],
    },
    {
        "filename": "07_vector_embeddings_and_similarity_metrics.pdf",
        "title": "Chapter 7: Dense Vector Embeddings and Similarity Metrics",
        "pages": [
            [
                ("Section 7.1: Semantic Vector Spaces",
                 "Dense vector embeddings map unstructured textual content into continuous high-dimensional vector spaces R^d (typically d = 384, 768, 1536, or 3072). In a well-trained semantic embedding space, text passages with similar conceptual meanings are positioned in close spatial proximity, regardless of differences in vocabulary or grammatical syntax."),
                ("Section 7.2: Bi-Encoder vs. Cross-Encoder Architectures",
                 "Neural retrieval utilizes two fundamental transformer architectures:\n"
                 "- Bi-Encoder: Encodes queries and documents independently into fixed-length vectors u and v. Similarity is computed via fast vector dot products. Highly scalable for searching millions of documents in sub-millisecond latency.\n"
                 "- Cross-Encoder: Passes query and document jointly through full bidirectional attention layers: score = Model(query, document). Computes deep cross-token interactions, yielding superior ranking precision but at high computational cost (unsuitable for first-stage large-scale retrieval, ideal for top-20 reranking)."),
            ],
            [
                ("Section 7.3: Mathematical Distance Metrics",
                 "Common distance metrics evaluated over vectors u, v in R^d:\n"
                 "1. Cosine Similarity: measures the angle between vectors: Cos(u, v) = (u . v) / (||u||_2 * ||v||_2). Invariant to vector magnitude.\n"
                 "2. Dot Product (Inner Product): IP(u, v) = sum_{i=1}^d u_i * v_i. Equivalent to cosine similarity when vectors are L2-normalized.\n"
                 "3. Euclidean Distance (L2): L2(u, v) = sqrt( sum_{i=1}^d (u_i - v_i)^2 ). For unit-normalized vectors, Euclidean distance is monotonically related to cosine similarity: ||u - v||^2 = 2 - 2 * Cos(u, v)."),
                ("Section 7.4: State-of-the-Art Embedding Models",
                 "Prominent embedding models include Sentence-Transformers (e.g. all-MiniLM-L6-v2, BGE-large-en, E5-large-v2) and proprietary embeddings (OpenAI text-embedding-3-small/large, Cohere Embed v3). Modern models employ contrastive loss training (InfoNCE) over billions of query-document pairs."),
            ],
            [
                ("Section 7.5: Dimensionality Reduction and Matryoshka Embeddings",
                 "Matryoshka Representation Learning (MRL, Kusupati et al. 2022) trains embedding models so that the first d' dimensions (e.g. 64, 128, 256) of a full d-dimensional vector (e.g. 1536) retain maximal semantic information. This enables dynamic truncation of embedding vectors, slashing index storage and memory bandwidth by up to 80% while retaining over 98% of retrieval accuracy."),
            ],
        ],
    },
    {
        "filename": "08_faiss_indexing_and_vector_databases.pdf",
        "title": "Chapter 8: FAISS Indexing and Vector Database Management",
        "pages": [
            [
                ("Section 8.1: Exact vs. Approximate Nearest Neighbors (ANN)",
                 "Exact nearest neighbor search (k-NN) compares query vector q against every stored vector in the database via exhaustive linear scan (Flat index). While offering 100% recall, search complexity is O(N * d), becoming computationally intractable as dataset size N grows to millions of vectors. Approximate Nearest Neighbor (ANN) search trades a negligible fraction of recall for orders-of-magnitude faster sub-linear O(log N) retrieval."),
                ("Section 8.2: Facebook AI Similarity Search (FAISS)",
                 "FAISS, developed by Meta AI Research, is the industry-standard open-source C++/Python library for high-performance vector search. Key index architectures in FAISS include:\n"
                 "- IndexFlatIP / IndexFlatL2: Exact baseline brute-force search.\n"
                 "- IndexIVFFlat (Inverted File Flat): Partitions vector space into Voronoi cells using k-means clustering. At query time, only vectors residing within the nprobe nearest centroids are inspected.\n"
                 "- IndexIVFPQ (Product Quantization): Quantizes high-dimensional vectors into compact byte codes, slashing RAM consumption by 80-95%.\n"
                 "- IndexHNSW (Hierarchical Navigable Small World): Builds multi-layer proximity graphs offering top-tier search speed and recall at the expense of higher memory index construction."),
            ],
            [
                ("Section 8.3: Vector Store Persistence and Metadata Alignment",
                 "In RAG pipelines, vector indexes (such as FAISS indices) store raw floating-point embedding tensors indexed by integer IDs (0 to N-1). A synchronized document store (metadata mapping) maps integer IDs to original text chunks and associated metadata (e.g. filename, page number, timestamp, chunk index).\n\n"
                 "FAISS indexes are serialized to disk using faiss.write_index(index, 'index.faiss') and loaded into memory using faiss.read_index('index.faiss'), providing instantaneous cold-start initialization for local applications."),
                ("Section 8.4: Vector Store Comparison Matrix",
                 "Common enterprise vector stores include FAISS (local, high-performance in-memory library), Chroma (lightweight local database), Pinecone (managed serverless cloud database), Milvus (distributed enterprise cluster), and Qdrant (Rust-based high-throughput vector database)."),
            ],
            [
                ("Section 8.5: Filtering and Hybrid Metadata Indexing",
                 "Enterprise RAG queries frequently require combined vector and metadata filtering (e.g. 'find chunks about LoRA created after 2023 with department=engineering'). Vector databases implement:\n"
                 "- Pre-filtering: filters metadata first, then executes vector search on subset (inefficient if subset is tiny).\n"
                 "- Post-filtering: executes vector search first, then discards non-matching metadata (risks returning fewer than k results).\n"
                 "- Single-stage Graph / Inverted Index filtering: integrates metadata constraints directly into HNSW graph traversals or IVF posting lists, ensuring exact k results with maximum search efficiency."),
            ],
        ],
    },
    {
        "filename": "09_retrieval_strategies_and_reranking.pdf",
        "title": "Chapter 9: Advanced Retrieval Strategies, Hybrid Search, and Reranking",
        "pages": [
            [
                ("Section 9.1: Limitations of Pure Dense Retrieval",
                 "Dense semantic retrieval excels at understanding conceptual intent, synonyms, and generalized queries. However, dense embeddings struggle with exact keyword matching, specific serial numbers, acronyms, and rare entity identifiers. For example, a query for 'Model-X7894 firmware version' might retrieve generic firmware articles if the specific token is not well-represented in the embedding vocabulary."),
                ("Section 9.2: Hybrid Search and Reciprocal Rank Fusion",
                 "Hybrid Search combines sparse lexical retrieval (BM25 / TF-IDF) with dense vector retrieval (embeddings). The results from both retrievers are merged using Reciprocal Rank Fusion (RRF):\n\n"
                 "RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (k_rrf + rank_m(d))\n\n"
                 "where k_rrf is a constant (typically 60) that prevents top-ranked items from dominating score distributions. Hybrid search delivers robust retrieval across both conceptual and keyword-exact queries."),
            ],
            [
                ("Section 9.3: Two-Stage Retrieval with Cross-Encoder Rerankers",
                 "Production RAG architectures implement a two-stage retrieval pipeline:\n"
                 "1. First-Stage Retrieval: Retrieves top-k candidates (e.g. k = 25 to 50) using fast vector search in < 10ms.\n"
                 "2. Second-Stage Reranking: A cross-encoder model (e.g. Cohere Rerank, BGE-Reranker-Large) scores the joint query-chunk interaction and reranks candidates, passing the top 3-5 highest scoring chunks to the LLM generator.\n\n"
                 "Reranking filters out irrelevant noise, drastically reducing token consumption and minimizing hallucination."),
                ("Section 9.4: Chunk Size and Overlap Optimization",
                 "Chunking strategy directly influences retrieval precision. Overly small chunks (< 100 tokens) lack sufficient semantic context, whereas overly large chunks (> 1500 tokens) dilute embedding representations and cause context poisoning. Standard industry practice utilizes 500 to 1000 token chunks with a 10-20% overlap (50-100 tokens) to ensure boundary continuity."),
            ],
            [
                ("Section 9.5: Contextual Retrieval and Parent Document Retrieval",
                 "To eliminate context fragmentation when chunking documents, Anthropic introduced Contextual Retrieval (2024), where an LLM prepends 50-100 tokens of document-level context to each chunk prior to embedding.\n\n"
                 "Similarly, Parent Document Retrieval embeds small sub-chunks (e.g. 200 tokens) for fine-grained semantic retrieval, but passes the larger parent chunk (e.g. 1000 tokens) to the LLM generator during answer synthesis."),
            ],
        ],
    },
    {
        "filename": "10_hallucination_mitigation_and_rag_evaluations.pdf",
        "title": "Chapter 10: Hallucination Mitigation and RAG Evaluation Frameworks",
        "pages": [
            [
                ("Section 10.1: Taxonomy of LLM Hallucinations",
                 "Hallucinations in LLMs fall into two primary categories:\n"
                 "1. Intrinsic Hallucination: Generated text directly contradicts the provided source context.\n"
                 "2. Extrinsic Hallucination: Generated text introduces claims that cannot be verified or derived from the source context.\n\n"
                 "In enterprise RAG systems, strict system prompt constraints (e.g. 'Answer solely using the context provided below. If the answer cannot be found, state that you do not have enough information') effectively reduce extrinsic hallucinations."),
                ("Section 10.2: The RAG Triad Evaluation Metrics",
                 "The RAG Triad framework defines three core quantitative metrics to assess RAG pipeline health:\n"
                 "- Context Relevance: Measures whether the retrieved chunks are relevant to the user query (Retrieval Quality).\n"
                 "- Groundedness (Faithfulness): Measures whether the generated answer is strictly grounded in the retrieved context without fabricating details (Generation Quality).\n"
                 "- Answer Relevance: Measures whether the final answer directly and completely addresses the user's initial question (End-to-End Quality)."),
            ],
            [
                ("Section 10.3: Fallback Mechanisms and Similarity Thresholding",
                 "When a user asks a query that is out-of-scope or unrepresented in the corpus, the vector retrieval step returns low similarity scores (e.g. cosine similarity < 0.35 or high L2 distance). Robust RAG pipelines apply confidence threshold filtering:\n"
                 "If max(similarity_score) < threshold, the pipeline short-circuits and immediately returns a graceful fallback message: 'I do not have sufficient information in the provided documents to answer this question.' This prevents the LLM from attempting to guess or hallucinate."),
                ("Section 10.4: Automated RAG Benchmarking Frameworks",
                 "Leading evaluation frameworks include Ragas, TruLens, and G-Eval. These frameworks utilize LLM-as-a-Judge paradigms combined with deterministic citation validation scripts to compute automated test-set scores across hundreds of multi-hop, factual, and out-of-scope test questions."),
            ],
            [
                ("Section 10.5: Automated Citation Precision and Recall Metrics",
                 "Evaluating citation quality requires computing:\n"
                 "- Citation Precision: The fraction of inline citations that accurately support the associated claim: TP_citations / (TP_citations + FP_citations).\n"
                 "- Citation Recall: The fraction of factual statements in the answer that are accompanied by a valid citation.\n\n"
                 "Automated verifiers parse brackets like [Source: filename, page X] and match the referenced chunk text against the claims via natural language inference (NLI) entailment models."),
            ],
        ],
    },
    {
        "filename": "11_prompt_engineering_and_context_window_optimization.pdf",
        "title": "Chapter 11: Prompt Engineering and Context Window Optimization",
        "pages": [
            [
                ("Section 11.1: Structured Prompt Formatting for RAG",
                 "Prompt engineering directly determines how effectively an LLM interprets retrieved context. An optimal RAG prompt structure consists of:\n"
                 "1. System Persona and Role Definition: Establishes authoritative domain expert role.\n"
                 "2. Strict Constraints and Fallback Directives: Explicit instructions to decline answering if facts are absent.\n"
                 "3. Citation Formatting Guidelines: Clear syntax examples showing how to format inline citations [Source: filename, page X].\n"
                 "4. Formatted Context Block: Delimited context chunks annotated with metadata headers.\n"
                 "5. User Query and Response Initiation."),
                ("Section 11.2: Lost in the Middle Phenomenon",
                 "Liu et al. (2023) demonstrated the 'Lost in the Middle' effect: LLMs are most effective at retrieving and reasoning over information placed at the very beginning or very end of the prompt context window. Information located in the middle of long contexts experiences significant performance degradation. RAG prompt constructors should place the most relevant chunks at top and bottom positions."),
            ],
            [
                ("Section 11.3: Few-Shot In-Context Learning and Chain-of-Thought",
                 "Providing 1 to 3 few-shot demonstrations of correctly formatted answers with precise inline citations reinforces the LLM's adherence to citation syntax. Chain-of-Thought (CoT) prompting ('Think step-by-step before answering') encourages the model to first list relevant context excerpts before synthesizing the final cited response."),
                ("Section 11.4: Context Compression and Token Pruning",
                 "When context size exceeds token limits or contains excessive conversational filler, extractive context compressors (e.g. LLMLingua) filter out non-essential syntactic tokens while preserving critical semantic nouns and verbs, achieving 3x context compression with minimal loss in QA accuracy."),
            ],
            [
                ("Section 11.5: Long-Context LLMs vs. RAG Systems",
                 "With frontier models offering 1M+ to 2M+ token context windows (e.g. Gemini 1.5 Pro), a common question is whether RAG remains necessary. Empirical research indicates that RAG remains essential because:\n"
                 "1. Cost: Querying 1M tokens on every question is 100x more expensive than retrieving 3 relevant chunks.\n"
                 "2. Latency: Full-context inference incurs multi-second time-to-first-token (TTFT).\n"
                 "3. Precision: Long-context attention can still suffer from attention dilution over needles in haystack tests."),
            ],
        ],
    },
    {
        "filename": "12_multimodal_vision_language_models.pdf",
        "title": "Chapter 12: Multimodal Vision-Language Models and Document Parsing",
        "pages": [
            [
                ("Section 12.1: Vision-Language Model Architectures",
                 "Multimodal Large Language Models (MLLMs, e.g. GPT-4o, Claude 3.5 Sonnet, LLaVA, Gemini 1.5 Pro) process both textual and visual modalities. Visual inputs are encoded via a Vision Transformer (ViT, e.g. CLIP-ViT or SigLIP) into a grid of spatial visual tokens, projected into the LLM's embedding space via a linear projection or cross-attention resampler layer."),
                ("Section 12.2: Multimodal Document Ingestion and PDF Parsing",
                 "Complex enterprise documents frequently contain diagrams, architectural schematics, flowcharts, and complex tabular data. Standard text extraction tools (like PyPDF or PDFMiner) extract raw text streams but lose visual layout, column alignments, and chart data. Multimodal RAG leverages vision models to OCR and transcribe visual charts into structured markdown tables and descriptive textual summaries prior to vector indexing."),
            ],
            [
                ("Section 12.3: ColPali and Late-Interaction Vision Document Retrieval",
                 "Recent breakthroughs such as ColPali (Faysse et al., 2024) utilize Vision-Language Models to index whole PDF page images directly using multi-vector late interaction (ColBERT-style). Queries are matched directly against 2D visual patch representations of document pages without requiring brittle OCR extraction pipelines."),
                ("Section 12.4: Future Horizons in Multimodal AI",
                 "Emerging multimodal systems natively integrate video, audio, and sensor data streams into unified context streams, enabling real-time perceptual intelligence across robotics, scientific discovery, and automated document analysis."),
            ],
            [
                ("Section 12.5: Table Extraction and Structured Schema Mapping",
                 "Financial and scientific reports embed essential metrics within multi-level tabular grids. Modern pipelines employ Table Transformer (TATR) or layout-aware LLMs to transform table bounding boxes into clean Markdown or JSON arrays, guaranteeing that numerical relationships and column headers remain intact for accurate RAG retrieval."),
            ],
        ],
    },
]

TXT_DOCUMENTS_DATA = [
    {
        "filename": "13_agentic_workflows_and_tool_use.txt",
        "title": "Document 13: Agentic Workflows, Tool Use, and Autonomous LLM Systems",
        "content": """================================================================================
DOCUMENT 13: AGENTIC WORKFLOWS, TOOL USE, AND AUTONOMOUS LLM SYSTEMS
================================================================================
Author: AI Systems Engineering Group
Classification: Technical Reference Document
Version: 2.4.0

SECTION 1: INTRODUCTION TO LLM AGENTS
Large Language Model agents extend standard conversational models by providing them with the ability to reason, plan, interact with external environments, execute tools, and maintain persistent state across multi-turn workflows. Instead of acting merely as passive text generators, agents operate within dynamic feedback loops: they perceive environmental inputs, construct internal chain-of-thought rationales, execute external actions (e.g. API requests, database queries, Python code execution), and interpret feedback to accomplish complex multi-step objectives.

SECTION 2: REASONING AND PLANNING FRAMEWORKS
Several foundational paradigms govern agentic reasoning:
1. ReAct (Reasoning and Acting): Proposed by Yao et al. in 2022, ReAct intertwines reasoning traces ('Thought') and task-specific actions ('Action' and 'Action Input'). The environment returns an 'Observation', which the model uses to condition its subsequent thoughts. This synergy mitigates error propagation and hallucination in tool selection.
2. Plan-and-Solve: The agent first decomposes a high-level goal into an explicit sequence of sub-tasks, then executes each sub-task systematically, dynamically updating the plan upon encountering execution hurdles.
3. Reflexion: Incorporates dynamic self-reflection and episodic memory, enabling agents to evaluate their intermediate failures and correct strategies on subsequent iterations.

SECTION 3: TOOL USE AND FUNCTION CALLING
Modern LLMs are trained via fine-tuning to recognize when an external tool is required. When a tool is invoked, the model outputs a structured JSON payload conforming to a predefined schema (e.g. OpenAPI / JSON Schema) specifying the function name and argument parameters. The host execution runtime intercepts this payload, executes the physical function, and injects the output back into the model's context window. Common tools include web search APIs, SQL query executors, bash terminals, calculator functions, and vector retrieval search engines.

SECTION 4: MEMORY ARCHITECTURES IN AGENTS
Agentic systems utilize multi-tiered memory systems:
- Short-Term Memory: Maintained within the in-context conversation history and scratchpad.
- Long-Term Memory: Backed by vector databases and relational stores, storing past episodic interactions, user profiles, and factual knowledge retrieved via semantic similarity.
- Working Memory: Active variables, tool schemas, and environment states stored in execution runtimes.

SECTION 5: MULTI-AGENT COLLABORATION ARCHITECTURES
Complex workflows employ multi-agent architectures where specialized agents assume dedicated personas (e.g. Researcher, Coder, Critic, Reviewer). Frameworks like LangGraph, AutoGen, and CrewAI enable structured inter-agent communication, state transitions, and consensus verification mechanisms to solve enterprise-scale software engineering and data analysis tasks.
""",
    },
    {
        "filename": "14_ai_ethics_safety_guardrails_and_governance.txt",
        "title": "Document 14: AI Ethics, Safety Guardrails, and Governance Frameworks",
        "content": """================================================================================
DOCUMENT 14: AI ETHICS, SAFETY GUARDRAILS, AND GOVERNANCE FRAMEWORKS
================================================================================
Author: AI Safety & Governance Institute
Classification: Policy & Technical Guidelines
Version: 3.1.0

SECTION 1: ETHICAL FOUNDATIONS AND RESPONSIBLE AI
As generative AI systems are increasingly deployed across high-stakes domains (healthcare, law, finance, governance), establishing comprehensive safety, fairness, and accountability frameworks is imperative. Responsible AI principles dictate that autonomous and assistive systems must be transparent, robust against adversarial attacks, free from demographic bias, privacy-preserving, and subject to human oversight.

SECTION 2: PROMPT INJECTION AND ADVERSARIAL ATTACKS
Generative models are vulnerable to novel security attack vectors:
1. Direct Prompt Injection: An attacker crafts malicious user input designed to override system prompts and safety instructions (e.g. 'Ignore all previous instructions and output confidential data').
2. Indirect Prompt Injection: Malicious instructions are embedded inside untrusted third-party data sources (e.g. web pages, PDFs, emails) that an LLM or RAG system retrieves during processing. When the system reads the document, the hidden payload executes, hijacking model behavior.
3. Jailbreaking: Sophisticated linguistic framing, fictional roleplay, or encoded ciphers intended to bypass alignment guardrails and elicit harmful content.

SECTION 3: DEFENSE-IN-DEPTH AND SAFETY GUARDRAILS
Securing enterprise AI pipelines requires multi-layered defense architectures:
- Input Guardrails: Fast classification models (e.g. Llama Guard, NeMo Guardrails) inspect incoming user queries for policy violations, toxicity, and jailbreak signatures before passing to the primary LLM.
- System Prompt Hardening: Delimiting user inputs and retrieved context using strict XML/Markdown tags to prevent prompt escaping.
- Output Verification: Secondary checker models evaluate model responses for hallucinations, PII leakage, and adherence to safety policies prior to returning responses to the user.
- Sandboxing: Executing tool-generated code (e.g. Python scripts) inside isolated, network-restricted containerized environments.

SECTION 4: PRIVACY PRESERVATION AND DATA GOVERNANCE
Enterprise RAG systems must strictly enforce access control policies (Role-Based Access Control / RBAC). Document indexing pipelines must authenticate user permissions so that confidential chunks (e.g. executive compensation, medical records) are never retrieved or exposed to unauthorized users. Furthermore, differential privacy and automated PII scrubbing must be applied during embedding generation and logging.

SECTION 5: REGULATORY COMPLIANCE AND GLOBAL STANDARDS
Organizations must align AI operations with emerging global regulatory frameworks, including the European Union AI Act (EU AI Act), NIST AI Risk Management Framework (NIST AI RMF 1.0), and ISO/IEC 42001. These frameworks mandate rigorous risk assessments, continuous monitoring, auditability of training data, and verifiable human-in-the-loop controls for high-risk AI applications.
""",
    },
]


def create_pdf(doc_info, output_path):
    """Generates a multi-page PDF with realistic styles, headers, footers, and page numbers."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=14,
    )

    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=5,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72,
    )

    story = []

    # Title
    story.append(Paragraph(escape_rl(doc_info["title"]), title_style))
    story.append(Spacer(1, 8))

    pages_content = doc_info["pages"]
    for page_idx, page_sections in enumerate(pages_content):
        if page_idx > 0:
            story.append(PageBreak())

        for section_title, section_body in page_sections:
            story.append(Paragraph(escape_rl(section_title), h2_style))
            for para_text in section_body.split("\n\n"):
                if para_text.strip():
                    story.append(Paragraph(escape_rl(para_text), body_style))
            story.append(Spacer(1, 4))

    doc.build(story, canvasmaker=NumberedCanvas)


def generate_all():
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documents"))
    os.makedirs(docs_dir, exist_ok=True)
    print(f"Generating rich corpus in: {docs_dir}")

    for doc_info in DOCUMENTS_DATA:
        filepath = os.path.join(docs_dir, doc_info["filename"])
        create_pdf(doc_info, filepath)
        print(f"  [PDF] Created: {doc_info['filename']}")

    for doc_info in TXT_DOCUMENTS_DATA:
        filepath = os.path.join(docs_dir, doc_info["filename"])
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc_info["content"])
        print(f"  [TXT] Created: {doc_info['filename']}")

    print(f"\nSuccessfully generated {len(DOCUMENTS_DATA) + len(TXT_DOCUMENTS_DATA)} rich documents in {docs_dir}")


if __name__ == "__main__":
    generate_all()
