# Architectural Diagrams

Mermaid visualizations of the system architecture.

> **Note:** GitHub automatically renders Mermaid diagrams. To view them locally, use an editor that supports Mermaid (VS Code with extension, Obsidian, etc.)

---

## General System Architecture

```mermaid
flowchart TB
    subgraph INPUT["INPUT"]
        CONFIG["config.yaml"]
        DATASETS["Datasets<br/>HuggingFace"]
        DOCS["Documents<br/>Knowledge Base"]
    end

    subgraph BRAIN["BRAIN Central System"]
        subgraph LOADER["Model Loader"]
            HF["HuggingFace<br/>Model"]
            QUANT["QLoRA<br/>4-bit Quantization"]
            LORA["LoRA<br/>Adapters"]
        end
        
        subgraph MEMORY["Memory System"]
            RAG["RAG<br/>VectorStore"]
            CHUNK["Smart<br/>Chunker"]
            SOP["SOP<br/>Manager"]
            RERANK["Reranker<br/>CrossEncoder"]
        end
        
        subgraph TRAINING["Training Engine"]
            AGENT["Agent<br/>Lightning"]
            GRPO["GRPO<br/>RL Algorithm"]
            REWARD["Reward<br/>Functions"]
            SFT["SFT<br/>Supervised"]
        end
    end

    subgraph OUTPUT["OUTPUT"]
        CKPT["Checkpoint<br/>LoRA Adapter"]
        LOGS["TensorBoard<br/>Logs"]
        MODEL["Fine-tuned<br/>Model"]
    end

    CONFIG --> LOADER
    DATASETS --> TRAINING
    DOCS --> MEMORY

    HF --> QUANT --> LORA
    LORA --> TRAINING
    
    MEMORY --> TRAINING
    RAG <--> CHUNK
    RAG <--> RERANK
    SOP --> TRAINING
    
    AGENT --> GRPO
    AGENT --> SFT
    GRPO --> REWARD
    
    TRAINING --> CKPT
    TRAINING --> LOGS
    CKPT --> MODEL

    style BRAIN fill:#1a1a2e,stroke:#16213e,color:#fff
    style MEMORY fill:#0f3460,stroke:#16213e,color:#fff
    style TRAINING fill:#533483,stroke:#16213e,color:#fff
    style LOADER fill:#e94560,stroke:#16213e,color:#fff
```

---

## Complete Training Workflow

```mermaid
flowchart LR
    subgraph INIT["1. Initialization"]
        A1["Load Config"] --> A2["Load Model"]
        A2 --> A3["Apply QLoRA"]
        A3 --> A4["Apply LoRA"]
    end

    subgraph DATA["2. Data Preparation"]
        B1["Load Datasets"] --> B2["Multi-Source<br/>Mixing"]
        B2 --> B3["Format<br/>ChatML"]
        B3 --> B4["Tokenize"]
    end

    subgraph TRAIN["3. Training Loop"]
        C1["Batch"] --> C2["Forward<br/>Pass"]
        C2 --> C3{"Algorithm?"}
        C3 -->|SFT| C4["Cross-Entropy<br/>Loss"]
        C3 -->|GRPO| C5["Generate K<br/>Responses"]
        C5 --> C6["Compute<br/>Rewards"]
        C6 --> C7["Normalize<br/>Advantages"]
        C7 --> C8["Policy<br/>Gradient Loss"]
        C4 --> C9["Backward"]
        C8 --> C9
        C9 --> C10["Update<br/>Weights"]
        C10 --> C1
    end

    subgraph SAVE["4. Output"]
        D1["Save<br/>Adapter"]
        D2["Log<br/>Metrics"]
        D3["Test<br/>Generation"]
    end

    INIT --> DATA --> TRAIN --> SAVE

    style INIT fill:#2d3436,stroke:#636e72,color:#fff
    style DATA fill:#0984e3,stroke:#74b9ff,color:#fff
    style TRAIN fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style SAVE fill:#00b894,stroke:#55efc4,color:#fff
```

---

## GRPO (Reinforcement Learning) Flow

```mermaid
flowchart TD
    START["Start"] --> PROMPT["Input Prompt"]
    
    PROMPT --> GEN["Generate K Responses<br/>(temperature > 0)"]
    
    GEN --> R1["Response 1"]
    GEN --> R2["Response 2"]
    GEN --> R3["Response 3"]
    GEN --> RK["Response K"]
    
    R1 --> REW1["Reward: 0.85"]
    R2 --> REW2["Reward: 0.42"]
    R3 --> REW3["Reward: 0.91"]
    RK --> REWK["Reward: 0.67"]
    
    REW1 --> NORM["Normalize Rewards<br/>Advantage = (R - mean) / std"]
    REW2 --> NORM
    REW3 --> NORM
    REWK --> NORM
    
    NORM --> ADV1["A1 = +0.52"]
    NORM --> ADV2["A2 = -1.23"]
    NORM --> ADV3["A3 = +0.89"]
    NORM --> ADVK["AK = -0.18"]
    
    ADV1 --> LOSS["Policy Gradient Loss<br/>L = -sum(Ai * log(pi(yi|x)))"]
    ADV2 --> LOSS
    ADV3 --> LOSS
    ADVK --> LOSS
    
    LOSS --> KL["KL Penalty<br/>L_total = L + beta * KL(pi|pi_ref)"]
    
    KL --> UPDATE["Update Policy"]
    
    UPDATE --> CHECK{"More<br/>prompts?"}
    CHECK -->|Yes| PROMPT
    CHECK -->|No| END["Done"]

    style START fill:#00b894,stroke:#00cec9,color:#fff
    style END fill:#00b894,stroke:#00cec9,color:#fff
    style GEN fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style NORM fill:#fdcb6e,stroke:#f39c12,color:#000
    style LOSS fill:#e17055,stroke:#d63031,color:#fff
    style KL fill:#e84393,stroke:#fd79a8,color:#fff
```

---

## Memory System (RAG + SOP)

```mermaid
flowchart TB
    subgraph USER["User Query"]
        Q["How do I debug this code"]
    end

    subgraph ROUTER["Query Router"]
        DETECT["Detect Intent"]
        DETECT --> IS_RAG{"Needs<br/>Knowledge?"}
        DETECT --> IS_SOP{"Needs<br/>Procedure?"}
    end

    subgraph RAG_SYSTEM["RAG System"]
        EMBED["Embed Query<br/>(Sentence Transformer)"]
        SEARCH["Vector Search<br/>(ChromaDB)"]
        RERANK["Rerank<br/>(CrossEncoder)"]
        CTX["Context"]
        
        EMBED --> SEARCH --> RERANK --> CTX
    end

    subgraph SOP_SYSTEM["SOP System"]
        MATCH["Trigger Matching"]
        SELECT["Select Best SOP"]
        FORMAT["Format Steps"]
        PROC["Procedure"]
        
        MATCH --> SELECT --> FORMAT --> PROC
    end

    subgraph LLM["LLM Generation"]
        SYSTEM["System Prompt<br/>+ Context + SOP"]
        GENERATE["Generate<br/>Response"]
        RESPONSE["Final Response"]
        
        SYSTEM --> GENERATE --> RESPONSE
    end

    Q --> ROUTER
    IS_RAG -->|Yes| RAG_SYSTEM
    IS_SOP -->|Yes| SOP_SYSTEM
    
    CTX --> SYSTEM
    PROC --> SYSTEM
    
    IS_RAG -->|No| SYSTEM
    IS_SOP -->|No| SYSTEM

    style USER fill:#74b9ff,stroke:#0984e3,color:#000
    style RAG_SYSTEM fill:#0f3460,stroke:#16213e,color:#fff
    style SOP_SYSTEM fill:#533483,stroke:#16213e,color:#fff
    style LLM fill:#00b894,stroke:#00cec9,color:#fff
```

---

## Smart Chunking Pipeline

```mermaid
flowchart LR
    subgraph INPUT["Source Code"]
        CODE["def calculate(x):<br/>    Docstring<br/>    return x * 2"]
    end

    subgraph PARSE["AST Parsing"]
        TS["tree-sitter<br/>Parser"]
        AST["Abstract<br/>Syntax Tree"]
        TS --> AST
    end

    subgraph EXTRACT["Extraction"]
        FUNC["Functions"]
        CLASS["Classes"]
        METHOD["Methods"]
        IMPORT["Imports"]
    end

    subgraph PROCESS["Processing"]
        SIZE{"Size<br/>Check"}
        SPLIT["Split Large"]
        MERGE["Merge Small"]
        SIZE -->|greater than max| SPLIT
        SIZE -->|less than min| MERGE
        SIZE -->|OK| KEEP["Keep"]
    end

    subgraph OUTPUT["Code Chunks"]
        C1["Chunk 1<br/># Function: calc<br/>def calc..."]
        C2["Chunk 2<br/># Class: Model<br/>class Model..."]
        C3["Chunk 3<br/># Method: forward<br/>def forward..."]
    end

    CODE --> PARSE
    AST --> FUNC
    AST --> CLASS
    AST --> METHOD
    AST --> IMPORT
    FUNC --> PROCESS
    CLASS --> PROCESS
    METHOD --> PROCESS
    IMPORT --> PROCESS
    SPLIT --> OUTPUT
    MERGE --> OUTPUT
    KEEP --> OUTPUT

    style PARSE fill:#e17055,stroke:#d63031,color:#fff
    style EXTRACT fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style PROCESS fill:#fdcb6e,stroke:#f39c12,color:#000
    style OUTPUT fill:#00b894,stroke:#00cec9,color:#fff
```

---

## RAG with Reranking (Two-Phase Retrieval)

```mermaid
flowchart TD
    Q["Query: How does GRPO work?"]
    
    subgraph PHASE1["Phase 1: Recall (Bi-Encoder)"]
        EMB["Embed Query<br/>all-MiniLM-L6-v2"]
        VEC["Vector Search<br/>Cosine Similarity"]
        TOP20["Top-20 Candidates"]
        
        EMB --> VEC --> TOP20
    end
    
    subgraph PHASE2["Phase 2: Precision (Cross-Encoder)"]
        PAIRS["Create Pairs<br/>query doc1 query doc2"]
        SCORE["Score Each Pair<br/>CrossEncoder"]
        SORT["Sort by Score"]
        TOP3["Top-3 Results"]
        
        PAIRS --> SCORE --> SORT --> TOP3
    end
    
    subgraph RESULTS["Final Results"]
        R1["[0.923] agent_lightning_trainer.py<br/>GRPO algorithm RL..."]
        R2["[0.871] README.md<br/>The system includes reward..."]
        R3["[0.834] RAG_E_SOP.md<br/>Agent Lightning enables..."]
    end
    
    Q --> PHASE1
    TOP20 --> PHASE2
    TOP3 --> RESULTS

    style PHASE1 fill:#0984e3,stroke:#74b9ff,color:#fff
    style PHASE2 fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style RESULTS fill:#00b894,stroke:#55efc4,color:#fff
```

---

## Layer Architecture

```mermaid
graph TB
    subgraph L4["Layer 4: Entry Points"]
        MAIN["main.py<br/>PyTorch Lightning"]
        MAIN_AL["main_agent_lightning.py<br/>Agent Lightning RL"]
    end

    subgraph L3["Layer 3: Training"]
        AGENT["LLMTrainingAgent<br/>LightningModule"]
        AL_TRAINER["AgentLightningTrainer<br/>GRPO / APO / SFT"]
        REWARD["RewardFunction<br/>coding / chat / fc"]
    end

    subgraph L2["Layer 2: Core"]
        LOADER["ModelLoader<br/>QLoRA + PEFT"]
        DATA["DataModule<br/>Multi-Source"]
        MEMORY["Memory System<br/>RAG + SOP"]
    end

    subgraph L1["Layer 1: Infrastructure"]
        HF["Transformers"]
        TORCH["PyTorch"]
        CHROMA["ChromaDB"]
        TREESIT["tree-sitter"]
    end

    L4 --> L3
    L3 --> L2
    L2 --> L1

    MAIN --> AGENT
    MAIN_AL --> AL_TRAINER
    AL_TRAINER --> REWARD
    
    AGENT --> LOADER
    AGENT --> DATA
    AL_TRAINER --> LOADER
    AL_TRAINER --> DATA
    AL_TRAINER --> MEMORY
    
    LOADER --> HF
    LOADER --> TORCH
    MEMORY --> CHROMA
    MEMORY --> TREESIT

    style L4 fill:#e84393,stroke:#fd79a8,color:#fff
    style L3 fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style L2 fill:#0984e3,stroke:#74b9ff,color:#fff
    style L1 fill:#2d3436,stroke:#636e72,color:#fff
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as Config
    participant M as Model
    participant D as Data
    participant T as Trainer
    participant R as Reward
    participant O as Output

    U->>C: Load config.yaml
    C->>M: Initialize model
    M->>M: Apply QLoRA (4-bit)
    M->>M: Apply LoRA adapters
    
    C->>D: Load datasets
    D->>D: Multi-source mixing
    D->>D: Format to ChatML
    D->>D: Tokenize
    
    M->>T: Pass model
    D->>T: Pass dataloader
    
    loop Training Loop
        T->>T: Get batch
        T->>M: Forward pass
        
        alt GRPO Mode
            M->>T: Generate K responses
            T->>R: Compute rewards
            R->>T: Return rewards
            T->>T: Compute advantages
            T->>T: Policy gradient + KL
        else SFT Mode
            T->>T: Cross-entropy loss
        end
        
        T->>M: Backward pass
        T->>M: Update weights
    end
    
    T->>O: Save LoRA adapter
    T->>O: Save config
    O->>U: Training complete!
```

---

## Reward Function Decision Tree

```mermaid
flowchart TD
    INPUT["(prompt, generation)"]
    
    DETECT{"Detect<br/>Task Type"}
    
    DETECT -->|contains code python| CODING
    DETECT -->|contains function_call tool| FC
    DETECT -->|other| CHAT
    
    subgraph CODING["Coding Reward"]
        C1["Extract code blocks"]
        C2{"Syntax<br/>valid?"}
        C2 -->|Yes| C3["+0.3"]
        C2 -->|No| C4["-0.3"]
        C5{"Has<br/>docstring?"}
        C5 -->|Yes| C6["+0.1"]
        C7{"Has type<br/>hints?"}
        C7 -->|Yes| C8["+0.1"]
        C9{"Length<br/>OK?"}
        C9 -->|50-2000 chars| C10["+0.1"]
        C9 -->|less than 20| C11["-0.2"]
    end
    
    subgraph FC["Function Calling Reward"]
        F1["Find function_call pattern"]
        F2{"JSON<br/>valid?"}
        F2 -->|Yes| F3["+0.3"]
        F2 -->|No| F4["-0.3"]
        F5{"Has name +<br/>arguments?"}
        F5 -->|Yes| F6["+0.2"]
        F7{"Tool<br/>exists?"}
        F7 -->|Yes| F8["+0.2"]
        F7 -->|No| F9["-0.2"]
    end
    
    subgraph CHAT["Chat Reward"]
        H1["Check length"]
        H2{"Length<br/>greater than 50?"}
        H2 -->|Yes| H3["+0.1"]
        H2 -->|less than 10| H4["-0.5"]
        H5{"Unique<br/>ratio?"}
        H5 -->|greater than 0.7| H6["+0.2"]
        H5 -->|less than 0.3| H7["-0.3"]
        H8["Keyword overlap with prompt"]
    end
    
    C3 --> SUM1["Sum -> reward"]
    C4 --> SUM1
    C6 --> SUM1
    C8 --> SUM1
    C10 --> SUM1
    C11 --> SUM1
    
    F3 --> SUM2["Sum -> reward"]
    F4 --> SUM2
    F6 --> SUM2
    F8 --> SUM2
    F9 --> SUM2
    
    H3 --> SUM3["Sum -> reward"]
    H4 --> SUM3
    H6 --> SUM3
    H7 --> SUM3
    
    SUM1 --> CLAMP["clamp(-1, 1)"]
    SUM2 --> CLAMP
    SUM3 --> CLAMP
    CLAMP --> OUTPUT["Final Reward"]

    style CODING fill:#e17055,stroke:#d63031,color:#fff
    style FC fill:#0984e3,stroke:#74b9ff,color:#fff
    style CHAT fill:#00b894,stroke:#00cec9,color:#fff
```

---

## Project Structure

```mermaid
graph LR
    subgraph ROOT["llm-finetuning-agent-lightning"]
        README["README.md"]
        MAIN["main.py"]
        MAIN_AL["main_agent_lightning.py"]
        MAIN_RS["main_reasoning.py"]
        
        subgraph SRC["src/"]
            subgraph MODELS["models/"]
                ML["model_loader.py"]
            end
            subgraph AGENT["agent/"]
                TA["training_agent.py"]
                ALT["agent_lightning_trainer.py"]
                TOOLS["tools.py"]
            end
            subgraph MEM["memory/"]
                VS["vector_store.py"]
                PM["procedural_memory.py"]
                SC["smart_chunker.py"]
            end
            subgraph DAT["data/"]
                DM["data_module.py"]
            end
            subgraph REASONING["reasoning/"]
                LUFFY["luffy_trainer.py"]
                SEARCHR1["search_r1.py"]
            end
        end
        
        subgraph CONFIG["config/"]
            CFG["config.yaml"]
        end
        
        subgraph SCRIPTS["scripts/"]
            CHECK["check_installation.py"]
            INGEST["ingest_knowledge.py"]
        end
    end

    MAIN --> TA
    MAIN_AL --> ALT
    MAIN_RS --> LUFFY
    MAIN_RS --> SEARCHR1
    ALT --> TA
    TA --> ML
    TA --> DM
    ALT --> VS
    ALT --> PM
    LUFFY --> ML
    SEARCHR1 --> VS
    VS --> SC

    style ROOT fill:#2d3436,stroke:#636e72,color:#fff
    style SRC fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style CONFIG fill:#fdcb6e,stroke:#f39c12,color:#000
    style SCRIPTS fill:#00b894,stroke:#00cec9,color:#fff
    style REASONING fill:#e17055,stroke:#d63031,color:#fff
```

---

## LUFFY Off-Policy Learning Flow

```mermaid
flowchart TD
    START["Start Training"] --> PROMPT["Input Prompts"]
    
    PROMPT --> ONPOL["Generate On-Policy<br/>(temperature > 0)"]
    PROMPT --> OFFPOL["Load Off-Policy Traces<br/>(DeepSeek-R1)"]
    
    ONPOL --> R1["Response 1<br/>reward: 0.6"]
    ONPOL --> R2["Response 2<br/>reward: 0.8"]
    ONPOL --> R3["Response 3<br/>reward: 0.4"]
    
    OFFPOL --> T1["R1 Trace 1<br/>reward: 0.95"]
    OFFPOL --> T2["R1 Trace 2<br/>reward: 0.88"]
    
    R1 --> MIXER["Off-Policy Mixer<br/>on_weight: 0.5<br/>off_weight: 0.5"]
    R2 --> MIXER
    R3 --> MIXER
    T1 --> MIXER
    T2 --> MIXER
    
    MIXER --> BATCH["Mixed Batch"]
    
    BATCH --> GRPO["GRPO Loss<br/>+ Importance Sampling"]
    
    GRPO --> KL["KL Penalty<br/>(vs Reference Model)"]
    
    KL --> UPDATE["Update Policy"]
    
    UPDATE --> EXBUF["Add to ExGRPO<br/>Experience Buffer"]
    
    EXBUF --> CHECK{"More<br/>Steps?"}
    CHECK -->|Yes| PROMPT
    CHECK -->|No| END["Improved<br/>Reasoning Model"]

    style START fill:#00b894,stroke:#00cec9,color:#fff
    style END fill:#00b894,stroke:#00cec9,color:#fff
    style MIXER fill:#e17055,stroke:#d63031,color:#fff
    style GRPO fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style EXBUF fill:#fdcb6e,stroke:#f39c12,color:#000
```

---

## Search-R1 Reasoning with Search

```mermaid
flowchart TD
    Q["Question: What is the tallest mountain?"]
    
    Q --> THINK1["🧠 Think Step 1<br/>Let me search for this..."]
    
    THINK1 --> SEARCH["&lt;search&gt;tallest mountain world&lt;/search&gt;"]
    
    SEARCH --> ENGINE["🔍 Hybrid Search Engine"]
    
    subgraph HYBRID["Hybrid Search"]
        VEC["Vector Search<br/>(Semantic)"]
        BM25["BM25 Search<br/>(Keywords)"]
        RRF["Reciprocal Rank<br/>Fusion"]
        
        VEC --> RRF
        BM25 --> RRF
    end
    
    ENGINE --> HYBRID
    RRF --> RESULTS["Top-3 Results"]
    
    RESULTS --> CTX["&lt;context&gt;<br/>[1] Mount Everest 8,849m...<br/>[2] K2 8,611m...<br/>[3] Kangchenjunga 8,586m...<br/>&lt;/context&gt;"]
    
    CTX --> THINK2["🧠 Think Step 2<br/>Based on the search results..."]
    
    THINK2 --> ANSWER["✅ Answer: Mount Everest<br/>at 8,849 meters"]
    
    subgraph REWARD["RL Reward"]
        R1["Correctness: +0.7"]
        R2["Search Bonus: +0.1"]
        R3["Reasoning: +0.2"]
    end
    
    ANSWER --> REWARD

    style Q fill:#74b9ff,stroke:#0984e3,color:#000
    style ENGINE fill:#e17055,stroke:#d63031,color:#fff
    style ANSWER fill:#00b894,stroke:#00cec9,color:#fff
    style HYBRID fill:#0f3460,stroke:#16213e,color:#fff
```

---

## Combined LUFFY + Search-R1 Architecture

```mermaid
flowchart TB
    subgraph INPUT["Input"]
        PROMPTS["Training Prompts"]
        TRACES["DeepSeek-R1<br/>Reasoning Traces"]
        KB["Knowledge Base<br/>Documents"]
    end

    subgraph PHASE1["Phase 1: LUFFY Training"]
        LUFFY_MIX["Off-Policy Mixer"]
        LUFFY_GRPO["GRPO + KL Loss"]
        LUFFY_EXP["ExGRPO Buffer"]
    end

    subgraph PHASE2["Phase 2: Search-R1 Training"]
        SEARCH_ENG["Hybrid Search<br/>Engine"]
        SEARCH_GEN["Generate with<br/>Search Calls"]
        SEARCH_REWARD["Search-Aware<br/>Reward"]
    end

    subgraph OUTPUT["Output"]
        MODEL["Fine-tuned Model<br/>+ Reasoning<br/>+ Search"]
    end

    PROMPTS --> LUFFY_MIX
    TRACES --> LUFFY_MIX
    
    LUFFY_MIX --> LUFFY_GRPO
    LUFFY_GRPO --> LUFFY_EXP
    LUFFY_EXP --> PHASE2
    
    KB --> SEARCH_ENG
    SEARCH_ENG --> SEARCH_GEN
    SEARCH_GEN --> SEARCH_REWARD
    
    SEARCH_REWARD --> MODEL

    style PHASE1 fill:#e17055,stroke:#d63031,color:#fff
    style PHASE2 fill:#0984e3,stroke:#74b9ff,color:#fff
    style OUTPUT fill:#00b894,stroke:#55efc4,color:#fff
```

---

*These diagrams are automatically rendered on GitHub. To edit them, use [Mermaid](https://mermaid.js.org/) syntax.*
