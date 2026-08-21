# Capstone 5: Multi-Agent Architecture Plan

Imee Cuison

## Problem and rationale for a multi-agent approach

Instagram Forecaster helps a creator decide what to post using available media, historical Instagram performance, semantically similar posts, and Google Trends. It must analyze different media types, retrieve evidence, explore creative strategies, validate recommendations, and explain the final choices.

A single general-purpose agent could attempt all these tasks, but it would combine too many responsibilities and failure modes in one prompt. Visual analysis requires different instructions and tools from vector retrieval. Evidence collection should remain separate from creative generation so the generator cannot invent support. Recommendation evaluation should also be separated from generation so that candidate grounding and ranking are checked consistently. A multi-agent design improves reliability by giving each agent a narrow role, explicit inputs, and structured outputs.

The system will use multiple agents only for work requiring specialized judgment. Deterministic file loading, score calculation, validation, saving, and routing will remain Python or LangGraph nodes.

## Number of agents

The proposed architecture includes **five agents**:

1. Workflow Coordinator
2. Media Intelligence Agent
3. Evidence Retrieval Agent
4. Content Strategy Agent
5. Recommendation Critic

Five agents provide useful separation without excessive coordination overhead. Combining media analysis with retrieval would mix multimodal interpretation with search, while combining generation with criticism would weaken independent validation. Splitting historical, trend, and metrics work into additional agents would add messages without enough benefit at the current scale.

## Agent roles and responsibilities

### 1. Workflow Coordinator

The Workflow Coordinator receives Streamlit or chat requests, inspects saved artifacts, checks caches, routes required work, tracks failures, and assembles the response. It does not create content or alter evidence. LangGraph serves as its execution harness and state controller.

### 2. Media Intelligence Agent

The Media Intelligence Agent analyzes images and videos with Gemini and returns structured observations about content, themes, possible uses, quality, and opening moments. It may describe only observable information. Pydantic validates its results, which are cached for unchanged media.

### 3. Evidence Retrieval Agent

The Evidence Retrieval Agent builds media-specific queries, searches the vector index, calculates historical metrics, and retrieves ranked Google Trends signals. It classifies historical evidence as `healthy`, `sparse`, or `cold_start`; weak matches are filtered rather than presented as support.

### 4. Content Strategy Agent

The Content Strategy Agent performs the bounded Tree-of-Thoughts search using validated upstream evidence. It creates three strategy branches and expands them into six structured candidates containing media, format, concept, execution guidance, evidence, scores, and confidence. It explores alternatives but does not select the winners.

### 5. Recommendation Critic

The Recommendation Critic verifies cited evidence, applies adaptive weights, ranks candidates, and retains the strongest three. Historical evidence receives 30% weight in healthy mode, 15% in sparse mode, and 0% in cold-start mode. It may issue one structured revision request. Deterministic Python remains authoritative for identifiers, ranges, arithmetic, and pruning.

## Coordination strategy

The system uses a **hybrid graph-based strategy**. The main workflow is sequential because later stages depend on validated upstream outputs, but LangGraph provides conditional branches, caching, error routes, cold-start behavior, and one bounded feedback loop.

```text
User / Streamlit
       ↓
Workflow Coordinator
       ↓
Media Intelligence Agent
       ↓
Evidence Retrieval Agent
       ↓
Content Strategy Agent
       ↓
Recommendation Critic
       ├── approved → save and present recommendations
       └── revise once → Content Strategy Agent
```

The normal path is an efficient assembly line. Graph branches allow cached work to be reused, sparse history to activate adaptive scoring, and failures to reach a dedicated error outcome. Critic feedback is limited to one retry to prevent unbounded cost.

## Communication and shared state

Agents communicate through typed LangGraph state rather than informal conversation. Pydantic models define media analyses, strategy branches, and candidates; serializable records carry matches, metrics, and trends. Each agent reads only required fields and writes a defined output.

Communication is mostly one-way for efficiency and provenance. Two-way communication is reserved for one critic-to-strategist revision. The critic sends structured failure reasons rather than a free-form conversation.

LangSmith records runs, nodes, tool calls, inputs, outputs, errors, and timing. Thread IDs and checkpoints associate state with the correct workflow, while saved artifacts allow Streamlit to reuse results.

## Trade-offs

Specialized prompts, separation of evidence from generation, typed schemas, and independent criticism improve reliability. Graph routing makes prerequisites and failures visible.

The cost is greater complexity: more state contracts, traces, tests, and failure handling. Critic feedback improves quality but adds latency and a model call. The design limits overhead by keeping mechanical work deterministic, combining evidence retrieval, caching outputs, using shallow ToT, and allowing only one revision.

Sequential dependencies limit parallelism because historical retrieval depends on media analysis, although trends can be retrieved concurrently. Preserving evidence lineage is more important than maximum concurrency. Development mode supports integration testing without consuming Gemini quota.

## Effective and scalable problem solving

The architecture scales by separating orchestration from specialized work. Media assets can be analyzed independently and cached, while retrieval can use a larger index without changing downstream contracts. Models or scoring policies can be replaced within one role. LangGraph routing supports retries, cold-start handling, and future human approval.

Each stage is observable and independently testable, so failures can be attributed to analysis, retrieval, generation, or evaluation instead of one opaque agent. The five-agent design keeps coordination bounded and explainable.
