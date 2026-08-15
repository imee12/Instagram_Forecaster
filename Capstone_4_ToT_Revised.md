# Capstone 4

Imee Cuison

## Decide whether ToT reasoning is appropriate for your agent

Tree of Thoughts (ToT) reasoning is appropriate for the content recommendation generation and selection portion of Instagram Forecaster. It improves that stage by forcing the system to explore and compare several grounded content strategies before selecting an answer. Applying ToT to deterministic stages such as loading files, analyzing media, retrieving historical posts, calculating performance metrics, or retrieving Google Trends would hinder the design by adding model cost, latency, and unnecessary branching without improving those operations. For that reason, ToT is deliberately limited to recommendations.

The recommendation stage benefits from ToT because the agent must consider multiple combinations of:

- Available media
- Content format
- Creative strategy
- Google Trends signals
- Similar historical posts
- Opening hook
- Caption direction
- Execution strategy

A linear approach could commit to the first plausible recommendation before considering stronger alternatives. It would also have difficulty balancing the project’s interacting constraints: the media must exist, the format must suit that media, trend and historical citations must be grounded, and the creative hook and execution plan must remain useful. The high number of possible combinations creates a branching problem rather than a single-answer reasoning problem. The implemented bounded ToT process addresses premature commitment and constraint complexity by generating several distinct strategy branches, expanding them into competing candidates, validating their evidence, scoring them, and selecting the strongest three.

## Where ToT is used in the architecture

ToT is used only in the recommendation portion of the forecasting workflow. Media analysis, historical retrieval, and Google Trends retrieval run before ToT and supply its evidence.

The recommendation search receives:

- Structured media analyses
- Similar historical posts
- Historical performance metrics
- Ranked Google Trends signals

The implementation then performs three visible LangGraph stages:

1. `tot_1_plan`: Generate three distinct strategic recommendation branches.
2. `tot_2_expand`: Expand those branches into six complete recommendation candidates.
3. `tot_3_rank`: Validate evidence, calculate weighted scores, prune weaker candidates, and retain the top three.

This is a shallow, bounded Tree-of-Thoughts search using a limited beam-search strategy. It demonstrates branching, exploration, evaluation, and selection without implementing a deeper recursive search.

## Implemented ToT structure

### Thought, node, branch, and depth definitions

A **thought** is a structured recommendation state. At the first generation level, it represents a strategic content hypothesis. At the second generation level, it represents a complete recommendation candidate containing media, format, creative direction, supporting evidence, execution guidance, and scores.

A **node** is one structured thought at a particular level of the search. The conceptual evidence root is also a node, although it contains evidence rather than a recommendation. A **branch** is the path from the evidence root through one strategic hypothesis to one or more complete recommendation candidates. **Depth** indicates how far a node is from the evidence root.

The implemented tree has a maximum depth of two:

- **Depth 0 — Evidence root:** Media analyses, historical matches and metrics, and ranked trend signals.
- **Depth 1 — Strategy thoughts:** Three distinct strategic branches.
- **Depth 2 — Complete candidates:** Six complete recommendations, normally two expanded from each strategy branch.

The expected branching factor at depth 1 is three. Each depth-1 strategy normally produces two depth-2 candidates, for six complete candidates in total.

### Root evidence

The conceptual root contains the evidence available to the recommendation system:

- Available media analyses
- Historical retrieval results
- Historical engagement and performance metrics
- Ranked Google Trends signals

The root does not contain a recommendation.

### Strategy branches

The planning stage creates three structured strategy branches. Each branch contains:

- A short strategy name
- A testable hypothesis
- Evidence dimensions to prioritize
- Target post formats

The branches must represent meaningfully different creative or evidence-based approaches. They are structured with Pydantic models rather than unrestricted text.

### Candidate expansion

The expansion stage distributes six recommendation candidates across the three strategy branches. Each complete candidate contains:

- An exact media filename
- Post format
- Content concept
- Opening hook
- Caption direction
- Rationale
- Execution notes
- Supporting trend topics
- Supporting historical post identifiers when reliable matches are available
- Four evaluation scores
- Confidence score

The prompts instruct Gemini to use only the supplied evidence and not invent media, events, songs, partnerships, locations, or backstory.

### Termination and final selection

The search terminates after depth 2 because every surviving thought is then a complete recommendation candidate. It terminates with an error if media or trend evidence is missing, Gemini fails to return the required structured output, or deterministic evidence validation fails. A missing or sparse historical index activates a fallback scoring mode instead of terminating the workflow. On successful completion, the controller ranks all valid complete candidates and selects the top three.

### Validation, scoring, and pruning

The final stage performs deterministic evidence validation. It rejects a candidate when it cites:

- A media file that was not supplied
- A Google Trends topic that was not retrieved
- A historical post identifier that was not retrieved

Pydantic validation also ensures required fields are present, score values are between 0 and 100, and the post format is supported.

Valid candidates receive a weighted overall score:

| Criterion | Weight |
|---|---:|
| Historical performance | 30% |
| Google Trends alignment | 30% |
| Media quality | 20% |
| Audience fit | 20% |

The calculation is:

```text
overall_score =
    historical_performance × 0.30
  + trend_alignment × 0.30
  + media_quality × 0.20
  + audience_fit × 0.20
```

Candidates are sorted by overall score and then confidence. The top three are retained as the final recommendations. This final ranking is the pruning step in the bounded search.

Historical weight adapts to the quality of the vector index:

| Historical mode | Historical | Trends | Media quality | Audience fit |
|---|---:|---:|---:|---:|
| Healthy | 30% | 30% | 20% | 20% |
| Sparse | 15% | 36.43% | 24.29% | 24.29% |
| Cold start | 0% | 42.86% | 28.57% | 28.57% |

The removed historical weight is redistributed proportionally across the other three criteria. This prevents a new or weak historical index from having the same influence as mature evidence.

## Search and evaluation strategy

### Primary strategy: Beam search

The primary search strategy is a shallow, bounded form of **beam search**. Beam search keeps a limited number of promising thoughts instead of exhaustively retaining every possibility. In this implementation, the evidence root produces three strategic branches. All three fit within the first-level beam and are expanded into six complete recommendation candidates. The evaluator then validates and ranks those candidates and retains the strongest three as the final beam.

The implemented search proceeds as follows:

1. Start with one evidence root.
2. Generate three distinct strategic branches.
3. Retain those three branches for expansion.
4. Expand the branches into six complete candidates.
5. Validate every candidate and fail the recommendation stage if any candidate cites invalid evidence.
6. Rank the remaining candidates by weighted score and confidence.
7. Retain the top three recommendations and discard the rest.

The effective beam width is three: no more than three thoughts are retained for continued use or final output at a selection boundary. Unlike a deeper recursive beam search, this implementation has only two generation levels—strategy planning and complete-candidate expansion. This limited depth is intentional and sufficient to demonstrate branching, evaluation, pruning, and selection.

Beam search fits the recommendation task better than the other listed strategies:

- **BFS** would retain every candidate at a level and would increase model usage as the recommendation space grows.
- **DFS** would explore one strategy deeply before comparing it with alternatives, increasing the risk of premature commitment.
- **Monte Carlo-style sampling** would add randomness and require more samples to produce stable, reproducible recommendations.
- **Beam search** compares multiple alternatives while imposing a fixed limit on how many results survive, providing a practical balance among exploration, quality, latency, and Gemini quota usage.

Its search limits are:

- Strategy branches: 3
- Total complete candidates: 6
- Final recommendations: 3
- Gemini planning calls: 1
- Gemini expansion calls: 1 per strategy branch

This bounded beam-search design demonstrates the important ToT behaviors while controlling complexity. A depth-three implementation would require more state, evaluation phases, model calls, latency, and quota. For this recommendation task, the shallow search provides a reasonable balance between exploration and cost.

Evaluation combines:

- Gemini-generated structured candidates and preliminary scores
- Retrieved historical evidence
- Google Trends evidence
- Media analysis evidence
- Deterministic Python validation
- Deterministic weighted ranking

The current implementation does not use a separate critic-model call. Candidate scores are generated as part of structured candidate expansion and are subsequently validated, weighted, sorted, and pruned by Python logic.

### Who performs the evaluation

Evaluation is shared between the Gemini thought generator and deterministic Python logic:

- Gemini assigns preliminary scores for historical performance, trend alignment, media quality, audience fit, and confidence as part of its structured candidate output.
- Pydantic validates required fields, supported formats, and score ranges.
- Python verifies that every cited media file, trend topic, and historical post exists in the retrieved evidence.
- Python calculates the weighted overall score, reranks the valid candidates, and applies the final beam-width cutoff.

This combination acts as the critic or evaluator. A separate critic agent is not used in the current implementation.

### Pruning thresholds and failure conditions

The implementation uses structural and ranking thresholds rather than an arbitrary minimum numerical score:

- Any candidate with nonexistent media, trend, or historical evidence fails deterministic validation and causes the current recommendation stage to fail. The current implementation does not silently remove that candidate and continue.
- Missing required fields, unsupported formats, or scores outside 0–100 fail Pydantic validation.
- A model response with the wrong number of strategy branches or candidates fails the stage.
- After validation, the beam-width threshold retains only three candidates; every candidate ranked fourth or lower is pruned.
- Missing media analyses or trend signals stops recommendation generation and routes the workflow to error handling. Missing historical matches activates cold-start mode instead.
- Gemini API errors, including quota exhaustion, stop the real ToT path and are captured in the LangGraph/LangSmith trace.

### Tie resolution and selection policy

Candidates are reranked rather than selected by voting. Ties are resolved in this order:

1. Higher weighted overall score.
2. Higher Gemini confidence score.

The top three rows after this reranking are selected. If both overall score and confidence are exactly equal, the current implementation has no additional explicit tie breaker; the existing dataframe ordering determines which row appears first. The implementation also does not apply a diversity-based tie breaker, so the document does not claim one.

### Compute, latency, and cost constraints

The search is constrained to three strategy branches, six complete candidates, and three final recommendations. A normal real-mode run uses one Gemini planning request and three expansion requests. Media analysis, historical retrieval, and trend retrieval are completed before the search and are not repeated inside each branch. These limits cap branching, latency, and model usage—an important constraint because the development API project has a small request quota. `DEV_MODE=true` provides a zero-Gemini mock path for testing graph routing, tracing, artifact saving, and UI behavior.

### Sparse-index and cold-start handling

The historical retrieval stage evaluates both index size and match quality. Matches below a cosine-similarity threshold of 0.35 are excluded. The evidence is classified as:

- **Healthy:** At least 10 indexed posts and at least two qualifying matches for every analyzed media asset.
- **Sparse:** Some qualifying history exists, but the index has fewer than 10 posts or one or more media assets have fewer than two qualifying matches.
- **Cold start:** The index is empty or no match reaches the similarity threshold.

Sparse mode reduces the historical-performance weight from 30% to 15%. Cold-start mode sets it to 0%. In cold-start mode, candidates return an empty historical-post list and are instructed not to claim historical support. Recommendations continue using media quality, trends, and audience fit, and the resulting evidence mode is saved in the recommendation output, included in traces, and displayed in Streamlit. As more historical posts are collected, rebuilding the vector index allows the workflow to move automatically from cold-start or sparse mode to healthy mode.

## Mapping ToT roles to implementation tools

### Thought generator

Implementation: Gemini through the Google GenAI client, with Pydantic structured outputs and LangSmith tracing. This functionality sits within the LangChain/LangGraph ecosystem, but the recommendation calls use the Google GenAI client directly rather than LCEL.

Gemini is appropriate for this role because it can propose varied creative strategies while following a strict response schema and using the supplied evidence context.

Gemini performs two generation responsibilities:

1. It creates the three structured strategic branches.
2. It expands each branch into structured recommendation candidates.

Pydantic response schemas constrain both outputs. The prompts explicitly request concise strategy hypotheses and structured recommendations rather than private or unrestricted chain-of-thought.

### Evaluator

Implementation: Gemini-provided candidate scores combined with deterministic Python validation and ranking.

This hybrid evaluator fits the project because Gemini can assess qualitative content fit while Python provides reproducible grounding checks, weighted arithmetic, and pruning without adding another model call.

Gemini supplies scores for historical performance, trend alignment, media quality, and audience fit. Python then:

- Confirms that cited evidence exists
- Enforces structured field and score constraints
- Calculates the weighted overall score
- Sorts candidates by overall score and confidence
- Selects the top three

### Decision maker and controller

Implementation: LangGraph, part of the LangChain ecosystem.

LangGraph is appropriate because it makes planning, expansion, ranking, conditional mode selection, and failure routing explicit and observable as stateful workflow nodes.

LangGraph controls the recommendation execution path through visible nodes:

```text
generate_recommendations
→ tot_1_plan
→ tot_2_expand
→ tot_3_rank
```

The workflow routes failures to an error-handling node and returns to project inspection after successful recommendation generation. LangSmith records the graph execution and nested Gemini calls.

### Memory and state manager

Implementation: LangGraph state and checkpointing.

LangGraph state is appropriate because the planning output and expanded candidates are structured, serializable data that must be passed reliably between graph nodes and associated with one workflow thread.

LangGraph state carries the serializable evidence contexts, strategy branches, and expanded candidates between the three ToT nodes. The application associates workflow execution with thread identifiers and uses checkpointing where configured. Generated recommendations are also saved as JSON and CSV project artifacts for later use by the Streamlit interface and conversational agent.

### Tool-selection summary

- **LangChain/LangGraph:** Selected for graph control flow, state passing, tool integration, conditional routing, checkpoint compatibility, and LangSmith observability.
- **CrewAI:** Not selected because the feature does not need a second multi-agent orchestration framework.
- **MCP:** Not selected for internal branch tracking because every ToT stage runs within the same local Python and LangGraph process.

## Development mode

The project includes a `DEV_MODE` configuration flag. When `DEV_MODE=true`, the recommendation workflow follows a `dev_mock_tot` node that produces deterministic, evidence-grounded mock recommendations without making Gemini ToT calls.

Development mode still exercises:

- LangGraph routing
- LangSmith tracing
- Evidence loading
- Recommendation validation and scoring
- Artifact saving
- Streamlit rendering

When `DEV_MODE=false`, the workflow uses the real `tot_1_plan`, `tot_2_expand`, and `tot_3_rank` nodes and makes the Gemini calls described above.

The current mock path represents the completed recommendation operation as one graph node. It does not simulate each real ToT phase separately.

## CrewAI decision

CrewAI is not used for this feature. LangGraph already provides explicit workflow nodes, state management, conditional routing, persistence support, and tracing. Adding another orchestration framework would duplicate responsibilities without improving this bounded recommendation search.

## MCP decision

MCP is not used for internal ToT state. The media, trend, retrieval, and recommendation components execute inside the same Python application, so LangGraph state is sufficient for passing branch and candidate data between nodes.

MCP could be considered later if project capabilities need to be exposed to external agent clients, but it is unnecessary for the current implementation.

## Risk and mitigation

### Risk: Branch explosion and high computational cost

The most relevant risk is branch explosion. Recommendation generation can combine many media assets, formats, trends, historical posts, hooks, and execution ideas. Expanding every combination would increase latency and could quickly exhaust the Gemini API quota without guaranteeing better recommendations.

### Mitigation

The workflow applies a fixed search budget: three strategy branches, six complete candidates, a maximum depth of two, and a final beam width of three. Evidence is retrieved once and reused across branches. Low-similarity historical matches are filtered, and adaptive scoring prevents sparse history from being over-weighted. Development mode replaces Gemini generation with deterministic mock output during integration testing. Together, these controls keep model calls and latency predictable while preserving enough branching to demonstrate ToT reasoning.

## Conclusion

Tree of Thoughts is appropriate for Instagram Forecaster because recommendation generation requires exploration and comparison among several plausible content strategies. The implemented design uses a deliberately shallow and bounded ToT search: three strategy branches produce six complete candidates, which are validated, scored, pruned, and reduced to three final recommendations.

This design demonstrates the central value of ToT—exploring alternatives before selecting an answer—while remaining compatible with the project’s model quota, latency requirements, LangGraph workflow, LangSmith observability, and Streamlit interface.
