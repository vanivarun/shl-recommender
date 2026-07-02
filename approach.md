# Approach Document: SHL Conversational Assessment Recommender

**Stack:** FastAPI · FAISS (`IndexFlatIP`) · TF-IDF (scikit-learn) · Groq API (Llama 3.1 8B Instant) · Render.com

## 1. Design Choices

**Stateless API.** The spec requires the full conversation history on every `POST /chat` call, so we keep no server-side session state. Each request is a self-contained replay: the agent re-derives context (role, seniority, skills, constraints, current shortlist) from the message history on every turn. This trades a small amount of recomputation for simplicity and horizontal scalability — important since we don't control how the evaluator load-balances requests on Render's free tier, and cold starts mean any given call may hit a fresh process.

**FAISS `IndexFlatIP` for retrieval.** Our scraped catalog of Individual Test Solutions has 338 assessments — small enough that exact search is cheap and an approximate index would only add complexity for no real speed gain. We use `IndexFlatIP` (exact inner-product search) over L2-normalized vectors, which is mathematically equivalent to cosine similarity ranking and keeps retrieval both accurate and simple to reason about. The index is built once at startup from the scraped catalog JSON and held in memory for the life of the process; a 338-item flat index rebuilds in well under a second, so cold starts aren't a bottleneck.

**TF-IDF (scikit-learn) for vectorization.** We chose classical TF-IDF over neural embeddings for this catalog size and time budget. `TfidfVectorizer` is fit once on the 338 catalog records (name + description + test type) at startup, producing sparse vectors that we densify and L2-normalize before loading into FAISS. This has near-zero inference cost, no model download, and no GPU/CPU tradeoff to manage — a meaningful advantage on Render's free tier where cold-start time and memory are both constrained. The tradeoff is that TF-IDF is a lexical/keyword-overlap method, not a semantic one, so it depends more heavily on query terms overlapping catalog vocabulary (see Section 5).

**Groq API with Llama 3.1 8B Instant as the LLM.** Used for conversation orchestration — intent parsing, clarification questions, refinement logic, comparison synthesis, and refusal handling. Groq's inference speed is a major factor given the 30-second per-call timeout: 8B Instant returns typically in well under a second of generation time, leaving comfortable margin for retrieval and validation steps in the same request. The LLM never invents catalog data; it only reasons over retrieved candidates and structures output into the required JSON schema.

**Separation of retrieval and generation.** FAISS/TF-IDF decides *which* catalog items are candidates; Llama decides *whether* to ask, recommend, refine, or refuse, and how to phrase the reply. The model is never asked to recall a URL from memory, only to select from a passed-in candidate list — this is our primary defense against hallucination.

## 2. Retrieval Setup

**Scraping.** We crawled the SHL product catalog page, filtered to Individual Test Solutions (excluding pre-packaged Job Solutions), and normalized each entry into a record with: name, catalog URL, test type code (e.g., K = Knowledge & Skills, P = Personality), description, and any listed job level/duration metadata. This became our source-of-truth JSON, which is the *only* place recommendation URLs can come from.

**Vectorization.** We fit a single `TfidfVectorizer` (scikit-learn) on the corpus of 338 catalog records, concatenating name + description + test type per record, with standard English stopword removal and unigram/bigram features. The fitted vectorizer transforms both the catalog (once, at startup) and each incoming query (per turn) into the same TF-IDF space. Vectors are L2-normalized and converted to dense float32 arrays before insertion into `IndexFlatIP`, so inner-product search returns cosine-ranked results.

**Query construction.** On each turn, we don't vectorize the raw last user message in isolation. We build a rolling "requirement summary" from the whole conversation (role, seniority, skills mentioned, test-type preferences volunteered or refined) and transform that summary through the fitted vectorizer. This is what makes "Actually, add personality tests" work as a refinement rather than a reset — the new constraint is merged into the summary before the next FAISS query, so prior context (e.g., "Java," "mid-level") is preserved.

**Search + filtering.** We retrieve the top 15–20 nearest neighbors from the 338-item index, then apply any hard constraints the user has stated explicitly (e.g., "no personality tests") as a post-filter before handing the candidate list to Llama for final selection and phrasing into 1–10 recommendations.

## 3. Prompt Design

We use a single system prompt with explicit branching instructions for the four required behaviors, plus the retrieved candidates (or none) injected per turn:

- **Clarify:** If the conversation summary lacks enough signal (role/skill area, and at least one of seniority or test-type preference), the agent must ask exactly one targeted follow-up question and return `recommendations: []`. This prevents premature, ungrounded shortlists on turn 1.
- **Recommend:** Once sufficient context exists, the agent selects 1–10 items *only* from the FAISS candidate list, each with the name, catalog URL, and test type taken verbatim from our scraped JSON — never generated by the LLM.
- **Refine:** The prompt instructs the model to treat new constraints as amendments to the standing requirement summary, not a new conversation, and to re-query the TF-IDF/FAISS index with the updated summary rather than discard prior shortlist context.
- **Compare:** For "what's the difference between X and Y" queries, we retrieve the specific catalog records for X and Y by name lookup (not similarity search) and instruct the model to answer strictly from their scraped descriptions/test types, refusing to fall back on prior knowledge of SHL products.
- **Scope guard:** A standing instruction refuses general hiring/legal advice and prompt-injection attempts ("ignore previous instructions," role-override attempts, etc.), returning a short redirect and `recommendations: []`.

Output is constrained via a JSON schema passed to the model, and we validate/repair the response server-side (e.g., stripping any URL not present in our catalog JSON) before returning it, as a second line of defense against hallucination.

## 4. Evaluation Approach

We ran the 10 provided public conversation traces against a local harness that replays each persona's facts turn-by-turn and computes Recall@10 against the labeled expected shortlist. We iterated on the requirement-summary construction and the clarify/recommend threshold specifically because early runs were recommending too early (before enough constraints were gathered) and losing recall as a result. We also wrote a handful of adversarial behavior probes ourselves — off-topic questions, legal-advice requests, and basic prompt-injection strings — to confirm the scope guard held and no recommendations leaked out on refusal turns. We manually spot-checked every returned URL against the scraped catalog to confirm zero hallucinated links.

## 5. What Didn't Work

Our first version queried the index on the raw last-turn message only, which caused the shortlist to "forget" earlier constraints on refinement turns (e.g., losing "Java" after the user added "personality tests"). Switching to a cumulative requirement summary fixed this. TF-IDF's lexical nature also showed a real limitation during testing: queries that described a role in different words than the catalog used (e.g., "people manager" vs. catalog wording like "supervisory") sometimes missed relevant assessments purely on vocabulary mismatch rather than any conceptual mismatch; we partially mitigated this by expanding the query summary with a few common synonyms per skill area, though this remains an area for improvement. We also initially let the LLM freely generate catalog URLs from its own knowledge for well-known SHL products (e.g., OPQ), which produced plausible but sometimes wrong or outdated links; we corrected this by making retrieval-then-select mandatory and adding a server-side URL validation step against the scraped catalog. Given the 7-hour window, we did not have time to tune the clarify-vs-recommend threshold against the (unseen) holdout traces, so some persona conversations may still recommend one turn earlier or later than ideal.

## 6. AI Tools Used

We used Claude for scaffolding the FastAPI service, the scraping script, and the FAISS/TF-IDF indexing and query code, and for drafting this document, with the team reviewing and adjusting the retrieval logic, prompt branching, and schema validation by hand to ensure the design choices above were understood and defensible, not just generated.