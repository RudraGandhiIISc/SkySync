"""Every numeric target from the PRD (docs/PRD.md sections 4 and 5).

Code and tests import these names instead of repeating the numbers.
"""

from typing import Final

# FR-1: grounded answers.
FR1_MIN_CITED_ANSWER_FRACTION: Final = 0.95
FR1_MIN_ABSTAIN_FRACTION: Final = 0.90
REFUSAL_TEXT: Final = "I don't know based on the provided documents."

# FR-2: filtered retrieval.
FR2_ACCESS_PROBES: Final = 10_000
FR2_MAX_FORBIDDEN_CHUNKS: Final = 0
FR2_MIN_RECALL_AT_K: Final = 0.95
RETRIEVAL_TOP_K: Final = 8
# Reranked chunks offered to the model (FR-2 recall is still measured at
# RETRIEVAL_TOP_K). 12 rather than 8: on the 360 answerable v3 questions
# the key fact reached the prompt for 89% instead of 84% (job 1734,
# eval/versioned/retrieval.py).
PROMPT_TOP_K: Final = 12
HNSW_M: Final = 16
HNSW_EF_CONSTRUCTION: Final = 200
HNSW_EF_SEARCH: Final = 64
# BAAI/bge-m3 (phase 14): 1,024 dimensions, 8k-token passages.
EMBEDDING_DIM: Final = 1024
EMBED_MODEL: Final = "BAAI/bge-m3"
RERANK_MODEL: Final = "BAAI/bge-reranker-v2-m3"
# Hybrid-search candidates the cross-encoder reorders before the top_k
# are kept; its input is cut to this many characters per chunk.
# 80 rather than 40: fact recall 0.84 vs 0.79 (job 1734); 150 adds nothing.
RERANK_CANDIDATES: Final = 80
RERANK_PASSAGE_CHARS: Final = 2000
# Questions about several years (or "what changed") search each year (an
# academic year at its start and its end); at most this many as-of times.
MAX_TIME_SCOPES: Final = 4
# NFR-7 hybrid search: vector and BM25 candidates fused by reciprocal rank
# (chosen on the held-out set: docs/results/retrieval-v2/README.md).
HYBRID_CANDIDATES: Final = 20
RRF_K: Final = 60
KEYWORD_WEIGHT: Final = 2.0

# FR-3 / NFR-10: access decisions.
FR3_REVOCATION_SECONDS: Final = 60.0
FR3_MIN_MATRIX_CASES: Final = 200
NFR10_MAX_FALSE_ALLOWS: Final = 0
NFR10_MAX_FALSE_DENY_FRACTION: Final = 0.01
NFR10_AUTHORIZE_MS: Final = 10.0
NFR10_FILTER_MS: Final = 20.0
NFR10_DECISIONS_PER_SECOND: Final = 500
ACCESS_CLIENT_TIMEOUT_S: Final = 0.050

# FR-5 / NFR-8: versions and freshness.
FR5_REVISION_CHECKPOINTS: Final = 60
FR5_STALE_DEMOTION_SECONDS: Final = 15 * 60.0
NFR8_MIN_CURRENT_FRACTION: Final = 0.99
NFR8_REVISED_DOCS_PER_HOUR: Final = 50
RECONCILE_INTERVAL_S: Final = 5 * 60.0

# Ingestion (A3).
CHUNK_TARGET_TOKENS: Final = 190
CHUNK_OVERLAP_TOKENS: Final = 30
CORPUS_DOCUMENTS: Final = 500
CORPUS_CHUNKS: Final = 15_000

# FR-6: load limits.
FR6_OVERLOAD_FACTOR: Final = 3.0
FR6_REJECT_WITHIN_MS: Final = 50.0
FR6_MAX_P95_RISE_FRACTION: Final = 0.20

# FR-7: batching and prefix caching.
FR7_MIN_THROUGHPUT_GAIN: Final = 2.0
FR7_MIN_P99_REDUCTION: Final = 0.30
FR7_MIN_CACHE_HIT_RATE: Final = 0.50
FR7_MIN_TTFT_REDUCTION: Final = 0.25

# FR-8: models.
# The PRD has 9. Gemma-3-4B and Llama-3.1-8B (licence-gated) and
# Llama-3.1-8B-AWQ (its parent gone) were dropped on 2026-11-02: two each of
# 4B, 8B and 4-bit 8B.
FR8_MODEL_COUNT: Final = 6
FR8_MIN_KV_ROOM_GAIN_4BIT: Final = 2.0
LLM_MAX_TOKENS: Final = 256
LLM_TEMPERATURE: Final = 0.0
MAX_QUESTION_TOKENS: Final = 512
# 8,192 rather than the README's 4,096: dense, number-heavy chunks run at
# ~1.7 characters per token, so 9,000 characters reached 5,152 tokens
# (job 1360). At that density the budget below is ~6k tokens, which fits
# with the system prompt, a 512-token question and the 256-token answer.
# Doubled with the prompt budget (12 chunks, 16,000 characters): ~9.4k
# tokens of context at the worst density, plus prompt, question, answer.
LLM_MAX_MODEL_LEN: Final = 16384
# Retrieved text in one prompt.
PROMPT_CONTEXT_CHARS: Final = 16000
PROMPT_CHUNK_CHARS: Final = 2000
# An answer whose content words are almost all missing from the chunks it
# cites came from outside knowledge ("Paris", citing a seminar page) and
# becomes the refusal. Tuned on the 400 edition-aware answers of
# 2026-10-27: at 0.3 no graded answer changes; at 0.4 five correct answers
# would be lost for two wrong ones caught.
MIN_GROUNDED_SHARE: Final = 0.3
# Multi-turn chat: earlier turns sent with a follow-up question.
CHAT_HISTORY_MESSAGES: Final = 6
CHAT_HISTORY_CHARS: Final = 1500

# FR-9 / FR-10 / NFR-4: autoscaling.
MAX_GPU_WORKERS: Final = 2
FR9_SCALE_OUT_SECONDS: Final = 180.0
FR10_RELEASE_AFTER_IDLE_SECONDS: Final = 60.0
FR10_SCRIPTED_COLD_STARTS: Final = 20
AUTOSCALER_HYSTERESIS_S: Final = 30.0
# R-3: a card counts as shared at a worker's start when its used memory
# exceeds what our own processes account for by more than this (driver and
# context overhead), for when NVML hides other users' processes.
GPU_SHARED_SLACK_MIB: Final = 1024

# FR-11: fairness.
FR11_LOAD_RATIO: Final = 10.0
FR11_MAX_HEAVY_SHARE: Final = 0.55
FR11_MIN_LIGHT_SHARE: Final = 0.20

# NFR-1 / NFR-2: latency.
NFR1_P95_TOTAL_S: Final = 3.5
NFR1_P95_TTFT_S: Final = 0.800
NFR2_COLD_TTFT_SMALL_S: Final = 90.0
NFR2_COLD_TTFT_8B_S: Final = 150.0

# NFR-3: throughput per worker (provisional), output tokens per second.
NFR3_TOKENS_PER_S_4B: Final = 600.0
NFR3_TOKENS_PER_S_8B: Final = 250.0
NFR3_TOKENS_PER_S_8B_4BIT: Final = 450.0
NFR3_CROSSCHECK_TOLERANCE: Final = 0.03
NFR3_CONCURRENCY: Final = 64  # throughput is read at this concurrency

# NFR-4: scale.
NFR4_USERS_PER_WORKER: Final = 60
NFR4_MAX_ERROR_FRACTION: Final = 0.01
NFR4_MAX_P95_RISE_FRACTION: Final = 0.20
NFR4_SEARCH_P95_S: Final = 0.150

# NFR-5: availability.
NFR5_MIN_SUCCESS_FRACTION: Final = 0.99
NFR5_RESTORE_SECONDS: Final = 180.0

# NFR-6: cost.
NFR6_MAX_GPU_HOURS_PER_1K: Final = 0.30

# NFR-7: quality.
NFR7_MIN_ACCURACY: Final = 0.80
NFR7_MIN_CITATION_CORRECTNESS: Final = 0.90
NFR7_HUMAN_SAMPLE: Final = 60

# NFR-9: security.
NFR9_MAX_LEAKS: Final = 0
NFR9_MAX_INJECTION_SUCCESS: Final = 0.05
NFR9_ATTACK_CASES: Final = 90
