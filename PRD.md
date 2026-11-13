# Ask IISc: Multi-Tenant Retrieval-Augmented Question Answering over Versioned Institute Documents

| | |
|---|---|
| **Team Name** | SkySync |
| **Team Members** | Rudra Kumar Gandhi (RKG), rudragandhi@iisc.ac.in · Muthangi Vighnesh (MV), muthangiv@iisc.ac.in · Mohammad Haris (MH), harism@iisc.ac.in |
| **Summary** | Ask IISc answers questions over versioned institute documents on a departmental GPU cluster, enforcing per-user access through a standalone access-control service, and compares six open 4B, 8B and 4-bit 8B models on accuracy, latency, cost and scalability. |
| **Course** | DS252 Cloud Computing |

> Final: the team's PRD with the server updates to A1–A2, Section 6 and risks R-1 to R-4; where the built system differs from the design in Section 6 (the CSA server stack), see [the deviations in docs/STATUS.md](docs/STATUS.md#4-deviations-from-the-prd).

## 1. Problem statement and motivation

IISc's important documents, such as regulations, circulars and exam notices, are spread across about forty department websites and a shared drive. Students and staff struggle to find answers, office staff keep repeating the same replies, and people sometimes follow a circular that has already been replaced. Public AI assistants can't be used, because many of these documents are internal and must stay on institute systems.

We need an assistant that answers from the latest documents, cites its sources, and shows each person only what they are allowed to see. It must also run on a small, shared pool of GPUs that cannot handle heavy traffic all the time. Usage spikes during registration week and nearly disappears at night, so the system must handle peaks fairly and cheaply without wasting GPUs when idle. This project builds that service and measures how model size affects its accuracy, speed and cost.

## 2. Scope: in-scope, out-of-scope, assumptions

### 2.1 In scope

**Document intake:** read PDF and web documents (with readable text), split them into small pieces and keep every version of each document. **Search:** a database (Milvus) that stores each piece with its version and who may read it and only searches what the user is allowed to see. **Outdated content checker** finds old versions and stops them from being used. **Access-control service:** a separate service for deciding who can read what, so other teams can reuse it. **Model serving:** run six open models (two each at 4B, 8B and 4-bit 8B size) with batching and cache reuse so GPUs are used well. **Gateway:** limits load, shares capacity fairly between departments and keeps a log of every request. **Autoscaler:** adds GPU nodes when requests rise (up to 2) and releases them all when idle. **Load-testing tool** simulates many users at different scales. **Cost model:** counts GPU-hours, including start-up time and time spent idle but still held.

### 2.2 Out of scope

We will **not train or fine-tune** any model. We only compare ready-made models, because training would add a second source of change and make the accuracy, speed and cost results hard to compare. For the same reason **we will not compare different hardware**. We use one cluster with one type of GPU, so any difference in results comes from the model and not from the machine.

We also **do not promise that every answer is true**. The system gives answers with citations and says "I don't know" when the documents do not support an answer, and we measure this on a fixed set of test questions. We do not connect to the institute's single sign-on, because registering an application with it was not possible: allowlisted IISc people sign in with their e-mail address and a password, and departments and roles still come from the institute's user directory. We **will not handle scanned documents or non-English questions**, because documents need readable text.

Finally, this is a course project and not a live service. There is **no round-the-clock support, no backups and no uptime promise**. We protect against harmful prompts and poisoned documents, but not against huge floods of traffic.

### 2.3 Assumptions

- **A1:** The CSA `automation` server gives us at most 2 of its 3 NVIDIA H200 NVL GPUs at a time, plus CPU cores, on one Slurm node reached over the campus VPN. Slurm reserves no GPU, so other users' jobs can fill every card; while they do, no GPU node can start. In this document a "GPU node" is one vLLM worker on one GPU.
- **A2:** Our code can start and release GPU workers, and the start-up time is measurable. Slurm on this server reserves no GPU and offers no containers, so the autoscaler starts and stops vLLM processes inside the run's Slurm job, which is the fallback this assumption allowed for. Node acquisition is measured as choosing a free card plus an optional emulated delay, reported separately from model load.
- **A3:** About 500 documents (about 15,000 pieces) are available with readable text and a last-changed date. If there is no date, we compare document contents instead.
- **A4:** Every request comes with a verified user ID from the sign-in. Departments and roles come from a directory export, never from the user's device. If there is no export, we use a fake directory with the same format.
- **A5:** Users may send harmful prompts or poisoned documents, but they will not flood the system with traffic.

## 3. Stakeholders and usage scenarios

### 3.1 Users

**Reader** (student, researcher or staff): asks questions and gets cited answers from only the documents they may read. **Document owner** (department admin): publishes and updates documents and sets who can read them. **Cluster operator:** sets limits and the idle timeout, and watches GPU cost. **Auditor:** checks which document version produced an answer.

### 3.2 Scenarios

- **S1 – Registration-week rush:** About 400 users ask similar questions within 10 minutes. The system adds GPU nodes, reuses work across similar questions and turns away excess requests instead of letting them queue forever.
- **S2 – Overnight idle, then morning:** All GPUs are released overnight, so nothing is charged. The first question at 08:41 is held, not failed, while a node starts, and then it is answered.
- **S3 – Revised circular with a hidden attack:** A new version of a circular replaces the old one while questions are running. Old chunks stop being used within 15 minutes, and any hidden instruction in the new text, such as "ignore prior instructions", has no effect. An auditor can still see which version answered an earlier query.

## 4. Functional requirements

| ID | Requirement | Priority | Acceptance criterion |
|---|---|---|---|
| FR-1 | Answer only from retrieved text, cite sources and say I don't know when unsupported | Must | On the edition-aware question set, at least 95% of the citations in answers support them and at least 75% of unanswerable questions get an explicit I don't know |
| FR-2 | ACL and version-filtered retrieval: search only chunks the user may read and that are current | Must | In 10000 access probes zero forbidden chunks are returned and recall@8 is at least 0.95 against an exact filtered search |
| FR-3 | Decide access per user and apply permission changes quickly | Must | Decisions match a reference checker on the test matrix and 60 seconds after a revocation no query returns the revoked content |
| FR-4 | Provide access control as a separate and reusable service | Must | The service passes its conformance tests and with the service killed nothing is answered without an access check |
| FR-5 | Chunk staleness detection and demotion: keep every document version and stop using outdated chunks | Must | As-of-date queries are correct at all 60 revision checkpoints; superseded chunks leave search within 15 minutes; and no answer uses them without a warning flag |
| FR-6 | Limit load per tenant and reject requests that would wait too long | Must | At 3x load excess requests get HTTP 429 within 50 ms and the p95 latency of accepted requests rises by at most 20% |
| FR-7 | Use continuous batching and prefix cache reuse so GPUs serve more users | Must | Continuous batching gives at least 2x the output tokens per second of serving one request at a time (E0 sweep, concurrency 64 against 1); with prefix caching on, the cache hit rate is at least 50% and p95 time-to-first token, measured at the model server, is at least 25% lower than with it off |
| FR-8 | Serve six models (two families in each of the 4B and 8B and 4-bit 8B sizes) through one unchanged API | Must | All six answer the full test set through the same client; at the same GPU memory budget each 4-bit model has at least 2x the KV-cache room of its 8B parent; and switching models needs only a config change |
| FR-9 | Autoscaling (scaling out): add GPU nodes as request volume rises to a limit of 2 | Must | Under a burst the pool reaches the needed size within 180 seconds; never exceeds 2 nodes and loses no accepted request |
| FR-10 | Autoscaling (scaling in): release idle nodes after a timeout and hold the first request after zero instead of failing it | Must | 60 seconds after the idle timeout no node is active or charged; in 20 scripted cold starts every held request completes with a visible warming indication; and when no card has the memory a node needs (A1), a held request ends at once with a clear "GPUs full" message and a retry time |
| FR-11 | Inter-tenant fairness: share GPU capacity fairly between tenants | Must | When one tenant sends 10x the load of another and both keep more requests waiting than their fair share, the busy one gets at most 55% of tokens; the other always gets at least 20% of tokens or everything it asks for, whichever is less; with the other tenant idle a tenant runs at its full speed |
| FR-12 | Per-request audit record: write an audit record for every request | Must | Every request has a record whose document versions and policy version reproduce the answer and a search of the audit store for raw query text finds nothing |

## 5. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Latency (warm) | **Target:** p95 total time ≤4.0 s and p95 time-to-first token ≤800 ms. **Workload:** normal load (60 users; 3 tenants) with at least one node running, on a card no other job is using. **Measured by:** Locust client-side for total time and the model server for time-to-first token, over a 10-minute steady period; median of 3 runs. |
| NFR-2 | Latency (cold start) | **Target:** time-to-first token ≤90 s for 4B and 4-bit 8B models and ≤150 s for 8B models. **Workload:** first request after all GPUs were released. **Measured by:** 20 scripted cold starts per model size, autoscaler timestamps vs client log (node start time and model load time reported separately). |
| NFR-3 | Throughput | **Target:** per node ≥600 output tokens/s for 4B; ≥250 for 8B; ≥450 for 4-bit 8B, for the Qwen3 model of each size (the selected model and its 4B and 4-bit variants); the other models are reported. **Workload:** concurrency 64 on one node, on a card no other job is using (E0). **Measured by:** vLLM token counter over the run. |
| NFR-4 | Scale and elasticity | **Target:** scale from 1 to 2 nodes within 180 s and never above 2; back to 0 within idle timeout +60 s; one node handles 60 concurrent users with ≤1% errors; growing from 60 to 6000 users and 40k to 1M chunks raises warm p95 latency ≤20% with search p95 ≤150 ms. **Workload:** burst up to 3x normal load then no traffic; and growth in users and chunks one at a time. **Measured by:** autoscaler and node timestamps vs Locust arrival log; Locust plus Milvus metrics; 3 runs per point. |
| NFR-5 | Availability | **Target:** ≥99% non-error responses; capacity restored within 180 s after a node failure. **Workload:** 1-hour test at half normal load with one node killed at the 30-minute mark. **Measured by:** gateway logs and node-ready timestamps; requests held for a cold start count as successes. |
| NFR-6 | Cost | **Target:** ≤0.30 GPU-hours per 1000 completed queries including start-up and idle time. **Workload:** normal load and a 24-hour replay of a campus-like traffic pattern. **Measured by:** node allocation times and 1 Hz GPU sampling; reported for all six models, with the target applying to the selected model. |
| NFR-7 | Answer quality | **Target:** answer accuracy ≥80% and citation correctness ≥90%. **Workload:** held-out test questions on the selected model. **Measured by:** rubric grader and LLM judge; humans settle disagreements and check a 60-question sample. |
| NFR-8 | Freshness | **Target:** ≥99% of results use the current document version within 15 minutes of a change. **Workload:** 50 revised documents per hour. **Measured by:** an auditor compares Milvus validity times with source change markers every 60 s. |
| NFR-9 | Security and privacy | **Target:** 0 successful leaks of unauthorised text; ≤5% prompt-injection success; no raw query text stored; zero traffic leaving the institute network. **Workload:** red-team suite of 90 attack cases; and a live 30-minute normal-load run. **Measured by:** rubric grader and LLM judge with humans settling disagreements; hashed-query check of stored data; loopback-only service binding checked by a socket audit, and offline model loading (packet capture needs root, which the server does not give). |
| NFR-10 | Access correctness | **Target:** 0 false allows; ≤1% false denies; revocation within 60 s; decisions in ≤10 ms (authorize) and ≤20 ms (filter). **Workload:** 500 access decisions per second including overload runs. **Measured by:** replay against a reference checker; post-revocation probes; service timing histograms. |

## 6. System architecture and technology stack

```
 DEVICE TIER        DEPARTMENTAL COMPUTE CLUSTER (campus VPN; edge and cloud tiers collapsed into one)
 ┌──────────┐ (1) ┌──────────────────────────────┐ (2) ┌──────────────────────────────────────┐
 │ Readers  │────▶│ Gateway (CPU head node)       │◀──▶│ Access-control service (2 copies)     │
 │ browser/ │     │ password login, SSH tunnel    │     │ rules + directory export; fails closed│
 │ CLI, SSE │◀────│ per-dept limits, 429 "busy"   │     └──────────────────────────────────────┘
 └──────────┘ (5) │ fair turns, output filter,    │ (3) ┌──────────────────────────────────────┐
                  │ request log, answer cache     │◀──▶│ Search (CPU): BGE-base (CPU) +        │
 ┌──────────┐ (7) └──────────────────────────────┘     │ Milvus 2.4 HNSW, allowed+current, top 8│
 │Test tools│           │(4)          ▲(6)              └──────────────────────────────────────┘
 │ Locust,  │           ▼             │                                 ▲ chunks + embeddings
 │ grader + │  ┌────────────────────────────┐  ┌────────────┐  ┌──────────────────────────────┐
 │ judge,   │  │ GPU node pool: 0 to 2 nodes │◀─│ Autoscaler │  │ Document intake + freshness   │
 │ oracle,  │  │ vLLM, one model per node    │  │ add / to 0 │  │ (CPU): chunking, MinIO, every │
 │ Garak    │  │ 6 models: 4B | 8B | 8B 4-bit│  └────────────┘  │ version kept, 5-min checker   │
 └──────────┘  └────────────────────────────┘                   └──────────────────────────────┘
                 Document sources: ~40 dept. sites + shared drive · Observability: Prometheus, Grafana, NVML
```

**Figure 1.** Component diagram and data flow. (1) The question and login token reach the gateway. (2) The gateway asks the access service what this user may read. (3) It finds the 8 best chunks in the search database. (4) It sends the question and chunks to a model on a GPU. (5) The answer streams back with citations, or a "warming up" notice, or a "busy" reply. (6) The autoscaler watches demand and adds or releases GPU nodes. (7) The test tools run offline against the system.

**Tiers and boundaries.** There are two tiers: the user's device side and one departmental cluster. Because everyone is on the campus network, what would normally be separate edge and cloud tiers are combined in the cluster, which keeps internal documents inside the institute. Only arrows 1 and 5 cross between the tiers. Going in: the question (up to 512 tokens) and a login token that carries only the user ID, through an SSH tunnel into the cluster. Coming out: the streamed answer with citations (document, version, chunk, page and an out-of-date flag), a warming-up notice, or a busy reply. Documents, chunks and model files never leave the cluster, and text a user may not read never crosses the boundary.

### Technology stack

| Layer | Technology | What it does and why we use it |
|---|---|---|
| Hardware | Up to 2 of the 3 NVIDIA H200 NVL GPUs (143,771 MiB each, shared with other users) on the CSA automation server, plus its CPU cores | GPUs run the models and are the costly, shared resource. With only two GPUs, the memory actually free on each card decides how many users it can serve at once. CPU cores run everything else, so document updates never use GPU time. |
| Models | Six open models: Qwen3-4B and Phi-4-mini (4B); Qwen3-8B and Granite-3.3-8B (8B); Qwen3-8B-AWQ and Granite-3.3-8B-GPTQ (4-bit 8B) | Lets us measure how model size trades accuracy against cost. "4-bit" means a compressed copy that is smaller, loads faster and leaves room for more users. Models are used as released, with no fine-tuning. |
| Model serving | vLLM | Runs the models efficiently. It serves many users at once (continuous batching) and reuses repeated work such as shared instructions and popular chunks (prefix caching). |
| Search | Milvus 2.4 (HNSW index) and BGE-base embeddings | Finds the 8 chunks closest in meaning to the question, and only among chunks the user may read and that are current. The filtering happens inside the search, not after it. |
| Documents | Python and MinIO | Reads PDFs and web pages, cuts them into chunks and stores the originals. Every version is kept, and a check every 5 minutes retires chunks from replaced documents. |
| Gateway and access | FastAPI, in-memory answer cache, directory export, e-mail and password sign-in (Argon2id), SSH tunnel | The gateway is the front door: it checks login, limits load per department, shares capacity fairly and logs each request. A separate access service, using the directory export, decides who may read what. |
| Autoscaling | Custom Python controller | Adds a second GPU node (the limit is two) when requests rise, and releases all of them after an idle timeout so nothing is charged overnight. The first request after that waits while a node starts. |
| Testing | Locust, rubric grader and LLM judge, access oracle, Garak | Locust simulates many users to test speed and scale. The grader and judge score answer accuracy and citations. The oracle checks permissions, and Garak runs attack tests. |
| Monitoring | Prometheus, Grafana, Loki; a request id and per-stage timers on every request | Shows speed, queue length and GPU use, which feed the cost measurements. |

**Deployment on the CSA automation server.** The server runs Slurm with no Docker, Kubernetes or root access, so the components above run as ordinary processes. Each experiment is one Slurm job: a supervisor starts every service on `127.0.0.1`, the autoscaler starts and stops at most 2 vLLM workers (each pinned to one H200), the workload runs, and everything stops with the job. Three components are swapped for user-space equivalents with the same role: BGE-base is served on CPU with sentence-transformers instead of TEI; GPU metrics come from NVML at 1 Hz instead of DCGM; Milvus Lite stores the chunks and search runs exactly in memory, with the permission and validity filters applied inside the search; and because MinIO no longer publishes community binaries, document originals are stored as files on the server's NFS storage behind the same object-store interface. Model weights and raw results live on the server's NFS storage. The full mapping is in `docs/OPERATIONS.md`.

### Design decisions

**D1. One cluster, not an edge-plus-cloud split.** Internal documents must stay on institute systems, and every user is on one campus network. *Rejected:* splitting into edge and cloud tiers (two systems to run and a privacy question at every hop) and a managed cloud GPU service (documents cannot leave, and it hides the scheduling and node controls we need to study).

**D2. Our own autoscaler driven by request volume.** Arrival rate rises before GPUs look busy, and the idle timeout becomes a setting we can test. *Rejected:* keeping GPUs always on (an idle node overnight dominates cost), scaling by GPU busyness and ready-made scale-to-zero tools (built for small programs, while our models are 5 to 16 GB).

**D3. A separate access service applied inside the search.** One set of rules behind one interface, reusable by other teams. *Rejected:* department-level partitions (cannot express sections, embargoes or per-user rules) and checks written into the gateway (cannot be reused).

## 7. Evaluation plan

**Workload generator.** Locust 2.x in master/worker mode on CPU nodes, separate from the serving path, using a custom user class that parses the SSE stream and records time to first token (TTFT), inter-token latency, end-to-end latency, output-token count, cold-start flag and HTTP status (so 429s from FR-6 count as shed, not as failures).

**Where FR-6 is measured.** The 50 ms limit is judged at the serving node, from a request reaching the server to its 429 being sent (PRD v1.0 measured at the head node; here the serving job is that node); Locust's client-side times are reported alongside. Decided 2026-11-05.

| Workload | Definition |
|---|---|
| W1 (normal load) | 3 tenants with disjoint access groups; 60 concurrent users; think time Exp(mean 10 s), giving ~4.4 req/s; questions drawn from the held-out set with Zipf(α=1.1) popularity, a ~60% repeat rate, so prefix-cache hit rate is a dial rather than an accident; mean prompt ~1,800 tokens (system prompt + 8 chunks × ~190), mean output ~180 tokens. |
| W2 (burst) | W1 scaled to 1×, 2× and 3×. |
| W3 (24-hour trace) | Synthetic campus arrival trace with a 20× peak-to-median ratio and two multi-hour zero-traffic gaps, replayed 24× faster. The gaps exercise the idle timeout and cold start. |
| W4 (growth) | Users 60 → 6,000 and chunks 40k → 1M, one axis at a time. |

Every run is 60 s ramp, 10 min steady state and 60 s drain, repeated 3×; we report the median with min–max whiskers.

**Configurations under comparison.** One hardware setup throughout: the 2-node autoscaling departmental cluster. Only the model varies. `Qwen3-4B-Instruct-2507` is built and validated first as the single-model baseline (`baseline-v1`); the other five then run on the same code.

| Class | Models | Role in the argument |
|---|---|---|
| 4B, BF16 | Qwen3-4B-Instruct-2507, Phi-4-mini-instruct | Cheapest and fastest; tests whether a 4B model's accuracy clears NFR-7. |
| 8B, BF16 | Qwen3-8B, Granite-3.3-8B-Instruct | Accuracy reference. Largest weights, least KV-cache room, slowest cold start. |
| 8B, 4-bit | The same two 8B models as published 4-bit checkpoints: Qwen3-8B-AWQ and Granite-3.3-8B-GPTQ | The hypothesis: 8B-level accuracy at a fraction of its GPU-hours, via ≥2× KV-cache room (FR-8) and faster weight load. |

**Metrics.** Two quality metrics are plotted against three system metrics; the rest are diagnostic.

| Metric | Definition | Instrument |
|---|---|---|
| Answer accuracy | Fraction of answerable held-out questions judged correct against the reference answer | Rubric grader + LLM judge, human adjudication |
| Citation correctness | Fraction of emitted citations whose chunk actually supports the claim | Same, plus a 60-question human audit sample |
| Output tokens/s | Aggregate generation rate at saturating concurrency | vLLM Prometheus counter (NFR-3) |
| GPU-hours / 1k queries | Allocated GPU time per 1,000 completed queries, including start-up and idle-but-held time | Worker lifecycle timestamps + NVML at 1 Hz |
| p95 latency | End-to-end, warm; cold-start latency reported separately | Locust, client-side |

Diagnostic metrics, recorded every run but not plotted as headline results: p50/p99 latency; TTFT and inter-token latency; goodput; admitted vs shed and per-tenant token share; prefix-cache hit rate; KV-cache capacity and maximum batch; active node count; cold starts and the acquisition vs load split; Milvus p95; abstention rate; attack success rate; access-decision latency.

**Experiments.**

| ID | Validates | What is run |
|---|---|---|
| E0 | FR-7, FR-8, NFR-3 | **Saturation sweep and calibration.** Concurrency 1, 2, 4, …, 256 on one pinned node per model, measuring throughput and p95 latency to find each model's knee and its KV-cache capacity. Output: the per-model "rated" concurrency used in every later run, and the check of the NFR-3 targets. |
| E1 | FR-1–FR-4, FR-12, NFR-7, NFR-9, NFR-10 | **Quality and correctness.** The full held-out set: answer accuracy, citation correctness, abstention on unanswerable questions, grader/judge agreement with human adjudication. Also the 10,000 access probes against the oracle, 100 scripted revocations, the 90 red-team attack cases, audit completeness and raw-text absence, and the fail-closed test (access service killed). Produces the quality axis of E2. |
| E2 | FR-8, NFR-1, NFR-3, NFR-6, NFR-7 | **Headline: the accuracy/cost frontier across model sizes.** Every model under W1 on the same cluster at its E0-rated concurrency, three repetitions. Each model's two quality scores from E1 are paired with output tokens/s, GPU-hours per 1,000 queries and p95 latency (Fig. 2). Claim under test: 4-bit 8B keeps 8B accuracy while approaching 4B cost. |
| E3 | FR-9, FR-10, NFR-2, NFR-4, NFR-6 | **Elasticity: autoscaling, idle timeout and cold start.** (a) Scale-out under W2: threshold crossing to node-ready time, ceiling of 2 never exceeded, no request lost during transitions. (b) Idle-timeout sweep T_idle ∈ {0, 60 s, 300 s, 900 s, ∞} replayed against W3 for the best model of each class, recording GPU-hours, cold-start count and cold-start latency. (c) 20 scripted cold starts per model size, node acquisition and weight load separated (Fig. 3). |
| E4 | FR-6, FR-7, FR-11, NFR-1 | **Overload, batching and fairness.** W2 at 1×/2×/3×, each with load limits on/off and fair queueing on/off, plus a continuous-batching and prefix-caching ablation at fixed load. Claims: unbounded queueing collapses where admission control holds a goodput plateau; FR-7 margins hold; no tenant starves another. |
| E5 | FR-5, NFR-5, NFR-8 | **Freshness and resilience.** Replay 60 revisions across 20 documents (and 50 revised documents/hour for NFR-8) while W1 runs at half load: per-chunk time to demotion, unflagged stale answers (target 0), as-of correctness at 60 checkpoints. For NFR-5, run W1 at half load for 1 hour, kill one GPU node at 30 minutes and measure error rate and time to restore capacity, counting held cold-start requests separately from failures. |
| E6 | NFR-4 | **Growth.** W4, one axis at a time, 3 runs per point: warm p95 latency and Milvus search p95. |

**Cost model, per unit of useful work.** For model *m* and idle timeout T_idle, node *i* is held for *aᵢ* seconds of node acquisition, *ℓᵢ* of weight loading, *bᵢ* of serving and *dᵢ* resident but without requests (at most T_idle per scale-down). All are charged, because the department bills node occupancy, not utilisation, and the whole case for scale-to-zero is that *dᵢ* is a cost we can stop paying. With G the GPU-hours and Q the count of completed queries:

  G(m, T_idle) = (1/3600) · Σᵢ (aᵢ + ℓᵢ + bᵢ + dᵢ)  C(m, T_idle) = 1000 · G / Q  (GPU-hours per 1,000 queries)

C is reported three ways: (a) at rated steady load, where *dᵢ* is near zero and models separate on throughput alone; (b) normalised by goodput, so a model cannot buy a good number with slow or shed responses; (c) over the W3 24-hour replay, which charges each model for overnight idle time and every cold start the timeout causes. We also report GPU-hours per 1,000 *correctly answered* queries, since a cheap model that fails NFR-7 is not cheap.

**Expected form of the results.**
- **Fig. 2:** a 2×3 grid. Rows are the two quality metrics, columns are output tokens/s, GPU-hours per 1,000 queries and p95 latency. One marker per model, shape by class, colour by family. The upper-left of the GPU-hours column is the frontier; the conclusion is which model sits on it.
- **Fig. 3:** GPU-hours per 1,000 queries and p95 cold-start latency against T_idle (log x-axis), one line per class. Expected: rising cost crossing falling latency; the knee is the recommended setting.
- W3 timeline: arrival rate, active node count, queue depth and rolling p95 with cold starts marked.
- Goodput vs offered load, admission control on vs off, with the batching and prefix-cache ablations.
- Per-model summary table: weight size, KV-cache capacity, maximum batch, tokens/s, accuracy, citation correctness, GPU-hours/1k, p95 warm and cold.
- Freshness: time to demotion, unflagged stale answers, freshness violation-minutes.

## 8. Work plan, milestones and risks

Owners: RKG = Rudra Kumar Gandhi, MV = Muthangi Vighnesh, MH = Mohammad Haris. Build phases match the `phase-N` branches and tags in the repository.

| Week | Tasks (owner) |
|---|---|
| 1 | Repository, toolchain and CI; shared types and fake LLM server — Phases 0–1 (All). Server spikes (`docs/OPERATIONS.md`): vLLM smoke test on one H200, service binaries without root (Milvus standalone, MinIO, Redis), questions to the admin (RKG). Corpus collection, extraction, chunking, version lineage and reconciler — Phase 2, FR-5 (MV). Held-out question set design and grading rubric (MH). |
| 2 | Access policy engine, service, client and oracle — Phase 3, FR-3, FR-4 (RKG). Milvus schema with version and ACL fields, filtered search — Phase 4, FR-2 (MV). RAG prompt, citations, model config, held-out set v1 — Phase 5, FR-1, FR-8 (MH). |
| 3 | Gateway auth, load limits, fair queueing — Phase 6, FR-6, FR-11 (RKG). Audit, output filter, cache and API wiring — FR-12 (MV). Autoscaler policy and controller — Phase 7, FR-9, FR-10 (RKG, MH). Metrics, tracing and cost model — Phase 8, NFR-6 (MH). → **M1** |
| 4 | Locust workloads W1–W4 (MH); rubric grader and LLM judge (MV); access probes and red-team suite (RKG) — Phase 9. Security hardening — Phase 10 (All). Regression goldens and performance thresholds — Phase 11 (All). |
| 5 | Real-cluster deployment and `baseline-v1`: E0–E6 with Qwen3-4B-Instruct-2507, scorecard — Phase 12 (All). → **M2** |
| 6 | 4-bit models and the six-model sweep: E0, E1, E2 per model and E3 per class — Phase 13 (RKG, MV); figures, report, reproduction package, demo and viva preparation (All). → **M3** |

**Milestones** (each demonstrable at a spot check):

| Milestone | Deliverable |
|---|---|
| M1 — end of week 3 | **Complete system on the fake stack.** A question goes through gateway → access → search → fake LLM → output filter → audit and returns a streamed, cited answer; 429 shedding, fair queueing and the autoscaler's scale-out and scale-to-zero shown in simulation; one real answer from Qwen3-4B on one GPU node. |
| M2 — end of week 5 | **Single-model baseline on the cluster.** `baseline-v1` scorecard with every FR and NFR measured; the pool scaling 1→2 under a burst and back to zero after the idle timeout with metered cost going flat; a cold-started request held and answered rather than failed. |
| M3 — end of week 6 | **Evaluated system.** The 2×3 accuracy-versus-cost frontier (Fig. 2) and the idle-timeout knee (Fig. 3) across six models; the closed traceability table mapping every FR and NFR to a measured result; a reproduction package (deployment files, autoscaler config, Locust files, grading harness, analysis scripts) that regenerates every figure from raw logs. |

**Principal risks and mitigations.**

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | **The cluster will not let us take GPU nodes up and down on demand** — allocation is queued behind other departmental jobs, scale-to-zero is not permitted, or the cluster offers a batch scheduler (for example Slurm) rather than Kubernetes. FR-9, FR-10, NFR-2 and part of NFR-6 depend on it. | Spike node request/release in week 1 and measure the acquisition-latency distribution before committing. If acquisition is unbounded or queued, degrade in order: (1) hold a small reservation and emulate scale-to-zero by stopping and restarting the vLLM process on held nodes — this keeps the weight-load part of cold start and the whole idle-timeout argument, with node acquisition reported as a separate, cluster-imposed constant; (2) failing that, report autoscaling at a fixed pool size. The node-pool interface (README Phase 7) keeps this a config change. E1 and E2 are unaffected because they run at a fixed node count. **Trigger:** end of week 1. **Status (2026-10-18): triggered.** The CSA server runs Slurm with no GPU reservation, no Docker and no Kubernetes; we take mitigation (1), with workers as vLLM processes inside the run's job. |
| R-2 | **Corpus access or licensing** — internal documents cannot be released or lack a text layer. | Build against the public subset (academic regulations, course bulletin, public circulars) plus a generated three-tenant synthetic corpus for the access and injection tests. No systems claim depends on where the corpus came from; only the accuracy axis does, and the held-out set is built from whichever corpus we hold. **Trigger:** end of week 2. **Status:** mitigated; the corpus is the public IISc sites plus 22 synthetic restricted documents for the access tests. |
| R-3 | **The GPU is shared or smaller than assumed.** The PRD assumes ≥144 GB per GPU, but other users' jobs may hold part of it, and the figure still shows 24 GB. Less memory means less KV-cache room, smaller batches, and lower NFR-1, NFR-3 and per-node rated capacity for FR-9. | Confirm the GPU model, memory and exclusivity in week 1; record free memory at the start of every run; set vLLM's memory fraction from what is actually free. Check NFR-3 against E0 and report the measured memory with every result. **Trigger:** end of week 1. **Status (2026-10-18):** the GPU is an H200 NVL with 143,771 MiB, and it is shared: two of the three cards each had 53 GB in use by other jobs. The vLLM memory fraction is now set from free memory at worker start. |
| R-4 | **Server monitors and scheduling.** Jobs below 10% GPU use, or with negligible CPU, for 30 minutes may be killed; services may be read as unauthorised background processes; and CPU (`dev`) jobs wait behind any GPU job. Long quiet periods (scale to zero, idle-timeout sweeps) and separate service jobs are exposed. | Keep every quiet period under 30 minutes of wall time (W3 replayed 24x faster does this); start CPU services inside the GPU job; chain runs with job dependencies; bind every service to `127.0.0.1` inside the job so it stops with the job. **Trigger:** before the first stack run. **Status:** mitigated; every run follows these rules (`docs/OPERATIONS.md`). |

**MVP** — the subset that is a complete project on its own: FR-1 to FR-10 and FR-12; NFR-1, NFR-2, NFR-3, NFR-4 (scale-out and scale-to-zero), NFR-6, NFR-7, NFR-9 and NFR-10; evaluated by E0–E4 on one corpus of ~40k chunks with three tenants. This gives a multi-tenant RAG service on the cluster with continuous batching, prefix-cache reuse, quantized serving, admission control and an autoscaler that scales to zero and back, plus the accuracy-versus-cost frontier across the nine models and the idle-timeout knee. That pair of results is the claim the project exists to make.

**Stretch goals (outside the MVP):** FR-11 fairness at full measurement depth; NFR-5 and NFR-8 full-duration soak (E5); NFR-4 growth to 6,000 users and 1M chunks (E6); E3 idle-timeout sweep for all six models rather than one per class; predictive scale-out that warms a node before registration opens; speculative decoding; an operator console for active nodes, metered GPU-hours, staleness backlog and queue depth.

## 9. References

1. Kwon, W. et al. Efficient Memory Management for LLM Serving with PagedAttention (vLLM). SOSP 2023.
2. Yu, G.-I. et al. Orca: A Distributed Serving System for Transformer-Based Generative Models (continuous batching). OSDI 2022.
3. vLLM documentation: batching, automatic prefix caching, quantization, metrics. Apache-2.0.
4. Lin, J. et al. AWQ: Activation-aware Weight Quantization. MLSys 2024; cf. Frantar, E. et al. GPTQ. ICLR 2023.
5. Wang, J. et al. Milvus. SIGMOD 2021, Apache-2.0; Malkov, Y. & Yashunin, D. HNSW. TPAMI 2018.
6. Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
7. Fu, Y. et al. ServerlessLLM. OSDI 2024; Shahrad, M. et al. Serverless in the Wild. ATC 2020. (FR-10, D2, E3)
8. Greshake, K. et al. Indirect prompt injection. ACM AISec 2023.
9. OWASP Top 10 for LLM Applications, 2025.
10. Model weights: Qwen3 (Apache-2.0); Phi-4-mini (MIT); Granite 3.3 (Apache-2.0); bge-m3 (MIT); bge-reranker-v2-m3 (Apache-2.0); bge-base-en-v1.5 (MIT).
11. Software: Apache-2.0 — vLLM, Milvus, Garak, Prometheus, llm-compressor, sentence-transformers; MIT — Locust, FastAPI; AGPL-3.0 — MinIO, Grafana; RSALv2 — Redis; GPL-2.0 — Slurm (provided by the server).
12. LLM tools (per the course instructions): Claude (Anthropic), used to stress-test scope, surface omitted requirements and check evaluation baselines. All requirements, targets and decisions were reviewed, amended and adopted by the team.
