# Security policy

## Reporting a vulnerability

Please report security or privacy findings privately:

1. On GitHub, open the **Security** tab and choose **Report a
   vulnerability**. This creates a private advisory that only the
   maintainers can see.
2. Tell the project mentor as well.

Do not put secrets, credentials, personal data or working exploit steps in
a public issue. If a public issue is needed, it should hold only a short,
sanitised summary that points to the private report.

We acknowledge private reports within 12 hours and rotate any exposed
credential at once.

## How the deployment is protected

- Every service listens on `127.0.0.1` only. People reach the chat app,
  Grafana, Prometheus and the Locust UI through their own SSH tunnel; the
  last three need a password created for each deployment.
- Every user of the shared server can reach a loopback port. The vLLM
  workers and the embedding server therefore accept only calls that carry
  the serving job's key (`Authorization: Bearer`). The key is made fresh
  for each serving job and passed to them in their environment, never on a
  command line. It is checked by our middleware on every path
  (`src/askiisc/common/servicekey.py`), because vLLM's own `--api-key`
  leaves `/invocations` open. Health checks and the workers' Prometheus
  counters stay open.
- The access service checks a signed token (RS256, a new key pair for each
  serving job) on every decision and fails closed.
- Secrets live only in `private/app/` on the server (mode 600), never in
  this repository; gitleaks runs over the whole history in the gate.

## Known advisories in the GPU-side packages

`make audit` checks the project's own lock file and reports no known
vulnerability (soupsieve moved to 4.1 on 2026-11-13 for CVE-2026-126190). The GPU-side environment (`slurm/serve-requirements.txt`)
is separate. On 2026-11-12, pip-audit listed advisories for vLLM 0.11.0
(93), torch 2.8.0 (9), transformers 4.57.6 (9), setuptools (3), xgrammar
(2), diskcache (2) and sentence-transformers 5.1.1 (1). The embedding
library moved to sentence-transformers 5.7.0, which fixes its advisory and
gives the same embeddings and rerank order. None of the others can be
reached through Ask IISc:

- Only the gateway can call a worker or the embedding server: without the
  serving job's key every call is refused, on every route.
- The gateway sends one kind of request: a streamed chat completion with
  fixed parameters (model, token limit, temperature, prefix-cache salt and,
  for Qwen3, `enable_thinking: false`). It never sends prompt embeddings,
  images, audio, tools, structured-output grammars, log-probabilities or a
  chat template. A user's input reaches vLLM only as message text: the
  question (at most 512 words) and up to 6 earlier turns of the same chat.
- Models are loaded offline (`HF_HUB_OFFLINE=1`) from fixed names in
  `src/askiisc/llm/models.yaml`, so no user can make a worker or the
  embedding server load a model or code from a path of their choosing.

We keep vLLM at 0.11.0 for this evaluation because every measured result
(freshness, latency, throughput, cold starts, answer quality) was taken
with it. Moving to a fixed release (0.32 or later) changes the scheduler
and the kernels, so all of them would have to be measured again; that is
planned after the evaluation, together with torch and transformers.
