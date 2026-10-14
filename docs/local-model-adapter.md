# Local vision-model adapter

The development runner connects an installed Ollama vision model to the
[public optical protocol](agent-protocol.md). It retains the model-visible
conversation, raw HTTP responses, emitted actions and the corresponding optical
measurements. The direct-action and estimator-assisted tracks remain separate.
Transport tests use a deterministic service fixture; they are not evidence of
language-model calibration performance.

## Execution

Provide an already installed local vision model and an idle, compatible Ollama
service. The runner does not download models, start a server or alter its settings.
Edit the model name and declared sampler configuration in a copy of
`configs/local_vision_model.json` when using a different installed model.

```sh
uv run --locked python -m probebench.model_agent \
  --model-config configs/local_vision_model.json \
  --session-config configs/agent_protocol_development.json \
  --output work/local-vision-development

uv run --locked python -m probebench.model_agent \
  --verify work/local-vision-development
```

Use the same command with `--resume` after interruption. Source files, numerical
runtime, both configurations, prompt and model identity must match. Verification
and resumption of a completed run do not contact the model service. A failed
completed episode remains a failed episode; it is not retried by `--resume`.

The supplied optical configuration contains a known development wavefront.
Changing its detector seed does not create a held-out wavefront. The current
runner accepts development configurations only.

## Model admission and provenance

Only literal loopback HTTP endpoints are accepted. The client does not follow
redirects or environment proxy settings. It exposes only the version, installed
model, model-detail, running-model and chat endpoints. Explicit remote-model
metadata, cloud-labelled names, missing local GGUF metadata, absent vision
capability and missing model digests are rejected.

The initial service must report no loaded models. Subsequent checks permit only
the selected digest. This is a conservative interference check, not an exclusive
server lock: another client can start after a check, and concurrent use of the
same model cannot be identified this way. Use a dedicated service for comparative
measurements. No unload request is sent.

Before and after each generation, the runner compares the installed model record,
full model metadata and server version with their initial values. This retains
the weight digest, template, declared model parameters and quantization metadata.
It does not independently inspect the server's weight files or authenticate its
responses. The service and host are trusted execution dependencies.

The configurations explicitly declare temperature, seed, context allowance,
generation allowance, nucleus probability, top-k and thinking mode. A fixed seed
is not a guarantee of bitwise repeatability across backend versions or hardware.
The API's token counts and durations remain in the raw response; absent fields
are unavailable measurements, not zero values.

## Observation modality

Each decision supplies the exact fixed-scale camera PNG and a JSON projection of
public instrument, command, exposure and allowance metadata. Signed numeric pixel
arrays are retained by the optical driver but are not placed in the model prompt.
There is no image-dependent normalization, crop, rescaling or hidden-score hint.

In the assisted track, the model receives the estimated OPD vector, cache status,
fit status, optimizer-success flag and declared fallback. The driver retains full
optimizer diagnostics separately. The direct track receives no estimator result.

Every request contains the entire retained conversation, including the current
image and earlier user/assistant messages. A repeated observation is repeated in
the conversation. The adapter does not trim history or summarize decisions.
Returned assistant content is replayed; separate backend thinking fields remain
in the raw response but are not replayed as conversation text.

The backend may impose its own context truncation. Saved request bytes establish
what was sent, not which tokens the backend ultimately processed. A model-specific
context-capacity check is required before interpreting longer episodes. This
adapter does not restore an internal KV cache or sampler state after interruption.

## Action and failure accounting

The backend is asked for JSON. A syntactically valid object is serialized onto one
line without changing its values; the optical driver checks its exact schema,
request identifier, allowed operation and numeric bounds. Duplicate keys and
nonfinite values are rejected. No code, shell command or native model tool call
is executed.

Malformed textual answers are encoded as JSON strings. They remain invalid actions,
consume a request allowance and receive the driver's rejection on the next turn.
There is no code-fence removal, numeric clipping, missing-field repair or replacement
optical decision. The original answer is always retained when a response was received.

HTTP failures, oversized responses/actions, exhausted generation limits, unexpected
native tool calls or model-identity changes terminate model interaction. Closing
the driver's input creates a failed optical outcome at its last measured command.
The result retains both the model failure and the driver's termination reason.
An optical input deadline can independently expire before an action arrives; the
verification checks retain the distinction between generated and consumed actions.

`decision_count` counts retained decision records. `generation_attempts` counts
durably recorded dispatch intents, including an interrupted call whose delivery
is unknown. It is not an exact count of server-completed generations. A failure
before dispatch is excluded from that second count. Raw successful responses provide
the available backend token and duration measurements; they do not fill gaps left
by uncertain calls.

## Interruption semantics

```text
public request -> prepared request -> dispatch intent -> HTTP response receipt
                                                     -> saved decision -> optical action
```

The request is written before metadata checks. A dispatch intent is written before
contacting the generation endpoint. The raw response is written before subsequent
identity checks, and the complete decision is written before sending an action.

- A prepared request without dispatch intent can proceed after admission checks.
- A dispatch intent without a response receipt becomes an explicit unknown-outcome
  failure. It is never automatically generated again.
- A partially published receipt is retained as an `.incomplete` artifact. Its bytes
  do not authorize an action or a repeated generation.
- A saved response can be revalidated without repeating generation.
- A saved action is replayed when the optical driver requests that same observation.
- A completed run verifies and returns without new camera or model calls.

Each HTTP operation runs in a separate client process with a wall-clock deadline
and a bounded response body. Terminating that client does not prove that the server
cancelled an in-flight generation. The adapter never retries an uncertain call.

## Evidence and verification scope

The output contains source/runtime bindings, an initial model record, signed
decision records, any raw pending receipts, the complete optical session and a
final result. Decision records form a hash chain. A completed manifest is written
only after the model/driver exchange has been checked.

Verification reconstructs each sent prompt from its retained public observation
and conversation history, derives the emitted action from the raw API response,
matches it to the driver's retained input, and verifies the optical replay and
final physics through the existing session verifier. Missing or reordered turns,
changed responses, altered actions and substituted observations are rejected.
Hashes establish internal consistency, not authorship or independent proof that
an external model produced a response.

The study reserves storage for bounded requests, responses and public messages
before starting. This admission does not guarantee sufficient accelerator memory.
Model-specific memory, context and latency qualification is separate from protocol
correctness. Journals and model metadata can contain sensitive text or machine
paths; they are not automatically suitable for distribution. See [security](../SECURITY.md).

The client follows the documented [chat API](https://docs.ollama.com/api/chat),
[vision input format](https://docs.ollama.com/capabilities/vision),
[model metadata](https://docs.ollama.com/api-reference/show-model-details),
[installed-model inventory](https://docs.ollama.com/api/tags) and
[running-model inventory](https://docs.ollama.com/api/ps).
