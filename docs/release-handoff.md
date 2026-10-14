# Release handoff

Inspected on 2026-10-14 using the live browser. Website implementation is assigned to
Claude; this file records research-package requirements and the requested reference style.

## The three linked reference releases

- [SplatBench](https://perpetualresearch.si/splatbench/): completed benchmark article with
  a concrete question, immediately visible results, precise observation/action contract,
  hidden grading, conventional references, interactive episode replay, limitations and
  access/citation block. Its actual links are a
  [Hugging Face Explorer Space](https://huggingface.co/spaces/perpetual-research/splatbench-explorer)
  and [dataset](https://huggingface.co/datasets/perpetual-research/splatbench).
  The user's spoken “Dash” reference corresponds most plausibly to this explorer/dataset
  pair; no named Dash product was found in the release links.
- [Vibemotion](https://perpetualresearch.si/vibemotion/): a sandbox release built around a
  video demonstration and the common action log used by humans and agents. Quantitative
  product claims are explicitly dated. Reuse the evidence-first structure, not its metrics.
- [MotionBench](https://perpetualresearch.si/motionbench/): explicitly forthcoming, with
  three research questions and no published results. It is a useful example of clearly
  distinguishing proposed work from a completed release.

Live computed typography on SplatBench is **EB Garamond Variable** for its title and
**Geist Variable** for body text; both families also appear on the other release pages.
Exact component typography belongs in Claude's site implementation. Original Noether
prose should use short question-led introductions, concrete experimental descriptions,
figures tied to raw trajectories and explicit limits; do not copy the reference prose.

## Noether release acceptance

The article must be generated or checked against a frozen result manifest, show every
declared method/condition, explain failures and uncertainty, and make its environment,
baselines, budget accounting and exclusions inspectable. Show the probe's temporary harm
and examples chosen by declared rules. Keep direct-action and tool-assisted agent tracks
separate. Link an actual working replay and loader rather than a future placeholder.

The requested citation contains title, author, year and the **final website URL**. Use
Noether Labs only as organizational attribution, with no unverified affiliation. Do not
invent a published URL before Claude's route exists. The user-provided contact is
interpreted as `crew@noether.one`; verify the address before final public copy. No email
has been sent.

## Hugging Face packaging contract

Hugging Face renders the repository README as a dataset card; YAML metadata describes
the dataset, files and selected licence. Document synthetic origin, schema, splits,
limitations and intended use. See the opened official
[dataset-card documentation](https://huggingface.co/docs/hub/datasets-cards).

Package versioned machine-readable records and images in an officially supported format,
and test the exact loading command against the packaged files and later the uploaded
revision. The official [dataset loading guide](https://huggingface.co/docs/datasets/loading)
supports file-based loading and revision-pinned Hub loading. A Space is a separate demo
artifact; uploading a dataset alone does not provide the interactive explorer.

Current implementation provides local NPZ/JSON scientific records only. HF conversion,
schema, loader, Space, organization identity, licence and public upload are pending.
Keep evaluator truth out of live agent transport; post-run development/research exports
may deliberately include it with explicit field names. Do not publish secret final
evaluation seeds in a controller-facing package.
