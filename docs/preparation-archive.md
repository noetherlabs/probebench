# Full-precision prior records

`probebench.preparation_archive` defines a versioned byte format for future
preparation archives. It retains every prior record, including primary and
conjugate coefficient vectors, record identities and RMS metadata. Configurations
and source documents remain attached. Reading an archive requires no sampler,
optical model, numerical libraries or filesystem access.

A configuration and cluster fingerprint alone cannot restore missing records.
Older preparation files retain summaries of some prior cohorts; those summaries
do not satisfy this format. Existing study commands and v1 archive readers remain
unchanged. The new boundary is intended for a separately admitted future
preparation that still owns its complete prior records.

## Declaration and custody

The declaration schema is `probebench.prior-record-declaration/1`. It contains
raw source-file fingerprints and an ordered `priors` list. Each prior declares:

| Field | Meaning |
| --- | --- |
| `role`, `index` | `required_prior` or `additional_prior`, with contiguous indices starting at zero within each role |
| `source` | Name of an attached JSON source document |
| `configuration_pointer` | JSON pointer to the prior configuration in that document |
| `clusters_sha256_pointer` | JSON pointer to its declared canonical full-record fingerprint |
| `configuration_sha256`, `clusters_sha256` | Canonical configuration and complete ordered record identities |
| `record_count` | Number of complete base records |

Required priors precede additional priors. Role/index assignments and both source
pointers are part of the frozen declaration. Source pointers must reproduce the
declared fingerprints; supplying independently authentic but mismatched
configurations and records is refused.

```python
from probebench.preparation_archive import build_archive, read_archive

packet = build_archive(
    preparation_bytes,
    declaration_bytes=frozen_declaration_bytes,
    expected_declaration_sha256=frozen_declaration_sha256,
    sources=source_bytes_by_name,
    records=original_records_by_role_and_index,
)
verified = read_archive(
    packet["manifest_bytes"],
    files=packet["files"],
    expected_manifest_sha256=packet["manifest_sha256"],
    expected_declaration_sha256=frozen_declaration_sha256,
)
```

`original_records_by_role_and_index` maps tuples such as `("required_prior", 0)`
to the original ordered record lists. The writer serializes them without rounding
into chunks of at most 256 records. A smaller `chunk_records` value is supported
within the file-count allowance. The reader reconstructs the complete canonical
record fingerprint across chunk boundaries and verifies each configuration,
record identity, member, source pointer and raw file identity. Its `prior_chunks`
result contains detached immutable bytes.

This format retains the existing six-coefficient cohort record schema. It does
not reinterpret coefficients or verify the reported physical RMS. Generic
`preparation_bytes` are checked as bounded JSON and preserved byte-for-byte;
their scientific relationship to the prior declaration is not inferred. The
trusted study binding must establish that relationship.

The manifest schema is `probebench.preparation-archive/2`. Files are bounded to
1 MiB each, the manifest and files together to 64 MiB, and the inventory to 400
files, 32 priors and 65,536 records. Each prior is limited to 4,096 records. Missing,
truncated, reordered, altered or additional files are refused. A v1 hash-only
summary is never silently upgraded.

The caller must obtain the expected declaration and manifest fingerprints from a
trusted study binding. A digest supplied alongside untrusted bytes is not
independent authentication. Within its existing storage admission and exclusive
ownership, a future writer must publish every returned file and retain the
manifest as its completion record; storage estimates must include these bytes.
This module does not publish files or authorize acquisition.

`declared_records_complete` certifies the attached declared records. It does not
establish optical correctness, independence, undisclosed-history coverage or
hardware reliability. `historical_global_closure_established` and
`execution_admitted` remain false.
