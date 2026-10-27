# Security

## Execution model

ProbeBench is a local scientific simulator. Python policies and archived source execute
with the permissions of the current process. Observation/evaluator separation is an API
contract, not process isolation. Do not execute untrusted policies or archives in a
process that can access private evaluation state, credentials or unrelated files.

The JSON-lines agent driver uses an explicit public-field allowlist and validates
actions before camera mutations. Its request identifiers prevent stale actions;
they are not authentication tokens. A trusted client receives only standard-output
protocol messages, not driver configuration, evaluator files or diagnostic output.
The client still needs separate OS/container isolation before hostile code is admitted.
Saved inputs can contain arbitrary client text, including secrets; retaining a valid
scientific manifest does not make a transcript safe to publish.

The local model adapter accepts literal loopback HTTP endpoints and checks installed
weight metadata before generation. It does not authenticate the service, isolate its
process, or prevent a trusted server from making network requests of its own. Use a
locally configured service with cloud features disabled when network isolation is
required. Model outputs are data: only validated optical JSON actions are forwarded.
No returned code or native tool call is executed. An HTTP deadline stops the client;
it does not establish cancellation of server-side work. Model metadata and transcripts
must be reviewed separately before distribution.

NumPy measurements are loaded with object deserialization disabled. Artifact verification
checks file membership, path containment and content hashes. It does not authenticate an
archive's author, prevent a malicious party from replacing both data and hashes, or bound
the resource use of an untrusted compressed file.

Sequential verification requires exactly the episode and exposure inventory derived
from the study configuration, even when extra files have valid manifest hashes.
Export and analysis tools use that verification before processing records. Keep input
archives unchanged during verification and export; these local tools do not defend
against another process concurrently replacing files. Archived source snapshots retain
their original behavior and do not inherit fixes in the current package.

## Maintained dependencies and acquisition snapshots

Use the root `uv.lock` for the simulator and
`requirements/huggingface-validation.txt` for the optional dataset loader.
These environments use Pillow 13.0.0; the dataset loader also uses filelock 4.2.4,
which address the corresponding upstream security advisories. See the
[Pillow release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)
and [filelock release](https://github.com/tox-dev/filelock/releases/tag/4.2.4).

The five `results/*/source/uv.lock` files retain Pillow 12.3.0 and its known
vulnerabilities. Their bytes are part of immutable acquisition manifests, not
supported runtime recommendations. Do not install those historical locks to
process untrusted inputs. Run verification and new work in the maintained root
environment. Dependency updates do not retroactively change recorded results or
establish byte-identical reproduction of historical image encodings. The
historical dependency alerts remain an explicit limitation.

## Reporting a vulnerability

Email [crew@noether.one](mailto:crew@noether.one) with the affected revision, impact,
and a minimal reproduction. Do not post exploitable details in a public issue before
maintainers have reviewed the report.

## Repository checks

Use `scripts/check_repository.py` to detect prohibited tracked metadata and local paths,
and Gitleaks to scan for recognized credential patterns. Audit pinned dependencies against
a current advisory database before a release. These checks have different scopes and
none establishes that arbitrary code is vulnerability-free.

Keep security reports and exploit details private until maintainers have reviewed them.
Do not include credentials or private datasets in issue reports, logs or reproductions.
