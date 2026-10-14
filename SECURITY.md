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

NumPy measurements are loaded with object deserialization disabled. Artifact verification
checks file membership, path containment and content hashes. It does not authenticate an
archive's author, prevent a malicious party from replacing both data and hashes, or bound
the resource use of an untrusted compressed file.

## Repository checks

Use `scripts/check_repository.py` to detect prohibited tracked metadata and local paths,
and Gitleaks to scan for recognized credential patterns. Audit pinned dependencies against
a current advisory database before a release. These checks have different scopes and
none establishes that arbitrary code is vulnerability-free.

Keep security reports and exploit details private until maintainers have reviewed them.
Do not include credentials or private datasets in issue reports, logs or reproductions.
