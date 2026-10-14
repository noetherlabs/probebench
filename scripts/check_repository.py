"""Check staged files or a commit for material unsuitable for distribution."""
import argparse
from pathlib import PurePosixPath
import re
import subprocess


EXCLUDED_NAMES = {"AGENTS.md", "PLAN.md", "STATUS.md", "DECISIONS.md"}
EXCLUDED_DIRECTORIES = {".codex", ".gstack", ".venv", "__pycache__", ".pytest_cache", "work"}
HOME_PATH = re.compile("/" + r"(?:Users|home)/[^/\s\"'<>]+")
LOCAL_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.local\b")
COORDINATION = re.compile(
    r"\b(?:Codex|Claude|gstack|workspace-credit|hourly continuation|research factory)\b", re.I)


def inspect_files(files):
    findings = []
    for name, content in files.items():
        path = PurePosixPath(name)
        if (path.name in EXCLUDED_NAMES or EXCLUDED_DIRECTORIES.intersection(path.parts)
            or path.suffix in {".pem", ".key"}
            or (path.name.startswith(".env") and path.name != ".env.example")
            or (name.startswith("results/verification-") and path.suffix == ".xml")):
            findings.append(f"{name}: excluded local or sensitive file")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if HOME_PATH.search(text) or LOCAL_HOST.search(text):
            findings.append(f"{name}: personal filesystem path or local hostname")
        if path.suffix.lower() == ".md" and COORDINATION.search(text):
            findings.append(f"{name}: internal coordination terminology")
    return findings


def tracked_files(ref=None):
    if ref is None:
        raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"])
    else:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], text=True).strip()
        raw = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision])
    files = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, second, third = header.decode().split()
        path = name.decode()
        if mode != "100644" and mode != "100755":
            raise ValueError(f"{path}: non-regular tracked file requires explicit review")
        if ref is None and third != "0":
            raise ValueError(f"{path}: unresolved index entry")
        blob = second if ref is None else third
        files[path] = subprocess.check_output(["git", "cat-file", "blob", blob])
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Check this commit instead of the index")
    args = parser.parse_args()
    files = tracked_files(args.ref)
    findings = inspect_files(files)
    if findings:
        parser.exit(1, "\n".join(findings) + "\n")
    print(f"Repository content check passed: {len(files)} tracked files.")


if __name__ == "__main__":
    main()
