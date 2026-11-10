"""Resolve package requirements without installing or executing old image codecs."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_package_install_replaces_vulnerable_pillow(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        import pytest
        pytest.skip("The offline installer regression requires uv")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    versions = {package["name"]: package["version"] for package in packages}
    dependencies = project["dependencies"]
    # An isolated environment contains inert metadata only: no vulnerable Pillow
    # distribution, NumPy, SciPy or matplotlib code is installed or imported.
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    site = Path(result.stdout.strip())
    assert site.resolve().is_relative_to(environment.resolve())
    for name in ("numpy", "scipy", "matplotlib", "pillow"):
        version = "12.3.0" if name == "pillow" else versions[name]
        metadata = site / f"{name}-{version}.dist-info"
        metadata.mkdir()
        # This is matplotlib 3.11.2's published Pillow edge; unrelated edges
        # are omitted to isolate the image-codec dependency decision.
        edge = "Requires-Dist: pillow>=9\n" if name == "matplotlib" else ""
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{edge}"
        )
        (metadata / "RECORD").write_text("")

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    with zipfile.ZipFile(wheels / "pillow-13.0.0-py3-none-any.whl", "w") as wheel:
        prefix = "pillow-13.0.0.dist-info/"
        wheel.writestr(prefix + "METADATA",
                       "Metadata-Version: 2.1\nName: pillow\nVersion: 13.0.0\n")
        wheel.writestr(prefix + "WHEEL",
                       "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        wheel.writestr(prefix + "RECORD", "")

    child_env = {key: value for key, value in os.environ.items()
                 if not key.startswith("UV_") and key not in {"PYTHONHOME", "PYTHONPATH"}}
    child_env.update(UV_PYTHON_DOWNLOADS="never", UV_CACHE_DIR=str(tmp_path / "cache"))

    def resolve(requirements, name):
        path = tmp_path / name
        path.write_text("\n".join(requirements) + "\n")
        result = subprocess.run(
            [uv, "pip", "install", "--dry-run", "--offline", "--no-index", "--no-cache",
             "--no-config", "--python", str(python), "--find-links", str(wheels),
             "--requirement", str(path)],
            env=child_env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout + result.stderr

    # Matched negative control: the old transitive-only graph permits 12.3.0.
    transitive_only = [item for item in dependencies
                       if not re.match(r"(?i)^pillow(?:\W|$)", item)]
    baseline = resolve(transitive_only, "transitive-only.txt")
    assert not re.search(r"(?m)^\s*[+-] pillow==", baseline), baseline
    protected = resolve(dependencies, "package.txt")
    assert re.search(r"(?m)^\s*- pillow==12\.3\.0\s*$", protected), protected
    assert re.search(r"(?m)^\s*\+ pillow==13\.0\.0\s*$", protected), protected
    # Both resolutions are dry runs: even the inert old metadata stays intact.
    assert (site / "pillow-12.3.0.dist-info/METADATA").is_file()
    assert not (site / "pillow-13.0.0.dist-info").exists()
