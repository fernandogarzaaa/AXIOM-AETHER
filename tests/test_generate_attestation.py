"""generate_attestation.py: SBOM and provenance round trips.

Behavior tests only: generate an attestation for a real file, verify
it, tamper with the file, and confirm verification fails. Tests call
the module in-process so coverage and mutation testing see the code;
one test runs the real CLI entry point as a subprocess smoke check.
The syft path is covered with a stub syft on PATH plus a real-syft
test that skips when syft is not installed.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_attestation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_attestation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def run_cli(*argv):
    """Run the module entry point in-process; returns its exit code."""
    return MODULE.main([str(a) for a in argv])


@pytest.fixture()
def artifact(tmp_path):
    content = tmp_path / "contents.txt"
    content.write_bytes(b"fake release archive contents\n")
    target = tmp_path / "demo-artifact.tar.gz"
    with tarfile.open(target, "w:gz") as tar:
        tar.add(content, arcname="contents.txt")
    return target


def test_generate_then_verify_passes(artifact, capsys):
    assert run_cli("generate", "--artifact", artifact, "--repo-root", REPO_ROOT) == 0
    sbom = json.loads((Path(str(artifact) + ".sbom.json")).read_text())
    assert sbom["spdxVersion"] == "SPDX-2.3"
    assert sbom["creationInfo"]["created"] != ""
    assert sbom["creationInfo"]["created"].endswith("Z")
    assert (
        run_cli(
            "verify",
            "--artifact",
            artifact,
            "--provenance",
            str(artifact) + ".provenance.json",
            "--sbom",
            str(artifact) + ".sbom.json",
            "--repo-root",
            REPO_ROOT,
        )
        == 0
    )
    assert "VERIFY: PASS" in capsys.readouterr().out


def test_tampered_artifact_fails_verify(artifact, capsys):
    run_cli("generate", "--artifact", artifact, "--repo-root", REPO_ROOT)
    tampered = artifact.with_name("tampered.tar.gz")
    tampered.write_bytes(artifact.read_bytes() + b"tampered\n")
    assert (
        run_cli(
            "verify",
            "--artifact",
            tampered,
            "--provenance",
            str(artifact) + ".provenance.json",
            "--sbom",
            str(artifact) + ".sbom.json",
            "--repo-root",
            REPO_ROOT,
        )
        == 1
    )
    assert "VERIFY: FAIL" in capsys.readouterr().out


def test_syft_required_fails_when_syft_absent(artifact, monkeypatch, tmp_path):
    empty_bin = tmp_path / "emptybin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    assert (
        run_cli(
            "generate",
            "--artifact",
            artifact,
            "--repo-root",
            REPO_ROOT,
            "--syft-required",
        )
        != 0
    )
    assert not Path(str(artifact) + ".sbom.syft.json").exists()


def _stub_syft(bin_dir, body):
    stub = bin_dir / "syft"
    stub.write_text(body)
    stub.chmod(0o755)
    return bin_dir


def test_syft_required_fails_when_syft_writes_nothing(artifact, monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub_syft(bin_dir, "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    assert (
        run_cli(
            "generate",
            "--artifact",
            artifact,
            "--repo-root",
            REPO_ROOT,
            "--syft-required",
        )
        != 0
    )
    assert not Path(str(artifact) + ".sbom.syft.json").exists()


def test_syft_required_fails_when_syft_output_is_not_spdx(
    artifact, monkeypatch, tmp_path
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub_syft(
        bin_dir,
        "#!/bin/sh\n"
        'for a in "$@"; do case "$a" in spdx-json=*) echo \'not json\' > "${a#spdx-json=}";; esac; done\n'
        "exit 0\n",
    )
    monkeypatch.setenv("PATH", str(bin_dir))
    assert (
        run_cli(
            "generate",
            "--artifact",
            artifact,
            "--repo-root",
            REPO_ROOT,
            "--syft-required",
        )
        != 0
    )


@pytest.mark.skipif(shutil.which("syft") is None, reason="syft not installed")
def test_real_syft_produces_valid_spdx(artifact):
    assert (
        run_cli(
            "generate",
            "--artifact",
            artifact,
            "--repo-root",
            REPO_ROOT,
            "--syft-required",
        )
        == 0
    )
    syft_sbom = json.loads((Path(str(artifact) + ".sbom.syft.json")).read_text())
    assert syft_sbom["spdxVersion"].startswith("SPDX-")


def test_attest_image_records_digest(tmp_path):
    out = tmp_path / "image.provenance.json"
    digest = "a" * 64
    assert (
        run_cli(
            "attest-image",
            "--image",
            "ghcr.io/example/axiom-aether:test",
            "--digest",
            f"sha256:{digest}",
            "--out",
            out,
            "--repo-root",
            REPO_ROOT,
        )
        == 0
    )
    prov = json.loads(out.read_text())
    assert prov["subject"][0]["digest"]["sha256"] == digest
    assert prov["subject"][0]["name"] == "ghcr.io/example/axiom-aether:test"
    assert (
        prov["predicate"]["buildDefinition"]["externalParameters"]["imageDigest"]
        == f"sha256:{digest}"
    )


def test_attest_image_rejects_bad_digest(tmp_path):
    assert (
        run_cli(
            "attest-image",
            "--image",
            "ghcr.io/example/axiom-aether:test",
            "--digest",
            "not-a-digest",
            "--out",
            tmp_path / "out.json",
            "--repo-root",
            REPO_ROOT,
        )
        == 2
    )


def test_cli_entry_point_smoke(artifact, tmp_path):
    """One real subprocess run: the `python3 script ...` path works."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "generate",
            "--artifact",
            str(artifact),
            "--repo-root",
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (Path(str(artifact) + ".provenance.json")).exists()


def test_generate_missing_artifact_exits_2(tmp_path):
    assert run_cli("generate", "--artifact", tmp_path / "nope.tar.gz") == 2


def test_verify_missing_artifact_exits_2(tmp_path):
    assert (
        run_cli(
            "verify",
            "--artifact",
            tmp_path / "nope.tar.gz",
            "--provenance",
            tmp_path / "nope.json",
        )
        == 2
    )


def test_verify_unreadable_provenance_exits_2(artifact, tmp_path):
    assert (
        run_cli(
            "verify",
            "--artifact",
            artifact,
            "--provenance",
            tmp_path,
        )
        == 2
    )


def test_missing_required_args_exit_2():
    with pytest.raises(SystemExit) as exc:
        run_cli("generate")
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run_cli("verify", "--artifact", "x.tar.gz")
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run_cli("attest-image", "--image", "ghcr.io/x/y:z")
    assert exc.value.code == 2


def test_git_helpers_return_fallbacks_when_git_fails(monkeypatch):
    def failing_run(*args, **kwargs):
        raise subprocess.SubprocessError("git gone")

    monkeypatch.setattr(MODULE.subprocess, "run", failing_run)
    assert MODULE.git_commit(str(REPO_ROOT)) == ""
    assert MODULE.git_status_clean(str(REPO_ROOT)) is False


def test_run_syft_timeout_returns_false(monkeypatch, artifact, tmp_path):
    monkeypatch.setattr(MODULE.shutil, "which", lambda _: "/bin/syft")

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="syft", timeout=300)

    monkeypatch.setattr(MODULE.subprocess, "run", timeout_run)
    assert MODULE.run_syft(str(artifact), str(tmp_path / "out.json")) is False


def test_run_syft_failing_exit_returns_false(monkeypatch, artifact, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub_syft(bin_dir, "#!/bin/sh\necho boom >&2\nexit 1\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    assert MODULE.run_syft(str(artifact), str(tmp_path / "out.json")) is False


def test_run_syft_empty_output_returns_false(monkeypatch, artifact, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub_syft(bin_dir, "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    assert MODULE.run_syft(str(artifact), str(tmp_path / "out.json")) is False


def test_provenance_records_commit_builder_and_materials(artifact):
    run_cli(
        "generate",
        "--artifact",
        artifact,
        "--repo-root",
        REPO_ROOT,
        "--builder",
        "test-builder",
    )
    prov = json.loads((Path(str(artifact) + ".provenance.json")).read_text())
    params = prov["predicate"]["buildDefinition"]["externalParameters"]
    # Expected commit comes from the same helper the tool uses, so the
    # test also passes in a testteeth work copy without a .git dir.
    assert params["sourceCommit"] == MODULE.git_commit(str(REPO_ROOT))
    assert params["builder"] == "test-builder"
    assert prov["predicate"]["runDetails"]["builder"]["id"] == "test-builder"
    recorded = {
        m["uri"] for m in prov["predicate"]["buildDefinition"]["resolvedDependencies"]
    }
    expected = {p for p in MODULE.MATERIAL_PATHS if (REPO_ROOT / p).is_file()}
    assert recorded == expected


def test_generate_honors_custom_output_paths(artifact, tmp_path):
    sbom_out = tmp_path / "custom.sbom.json"
    prov_out = tmp_path / "custom.prov.json"
    assert (
        run_cli(
            "generate",
            "--artifact",
            artifact,
            "--repo-root",
            REPO_ROOT,
            "--sbom-out",
            sbom_out,
            "--provenance-out",
            prov_out,
        )
        == 0
    )
    assert sbom_out.is_file()
    assert prov_out.is_file()


def test_verify_changed_material_fails(artifact, capsys):
    run_cli("generate", "--artifact", artifact, "--repo-root", REPO_ROOT)
    prov_path = Path(str(artifact) + ".provenance.json")
    prov = json.loads(prov_path.read_text())
    prov["predicate"]["buildDefinition"]["resolvedDependencies"][0]["digest"][
        "sha256"
    ] = "0" * 64
    forged = artifact.with_name("forged.provenance.json")
    forged.write_text(json.dumps(prov))
    assert (
        run_cli(
            "verify",
            "--artifact",
            artifact,
            "--provenance",
            forged,
            "--repo-root",
            REPO_ROOT,
        )
        == 1
    )
    assert "FAIL: material changed" in capsys.readouterr().out


def test_verify_wrong_sbom_checksum_fails(artifact, capsys):
    run_cli("generate", "--artifact", artifact, "--repo-root", REPO_ROOT)
    sbom_path = Path(str(artifact) + ".sbom.json")
    sbom = json.loads(sbom_path.read_text())
    sbom["packages"][0]["checksums"][0]["checksumValue"] = "0" * 64
    forged = artifact.with_name("forged.sbom.json")
    forged.write_text(json.dumps(sbom))
    assert (
        run_cli(
            "verify",
            "--artifact",
            artifact,
            "--provenance",
            str(artifact) + ".provenance.json",
            "--sbom",
            forged,
            "--repo-root",
            REPO_ROOT,
        )
        == 1
    )
    assert "FAIL: SBOM artifact checksum mismatch" in capsys.readouterr().out


def test_verify_missing_material_is_a_note_not_a_failure(artifact, tmp_path, capsys):
    run_cli("generate", "--artifact", artifact, "--repo-root", REPO_ROOT)
    assert (
        run_cli(
            "verify",
            "--artifact",
            artifact,
            "--provenance",
            str(artifact) + ".provenance.json",
            "--sbom",
            str(artifact) + ".sbom.json",
            "--repo-root",
            tmp_path,
        )
        == 0
    )
    assert "NOTE: material missing from tree" in capsys.readouterr().out


def test_attest_image_accepts_bare_and_uppercase_digest(tmp_path):
    out = tmp_path / "image.provenance.json"
    assert (
        run_cli(
            "attest-image",
            "--image",
            "ghcr.io/example/axiom-aether:test",
            "--digest",
            "A" * 64,
            "--out",
            out,
            "--repo-root",
            REPO_ROOT,
        )
        == 0
    )
    prov = json.loads(out.read_text())
    assert prov["subject"][0]["digest"]["sha256"] == "a" * 64
