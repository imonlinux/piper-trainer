"""Tests for train: build_command, epoch resolution, latest_checkpoint."""
import os
import time
from pathlib import Path

import pytest

from piper_trainer.config import Project, TIERS
from piper_trainer.train import (build_command, check_resume_ceiling,
                                 latest_checkpoint, resolve_max_epochs)


def make_project(tmp_path: Path) -> Project:
    return Project(root=tmp_path, name="marvin")


def flag_value(cmd: list[str], flag: str) -> str | None:
    if flag in cmd:
        return cmd[cmd.index(flag) + 1]
    return None


# ---------------------------------------------------------------- build_command

def test_medium_tier_has_no_architecture_flags(tmp_path):
    cmd = build_command(make_project(tmp_path), tier="medium")
    # sample_rate is a data attribute emitted for every tier; the medium
    # tier's architecture is piper1-gpl's default and needs no --model.* override
    arch = [c for c in cmd if c.startswith("--model.")
            and c != "--model.sample_rate"]
    assert arch == []
    assert flag_value(cmd, "--model.sample_rate") == \
        str(TIERS["medium"]["sample_rate"])


def test_build_command_selects_the_csv_logger(tmp_path):
    """The live loss curve's data source (§6.4): CSVLogger aimed at the
    same runs dir, so lightning_logs/version_N/checkpoints stays where
    latest_checkpoint globs it."""
    import json

    proj = make_project(tmp_path)
    cmd = build_command(proj, tier="medium")
    assert flag_value(cmd, "--trainer.logger") == "CSVLogger"
    kwargs = json.loads(flag_value(cmd, "--trainer.logger.dict_kwargs"))
    assert kwargs == {"save_dir": str(proj.runs("medium"))}


def test_low_and_high_tiers_emit_their_architecture_flags(tmp_path):
    for tier in ("low", "high"):
        cmd = build_command(make_project(tmp_path), tier=tier)
        for key in TIERS[tier]["model_args"]:
            assert f"--model.{key}" in cmd, (tier, key)
            assert flag_value(cmd, f"--model.{key}") == TIERS[tier]["model_args"][key]
        assert flag_value(cmd, "--model.sample_rate") == \
            str(TIERS[tier]["sample_rate"])


def test_resume_produces_ckpt_path_and_suppresses_warmstart(tmp_path):
    proj = make_project(tmp_path)
    cmd = build_command(proj, resume=proj.runs("medium") / "last.ckpt",
                        warmstart=proj.checkpoints / "base.ckpt")
    assert flag_value(cmd, "--ckpt_path") == str(proj.runs("medium") / "last.ckpt")
    assert "--model.warmstart_ckpt" not in cmd


def test_warmstart_alone(tmp_path):
    proj = make_project(tmp_path)
    base = proj.checkpoints / "base.ckpt"
    cmd = build_command(proj, warmstart=base)
    assert flag_value(cmd, "--model.warmstart_ckpt") == str(base)
    assert "--ckpt_path" not in cmd


def test_default_root_dir_points_at_runs_tier(tmp_path):
    proj = make_project(tmp_path)
    cmd = build_command(proj, tier="high")
    assert flag_value(cmd, "--trainer.default_root_dir") == str(proj.runs("high"))


def test_data_paths_use_project_dirs(tmp_path):
    proj = make_project(tmp_path)
    cmd = build_command(proj, tier="medium")
    assert flag_value(cmd, "--data.csv_path") == str(proj.metadata)
    assert flag_value(cmd, "--data.audio_dir") == str(proj.wavs)
    assert flag_value(cmd, "--data.cache_dir") == str(proj.cache("medium"))


def test_build_command_uses_voice_stem(tmp_path):
    """voice_name is {name}-{tier} — no language prefix (Task 5)."""
    cmd = build_command(make_project(tmp_path), tier="medium",
                        espeak_voice="en-gb")
    assert flag_value(cmd, "--data.voice_name") == "marvin-medium"


# ------------------------------------------------------- epoch resolution

def test_resolve_max_epochs():
    assert resolve_max_epochs(9999, 100, None) == 10099
    assert resolve_max_epochs(10, None, 500) == 500
    assert resolve_max_epochs(None, None, None) == 4000  # default


def test_resolve_max_epochs_needs_readable_epoch():
    with pytest.raises(RuntimeError, match="readable epoch"):
        resolve_max_epochs(None, 100, None)


def test_check_resume_ceiling_refuses_and_names_numbers():
    with pytest.raises(RuntimeError, match="epoch 9999"):
        check_resume_ceiling(9999, 4000)
    with pytest.raises(RuntimeError, match="max_epochs is 10"):
        check_resume_ceiling(10, 10)  # equal is already a no-op


def test_check_resume_ceiling_passes():
    check_resume_ceiling(9, 10)
    check_resume_ceiling(None, 10)  # unreadable epoch: cannot check, proceed


def test_max_epochs_default_agrees_with_help(capsys):
    """One constant feeds both the resolver and the --max-epochs help text,
    so they cannot drift (Task 1c)."""
    from piper_trainer import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    from piper_trainer.train import DEFAULT_MAX_EPOCHS
    assert f"default {DEFAULT_MAX_EPOCHS}" in out
    assert resolve_max_epochs(None, None, None) == DEFAULT_MAX_EPOCHS


# ---------------------------------------------------------- latest_checkpoint

def touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ckpt")
    os.utime(path, (mtime, mtime))


def test_latest_checkpoint_none_when_absent(tmp_path):
    assert latest_checkpoint(make_project(tmp_path), "medium") is None


def test_latest_checkpoint_prefers_last_ckpt(tmp_path):
    proj = make_project(tmp_path)
    v0 = proj.runs("medium") / "lightning_logs" / "version_0" / "checkpoints"
    t0 = time.time()
    touch(v0 / "epoch=5.ckpt", t0 - 100)
    touch(v0 / "last.ckpt", t0 - 50)
    # even though another checkpoint is newer, last.ckpt wins
    assert latest_checkpoint(proj, "medium") == v0 / "last.ckpt"


def test_latest_checkpoint_picks_newest_across_versions(tmp_path):
    proj = make_project(tmp_path)
    base = proj.runs("medium") / "lightning_logs"
    t0 = time.time()
    touch(base / "version_0" / "checkpoints" / "last.ckpt", t0 - 1000)
    touch(base / "version_1" / "checkpoints" / "last.ckpt", t0)
    assert latest_checkpoint(proj, "medium") == \
        base / "version_1" / "checkpoints" / "last.ckpt"


def test_latest_checkpoint_falls_back_to_newest_ckpt(tmp_path):
    proj = make_project(tmp_path)
    base = proj.runs("medium") / "lightning_logs"
    t0 = time.time()
    touch(base / "version_0" / "checkpoints" / "epoch=1.ckpt", t0 - 50)
    touch(base / "version_0" / "checkpoints" / "epoch=2.ckpt", t0)
    assert latest_checkpoint(proj, "medium") == \
        base / "version_0" / "checkpoints" / "epoch=2.ckpt"
