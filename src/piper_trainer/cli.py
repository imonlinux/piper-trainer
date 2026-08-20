"""piper-trainer CLI.

Subcommands map 1:1 onto the planned UI screens, and every command is a thin
wrapper over a function in this package — the future FastAPI layer calls the
same functions rather than shelling out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import (clean as clean_mod, doctor, export as export_mod, metadata,
               prepare, train as train_mod, transcribe)
from .config import Project, TIERS
from .validate import validate_checkpoint, validate_dataset


def _resolve_voice(proj: Project, given: str | None) -> str:
    """Effective espeak voice: explicit > saved > en-us.

    Phonemization must stay consistent across runs — a resume that silently
    switches from en-gb-x-rp to en-us re-phonemizes the whole dataset against
    the wrong accent, and the result trains cleanly and sounds wrong.
    """
    saved = proj.get("espeak_voice")
    if given:
        if saved and saved != given:
            print(f"! espeak voice changed: project has {saved!r}, "
                  f"using {given!r}. This re-phonemizes the dataset; "
                  f"checkpoints from the previous voice may not transfer well.",
                  file=sys.stderr)
        proj.set(espeak_voice=given)
        return given
    if saved:
        return saved
    print("! no espeak voice set for this project; defaulting to 'en-us'. "
          "Set one with: piper-trainer init <project> --espeak-voice <voice>",
          file=sys.stderr)
    return "en-us"


def _project(args) -> Project:
    root = Path(args.project).resolve()
    name = args.name
    if not name:
        # saved name wins over the directory name: `init <dir> --name x`
        # records x, and every later command must keep using it
        name = Project(root=root, name=root.name).meta().get("name")
    return Project(root=root, name=name or root.name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="piper-trainer",
        description="Dataset prep and training pipeline for piper1-gpl")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_project(sp):
        sp.add_argument("project", help="project directory (mounted volume)")
        sp.add_argument("--name", help="voice name (default: directory name)")

    # doctor -----------------------------------------------------------------
    sub.add_parser("doctor", help="verify the environment")

    sp = sub.add_parser("voices", help="list espeak voices")
    sp.add_argument("--prefix", default="")

    # sources ----------------------------------------------------------------
    sp = sub.add_parser("sources", help="list source recordings in raw/")
    add_project(sp)

    # init -------------------------------------------------------------------
    sp = sub.add_parser("init", help="create the project layout")
    add_project(sp)
    sp.add_argument("--espeak-voice",
                    help="e.g. en-gb-x-rp; recorded in project.json and used "
                         "as the default for prepare/train/export")
    sp.add_argument("--tier", default="medium", choices=list(TIERS))

    # prepare ----------------------------------------------------------------
    sp = sub.add_parser("prepare", help="raw audio -> dataset/wavs")
    add_project(sp)
    sp.add_argument("--tier", default="medium", choices=list(TIERS))
    sp.add_argument("--channel", choices=["left", "right"],
                    help="pick one channel instead of downmixing")
    sp.add_argument("--no-denoise", action="store_true")
    sp.add_argument("--energy-threshold", type=float, default=55)
    sp.add_argument("--min-dur", type=float, default=1.5)
    sp.add_argument("--max-dur", type=float, default=10.0)
    sp.add_argument("--max-silence", type=float, default=0.4)
    sp.add_argument("--pad", type=float, default=0.15,
                    help="leading/trailing silence kept per clip")
    sp.add_argument("--force", action="store_true",
                    help="re-run stages even when inputs and parameters "
                         "are unchanged")

    # transcribe -------------------------------------------------------------
    sp = sub.add_parser("transcribe", help="Whisper -> metadata.csv + audit.csv")
    add_project(sp)
    sp.add_argument("--model", default=None)
    sp.add_argument("--language", default="en")
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--retranscribe", action="store_true",
                    help="transcribe every clip, ignoring existing "
                         "metadata.csv rows")
    sp.add_argument("--only-missing", action="store_true",
                    help="transcribe only clips missing from metadata.csv "
                         "(the default; stated explicitly for the API layer)")

    # validate ---------------------------------------------------------------
    sp = sub.add_parser("validate", help="pre-flight checks")
    add_project(sp)
    sp.add_argument("--tier", default="medium", choices=list(TIERS))
    sp.add_argument("--batch-size", type=int)
    sp.add_argument("--espeak-voice")
    sp.add_argument("--checkpoint", type=Path)

    # clean ------------------------------------------------------------------
    sp = sub.add_parser(
        "clean", help="act on validation findings (dry run unless --apply)")
    add_project(sp)
    sp.add_argument("--tier", default="medium", choices=list(TIERS))
    sp.add_argument("--espeak-voice")
    sp.add_argument("--apply", action="store_true",
                    help="actually modify files; default is a dry run")
    sp.add_argument("--only", help="comma-separated finding codes to act on")
    sp.add_argument("--exclude", help="comma-separated finding codes to skip")
    sp.add_argument("--force", action="store_true",
                    help="allow removing more than a third of the dataset")

    sp = sub.add_parser("restore", help="move quarantined clips back")
    add_project(sp)
    sp.add_argument("--ids", help="comma-separated clip ids; default: all")

    # train ------------------------------------------------------------------
    sp = sub.add_parser("train", help="run training")
    add_project(sp)
    sp.add_argument("--tier", default="medium", choices=list(TIERS))
    sp.add_argument("--espeak-voice",
                    help="default: the project's saved voice, else en-us")
    sp.add_argument("--batch-size", type=int, default=32)
    # default None so an explicit --max-epochs can be told apart from the
    # default when --add-epochs is in play (they are mutually exclusive)
    sp.add_argument("--max-epochs", type=int, default=None,
                    help="absolute epoch ceiling (default 4000); "
                         "mutually exclusive with --add-epochs")
    sp.add_argument("--add-epochs", type=int, default=None,
                    help="train N more epochs on top of the resume "
                         "checkpoint's epoch; requires --resume")
    sp.add_argument("--validation-split", type=float, default=None,
                    help="fraction of clips held out for validation "
                         "(default 0.02; use 0 to disable)")
    sp.add_argument("--num-workers", type=int, default=8)
    sp.add_argument("--warmstart", type=Path,
                    help="fine-tune from another voice (weights only)")
    # NOT type=Path: "auto" is a sentinel, and argparse would convert it to
    # PosixPath('auto') before we could compare it, sending the literal string
    # through to Lightning.
    sp.add_argument("--resume", nargs="?", const="auto", metavar="CKPT|auto",
                    help="resume your own run; 'auto' finds the latest checkpoint")
    sp.add_argument("--accelerator", default="gpu")
    sp.add_argument("--precision", default="32-true")
    sp.add_argument("--skip-validate", action="store_true")
    sp.add_argument("--dry-run", action="store_true")

    # export -----------------------------------------------------------------
    sp = sub.add_parser("export", help="checkpoint -> .onnx + complete .onnx.json")
    add_project(sp)
    sp.add_argument("--tier", default="medium", choices=list(TIERS))
    sp.add_argument("--checkpoint", type=Path, help="default: latest")
    sp.add_argument("--voice-name", help="output stem; also the 'dataset' field")
    sp.add_argument("--espeak-voice",
                    help="default: the project's saved voice, else en-us")
    sp.add_argument("--length-scale", type=float)
    sp.add_argument("--noise-scale", type=float)
    sp.add_argument("--noise-w", type=float)

    args = p.parse_args(argv)

    # ------------------------------------------------------------------ doctor
    if args.cmd == "doctor":
        lines, ok = doctor.check()
        print("\n".join(lines))
        return 0 if ok else 1

    if args.cmd == "voices":
        print("\n".join(doctor.espeak_voices(args.prefix)))
        return 0

    proj = _project(args)

    if args.cmd == "init":
        proj.ensure()
        meta = proj.set(espeak_voice=args.espeak_voice, tier=args.tier)
        print(f"initialized {proj.root} (voice name: {proj.name})")
        print(f"  espeak voice: {meta.get('espeak_voice') or '(unset — pass '
              f'--espeak-voice, or specify it on train)'}")
        print(f"  tier:         {meta.get('tier')}")
        print(f"drop source recordings in {proj.raw}")
        return 0

    if args.cmd == "sources":
        rows = prepare.sources(proj)
        if not rows:
            print(f"no source recordings in {proj.raw}")
            return 0
        fmt = "{:<30} {:<6} {:>7} {:>3} {:>9} {:>9}"
        print(fmt.format("name", "codec", "rate", "ch", "dur(s)", "size"))
        for r in rows:
            print(fmt.format(r["name"][:30], r["codec"] or "?",
                             r["sample_rate"] or "?", r["channels"] or "?",
                             r["duration"] or "?",
                             f"{r['size'] / 1e6:.1f}MB"))
        return 0

    if args.cmd == "prepare":
        stats = prepare.run_all(
            proj, tier=args.tier, channel=args.channel,
            denoise_enabled=not args.no_denoise, force=args.force,
            energy_threshold=args.energy_threshold,
            min_dur=args.min_dur, max_dur=args.max_dur,
            max_silence=args.max_silence,
            max_leading_silence=args.pad, max_trailing_silence=args.pad)
        for k, v in stats.items():
            print(f"{k}: {v}")
        if stats["clips"] == 0:
            print("no clips produced — try a lower --energy-threshold",
                  file=sys.stderr)
            return 1
        return 0

    if args.cmd == "transcribe":
        if args.retranscribe and args.only_missing:
            print("--retranscribe and --only-missing are mutually exclusive",
                  file=sys.stderr)
            return 2
        stats = transcribe.transcribe(proj, model_size=args.model,
                                      language=args.language,
                                      device=args.device,
                                      retranscribe=args.retranscribe)
        print(f"{stats['transcribed']} transcribed, {stats['skipped']} "
              f"skipped, {stats['total_seconds']/60:.1f} min -> "
              f"{proj.metadata}")
        print(f"review {proj.audit} before training")
        return 0

    if args.cmd == "validate":
        findings = validate_dataset(proj, tier=args.tier,
                                    batch_size=args.batch_size,
                                    espeak_voice=args.espeak_voice)
        if args.checkpoint:
            findings += validate_checkpoint(args.checkpoint, args.tier)
        for f in findings:
            print(f)
        return 1 if any(f.level == "error" for f in findings) else 0

    if args.cmd == "clean":
        findings = validate_dataset(proj, tier=args.tier,
                                    espeak_voice=args.espeak_voice)
        plan = clean_mod.build_plan(
            proj, findings,
            only=set(args.only.split(",")) if args.only else None,
            exclude=set(args.exclude.split(",")) if args.exclude else None)
        total = len(metadata.read(proj.metadata)[0]) \
            if proj.metadata.exists() else 0
        lines = clean_mod.describe(plan, total)
        print("\n".join(lines) if lines else "nothing to do")
        if not plan.touched and not plan.normalize_file:
            return 0
        if not args.apply:
            print("\n(dry run — re-run with --apply to make these changes)")
            return 0
        try:
            stats = clean_mod.apply(proj, plan, force=args.force)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        print("\n" + ", ".join(f"{k}: {v}" for k, v in stats.items()))
        if stats["quarantined"]:
            print(f"quarantined clips: {proj.dataset / 'quarantine'} "
                  f"(restore with: piper-trainer restore)")
        return 0

    if args.cmd == "restore":
        ids = args.ids.split(",") if args.ids else None
        n = clean_mod.restore(proj, ids)
        print(f"restored {n} clip(s) to {proj.wavs}")
        if n:
            print("re-run transcribe (or edit metadata.csv) to re-add their rows")
        return 0

    if args.cmd == "train":
        args.espeak_voice = _resolve_voice(proj, args.espeak_voice)
        validation_split = args.validation_split \
            if args.validation_split is not None else 0.02
        if args.add_epochs is not None and args.max_epochs is not None:
            print("--add-epochs and --max-epochs are mutually exclusive",
                  file=sys.stderr)
            return 2
        if args.add_epochs is not None and args.resume is None:
            print("--add-epochs requires --resume (nothing to add to "
                  "otherwise)", file=sys.stderr)
            return 2
        if not args.skip_validate:
            findings = validate_dataset(proj, tier=args.tier,
                                        batch_size=args.batch_size,
                                        espeak_voice=args.espeak_voice,
                                        validation_split=validation_split)
            for f in findings:
                print(f)
            if any(f.level == "error" for f in findings):
                print("\nrefusing to start; fix the errors or --skip-validate",
                      file=sys.stderr)
                return 1
        resume = args.resume
        ckpt_epoch = None
        if resume is not None:
            if str(resume) == "auto":
                resume = train_mod.latest_checkpoint(proj, args.tier)
                if resume is None:
                    print("no checkpoint to resume from", file=sys.stderr)
                    return 1
            else:
                resume = Path(resume)
                if not resume.exists():
                    print(f"checkpoint not found: {resume}", file=sys.stderr)
                    return 1
            ckpt_epoch = train_mod.checkpoint_epoch(resume)
            where = f" (epoch {ckpt_epoch})" if ckpt_epoch is not None else ""
            print(f"resuming from {resume}{where}")
        try:
            max_epochs = train_mod.resolve_max_epochs(
                ckpt_epoch, args.add_epochs, args.max_epochs)
            if resume is not None:
                train_mod.check_resume_ceiling(ckpt_epoch, max_epochs)
        except RuntimeError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        # target_epochs: fresh runs (re)set it for the tier; a resume never
        # overwrites it (design doc §1.4 uses it as the progress denominator)
        if not args.dry_run:
            targets = proj.get("target_epochs") or {}
            if resume is None or args.tier not in targets:
                targets[args.tier] = max_epochs
                proj.set(target_epochs=targets)
        cmd = train_mod.build_command(
            proj, tier=args.tier, espeak_voice=args.espeak_voice,
            batch_size=args.batch_size, max_epochs=max_epochs,
            num_workers=args.num_workers,
            validation_split=validation_split,
            warmstart=args.warmstart, resume=resume,
            accelerator=args.accelerator, precision=args.precision)
        if args.dry_run:
            print(" \\\n  ".join(cmd))
            return 0
        return train_mod.run(cmd)

    if args.cmd == "export":
        args.espeak_voice = _resolve_voice(proj, args.espeak_voice)
        ckpt = args.checkpoint or train_mod.latest_checkpoint(proj, args.tier)
        if not ckpt:
            print("no checkpoint found", file=sys.stderr)
            return 1
        print(f"exporting {ckpt}")
        onnx_path, json_path = export_mod.export(
            proj, args.tier, ckpt, voice_name=args.voice_name,
            espeak_voice=args.espeak_voice, length_scale=args.length_scale,
            noise_scale=args.noise_scale, noise_w=args.noise_w)
        problems = export_mod.verify(onnx_path, json_path)
        print(f"{onnx_path}\n{json_path}")
        for p_ in problems:
            print(f"✗ {p_}", file=sys.stderr)
        return 1 if problems else 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
