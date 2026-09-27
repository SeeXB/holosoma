#!/usr/bin/env python3
"""Experimental full-video / bounded-GT batch; never install plans into training."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

from holosoma_retargeting.semantic_keyframes import bounded_signals as bounded
from holosoma_retargeting.semantic_keyframes import pipeline
from holosoma_retargeting.semantic_keyframes import range_delta
from holosoma_retargeting.semantic_keyframes import adjacent_delta
from holosoma_retargeting.semantic_keyframes import adjacent_delta_boundaries

PLANNERS = {"bounded": bounded, "range-delta": range_delta, "adjacent-delta": adjacent_delta,
            "adjacent-delta-boundaries": adjacent_delta_boundaries}
GENERATORS = {"bounded": bounded.generate_bounded_plan, "range-delta": range_delta.generate_range_delta_plan,
              "adjacent-delta": adjacent_delta.generate_adjacent_delta_plan,
              "adjacent-delta-boundaries": adjacent_delta_boundaries.generate_boundary_plan}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_error(exc):
    text = str(exc).replace(os.environ.get("OPENAI_API_KEY", "NO_SECRET"), "[REDACTED]")
    return type(exc).__name__+": "+re.sub(r"data:(video|image)/[^\s\"']+", "[media omitted]", text)[:4000]


def run_one(row, output_root, model, host, max_repairs, method="bounded"):
    root, task = Path(output_root), row["task"]
    audit = root / "audit" / task
    output = root / "plans/omomo" / task / "semantic_plan.json"
    for field in ("video", "signal_bundle"):
        if sha(row[field]) != row[field+"_sha256"]:
            raise ValueError(f"input changed: {task}/{field}")
    requests = []

    def call(content, prompt):
        if (root / "quota_stop.json").exists():
            raise RuntimeError("batch provider quota stop")
        if os.environ["OPENAI_MODEL"] != model or urlsplit(os.environ["OPENAI_BASE_URL"]).hostname != host:
            raise ValueError("request model/endpoint changed")
        index = len(requests)
        if index > max_repairs:
            raise RuntimeError("request budget exceeded")
        request_path = audit / f"request_{index}.json"
        if request_path.exists():
            raise FileExistsError("request slot already used")
        record = dict(attempt=index, model=model, endpoint_host=host, started_at=now(), status="sending",
                      prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      input_type="full_mp4_video", client_frame_sampling=False)
        requests.append(record)
        write(request_path, record)
        try:
            raw = pipeline.call_vlm(content, prompt)
        except Exception as exc:
            record.update(status="failed", finished_at=now(), error=safe_error(exc))
            write(request_path, record)
            if isinstance(exc, pipeline.VLMQuotaError):
                write(root / "quota_stop.json", dict(task=task, reason="provider insufficient_quota"))
            raise
        record.update(status="response_received", finished_at=now())
        write(request_path, record)
        return raw

    try:
        generate = GENERATORS[method]
        result = generate(video=Path(row["video"]), bundle_file=Path(row["signal_bundle"]),
            output=output, audit_dir=audit, max_repairs=max_repairs, call=call)
        boundaries = [e["boundary_evidence"][ep] for e in result["events"] for ep in ("start", "end")]
        summary = dict(task=task, status="resolved", output=str(output), event_count=len(result["events"]),
                       gt_resolved_boundaries=sum(b["status"]=="gt_resolved" for b in boundaries),
                       clip_censored_boundaries=sum(b["status"]=="clip_censored" for b in boundaries),
                       multiple_candidate_boundaries=sum(b.get("multiple_candidates", False) for b in boundaries))
    except Exception as exc:
        summary = dict(task=task, status="unresolved", error=safe_error(exc))
    summary.update(request_count=len(requests), saved_responses=len(list(audit.glob("attempt_*.response.txt"))),
                   model=model, finished_at=now())
    write(root / "results" / f"{task}.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Existing preparation JSON with rows/video/signal_bundle SHA256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("src/holosoma_retargeting/.env"))
    parser.add_argument("--model", default="qwen3.5-27b", help="Process-only override; .env is never modified")
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--method", choices=tuple(PLANNERS), default="bounded")
    args = parser.parse_args()
    planner = PLANNERS[args.method]
    if not 0 <= args.max_repairs <= 3 or not 1 <= args.max_workers <= 4:
        parser.error("max-repairs must be 0..3 and max-workers 1..4")
    for key in ("OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        os.environ.pop(key, None)
    pipeline.load_env_file(args.env_file)
    os.environ["OPENAI_MODEL"] = args.model
    host = urlsplit(os.environ.get("OPENAI_BASE_URL", "")).hostname
    if host != args.expected_host or not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("missing credentials or endpoint differs from expected host")
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    tasks = args.tasks or [r["task"] for r in rows]
    if len(tasks) != len(set(tasks)) or set(tasks)-{r["task"] for r in rows}:
        raise ValueError("duplicate or unknown task")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preparation = root / "preparation.json"
        if preparation.exists():
            prep = json.loads(preparation.read_text())
            if prep["model"] != args.model or prep["endpoint_host"] != host or prep["manifest_sha256"] != sha(args.manifest):
                raise ValueError("cannot resume with different model/endpoint/inputs")
            if prep["max_repairs"] != args.max_repairs:
                raise ValueError("cannot change request budget")
            if prep.get("method", "bounded") != args.method:
                raise ValueError("cannot resume a different planning method")
        else:
            video_info = []
            protected = dict(manifest.get("protected_files_sha256", {}))
            for row in rows:
                for field in ("video", "signal_bundle"):
                    if sha(row[field]) != row[field+"_sha256"]:
                        raise ValueError(f"input SHA mismatch: {row['task']}/{field}")
                    protected[row[field]] = sha(row[field])
                ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).parents[2] / "hssim/bin/ffmpeg")
                decoded = subprocess.run([ffmpeg, "-hide_banner", "-i", row["video"], "-map", "0:v:0",
                    "-an", "-progress", "pipe:1", "-f", "null", "-"], capture_output=True, text=True, check=True)
                frames = int(re.findall(r"^frame=(\d+)", decoded.stdout, re.M)[-1])
                fps = float(re.search(r", ([\d.]+) fps,", decoded.stderr).group(1))
                if frames != row["source_frames"] or abs(fps-row["fps"]) > .01:
                    raise ValueError(f"video/GT not one-to-one aligned: {row['task']}: {frames}, {fps}")
                video_info.append(dict(task=row["task"], frames=frames, fps=fps,
                                       duration_s=frames/fps, last_gt_sample_s=(frames-1)/fps))
            for path in args.manifest.parent.rglob("*"):
                if path.is_file() and path.suffix in {".json", ".txt", ".md"}:
                    protected[str(path.resolve())] = sha(path)
            # Snapshot actual current files, don't assume old stored hashes still apply.
            protected = {path:sha(path) for path in protected}
            code = [Path(__file__).resolve(), Path(planner.__file__), Path(pipeline.__file__)]
            code += [Path(module.__file__) for module in getattr(planner, "SOURCE_DEPENDENCIES", ())]
            prep = dict(schema=planner.SCHEMA, method=args.method, created_at=now(), model=args.model, endpoint_host=host,
                        manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
                        tasks=[r["task"] for r in rows], rows=rows, video_alignment=video_info,
                        max_repairs=args.max_repairs,
                        config=bounded.asdict(bounded.DetectorConfig()) if args.method=="bounded" else planner.POLICY,
                        protected_files_sha256=protected, code_sha256={str(p):sha(p) for p in code})
            for path in code:
                target = root / "source_snapshot" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            write(preparation, prep)
        changed_code = [p for p,h in prep["code_sha256"].items() if sha(p) != h]
        if changed_code:
            raise ValueError(f"code changed after experiment preparation: {changed_code}")
        if args.prepare_only:
            print(json.dumps(dict(prepared_tasks=len(rows), video_alignment_verified=True, api_calls=0)), flush=True)
            return
        for task in tasks:
            if (root / "results" / f"{task}.json").exists() or list((root / "audit" / task).glob("request_*.json")):
                raise ValueError(f"task already attempted; inspect audit before any resume: {task}")
        write(root / "state.json", dict(phase="generating", tasks=tasks, pid=os.getpid(), started_at=now()))
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(run_one, row, str(root), args.model, host, args.max_repairs, args.method)
                       for row in rows if row["task"] in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps(result, ensure_ascii=False), flush=True)
        results = [json.loads(p.read_text()) for p in sorted((root/"results").glob("*.json"))]
        write(root / "generation_summary.json", dict(results=results, model=args.model))
        changed = [path for path,h in prep["protected_files_sha256"].items() if sha(path)!=h]
        write(root / "preservation_check.json", dict(checked=len(prep["protected_files_sha256"]), changed=changed))
        write(root / "state.json", dict(phase="finished_pending_review", completed_tasks=len(results),
              resolved=sum(r["status"]=="resolved" for r in results), requests=sum(r["request_count"] for r in results),
              changed_protected_files=changed, finished_at=now()))
        if changed:
            raise RuntimeError(f"protected files changed: {changed}")


if __name__ == "__main__":
    main()
