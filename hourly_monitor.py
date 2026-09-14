"""Minute-level queue checks with restricted Luna Max operational repair."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import subprocess
import time
from pathlib import Path


WORKSPACE = Path("/workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting")
ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "runs/full_benchmark_20260913/manifest.json"
DEFAULT_RECORD = ROOT / "runs/full_benchmark_20260913/hourly_monitor.jsonl"
DEFAULT_LATEST_REPORT = ROOT / "runs/full_benchmark_20260913/luna_latest.txt"
DEFAULT_ACTION_REQUIRED = ROOT / "runs/full_benchmark_20260913/action_required.md"
DEFAULT_INTERVAL = 60
DEFAULT_LUNA_INTERVAL = 1800
DEFAULT_LOG_TAIL_LINES = 40
PROFILE_LEGACY = "legacy"
PROFILE_PORT_VALIDATION = "port_validation"
PROFILE_CHOICES = (PROFILE_LEGACY, PROFILE_PORT_VALIDATION)
JSON_DECODER = json.JSONDecoder()
LOG_SIGNALS = (
    ("traceback", re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE)),
    ("nan", re.compile(r"\bNaN\b", re.IGNORECASE)),
    ("inf", re.compile(r"\b(?:Inf|Infinity)\b", re.IGNORECASE)),
    ("oom", re.compile(r"\b(?:OOM|out of memory|cuda out of memory)\b", re.IGNORECASE)),
    ("runtime_error", re.compile(r"\bRuntimeError\b|\bCUDA error\b", re.IGNORECASE)),
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def pid_exists(pid):
    return None if pid is None else Path(f"/proc/{pid}").exists()


def tail_lines(path, count):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        return stream.read().decode(errors="replace").splitlines()[-count:]


def last_progress(path):
    path = Path(path)
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        if line.strip():
            row, _ = JSON_DECODER.raw_decode(line.lstrip())
            return row
    return None


def gpu_status():
    devices = {}
    for gpu in (0, 1):
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        if len(fields) != 3:
            raise ValueError(f"unexpected nvidia-smi output for GPU {gpu}: {result.stdout!r}")
        devices[str(gpu)] = {
            "index": int(fields[0]),
            "utilization_gpu_percent": int(fields[1]),
            "memory_used_MiB": int(fields[2]),
        }
    return devices


def signal_names(lines):
    names = set()
    for line in lines:
        for name, pattern in LOG_SIGNALS:
            if pattern.search(line):
                names.add(name)
    return sorted(names)


def current_phase_logs(job, current_job):
    if current_job["state"] not in {"running", "failed"}:
        return []
    return [
        phase["logpath"]
        for phase in job["steps"]
        if "logpath" in phase and phase["name"] == current_job["phase"]
    ]


def compact_snapshot(manifest_path, tail_count=DEFAULT_LOG_TAIL_LINES):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    status_path = Path(manifest["status_path"])
    current = read_json(status_path)
    scheduler_pid = current["scheduler_pid"]
    scheduler_pid_exists = pid_exists(scheduler_pid)
    observed_state = current["state"]
    if observed_state == "running" and scheduler_pid is not None and not scheduler_pid_exists:
        observed_state = "interrupted"

    jobs = []
    log_candidates = [{"path": status_path.parent / "queue.log", "job": None, "phase": "scheduler"}]
    active_outputs = []
    active_results = []
    active_progress = []
    active_logs = []
    progress_nonfinite = []
    for job in manifest["jobs"]:
        current_job = current["jobs"][job["id"]]
        progress_paths = [
            phase["progress_path"] for phase in job["steps"] if "progress_path" in phase
        ]
        progress_path = progress_paths[-1] if progress_paths else None
        progress = last_progress(progress_path) if progress_path is not None else None
        progress_view = (
            {
                key: progress[key]
                for key in ("step", "loss", "l1", "points", "seconds")
                if key in progress
            }
            if progress is not None
            else None
        )
        nonfinite = [
            key
            for key in ("loss", "l1", "points", "seconds")
            if key in progress_view
            and isinstance(progress_view[key], (int, float))
            and not math.isfinite(progress_view[key])
        ] if progress_view is not None else []
        if nonfinite:
            progress_nonfinite.append({"job": job["id"], "fields": nonfinite})

        active_phase_logs = current_phase_logs(job, current_job)
        log_candidates.extend(
            {"path": path, "job": job["id"], "phase": current_job["phase"]}
            for path in active_phase_logs
        )
        job_state = current_job["state"]
        paths = None
        if job_state in {"running", "failed"}:
            paths = {
                "output": job["output"],
                "result": job["resultpath"],
                "logs": active_phase_logs,
                "progress": progress_path,
            }
            active_outputs.append(job["output"])
            active_results.append(job["resultpath"])
            active_progress.extend(path for path in [progress_path] if path is not None)
            active_logs.extend(active_phase_logs)
        job_view = {
            "id": job["id"],
            "method": job["method"],
            "family": job["family"],
            "scene": job["scene"],
            "state": job_state,
            "gpu": current_job["gpu"],
            "phase": current_job["phase"],
            "pid": current_job["pid"],
            "pid_exists": pid_exists(current_job["pid"]),
            "progress": progress_view,
            "error": current_job.get("error"),
        }
        if paths is not None:
            job_view["paths"] = paths
        jobs.append(job_view)

    logs = []
    for candidate in log_candidates:
        tail = tail_lines(candidate["path"], tail_count)
        logs.append(
            {
                "path": str(candidate["path"]),
                "job": candidate["job"],
                "phase": candidate["phase"],
                "exists": Path(candidate["path"]).exists(),
                "signals": signal_names(tail),
                "tail": tail,
            }
        )
    queue_log = logs[0]
    failure_logs = [
        entry for entry in logs if entry["job"] is not None and entry["signals"]
    ]

    workers = {}
    for gpu, worker in current["workers"].items():
        worker_view = dict(worker)
        worker_view["pid_exists"] = pid_exists(worker["pid"])
        workers[gpu] = worker_view
    status = {
        "state": observed_state,
        "reported_state": current["state"],
        "updated_utc": current["updated_utc"],
        "scheduler_pid": scheduler_pid,
        "scheduler_pid_exists": scheduler_pid_exists,
        "failure": current.get("failure"),
        "workers": workers,
    }

    anomalies = []
    if current["state"] == "failed":
        anomalies.append(
            {
                "kind": "scheduler_failed",
                "subject": "scheduler",
                "detail": current.get("failure", "status.state=failed"),
            }
        )
    if current["state"] == "running" and scheduler_pid is not None and not scheduler_pid_exists:
        anomalies.append(
            {
                "kind": "scheduler_process_missing",
                "subject": str(scheduler_pid),
                "detail": "status.state=running but scheduler PID is absent",
            }
        )
    for gpu, worker in workers.items():
        if worker["state"] == "running" and worker["pid"] is not None and not worker["pid_exists"]:
            anomalies.append(
                {
                    "kind": "worker_process_missing",
                    "subject": gpu,
                    "detail": f"worker PID {worker['pid']} is absent",
                }
            )
    for job in jobs:
        if job["state"] == "failed" or job["error"]:
            anomalies.append(
                {
                    "kind": "job_failed",
                    "subject": job["id"],
                    "detail": job["error"] or "job state is failed",
                }
            )
        if job["state"] == "running" and job["pid"] is not None and not job["pid_exists"]:
            anomalies.append(
                {
                    "kind": "child_process_missing",
                    "subject": job["id"],
                    "detail": f"child PID {job['pid']} is absent",
                }
            )
    for item in progress_nonfinite:
        anomalies.append(
            {
                "kind": "nonfinite_progress",
                "subject": item["job"],
                "detail": ", ".join(item["fields"]),
            }
        )
    for entry in failure_logs:
        anomalies.append(
            {
                "kind": "log_signal",
                "subject": entry["path"],
                "detail": ", ".join(entry["signals"]),
            }
        )

    return {
        "sampled_utc": utc_now(),
        "manifest": str(manifest_path),
        "status": status,
        "gpu": gpu_status(),
        "jobs": jobs,
        "anomalies": anomalies,
        "queue_log": queue_log,
        "failure_logs": failure_logs,
        "logs": logs,
        "paths": {
            "workspace": str(WORKSPACE),
            "root": manifest["root"],
            "manifest": str(manifest_path),
            "status": str(status_path),
            "queue_log": str(status_path.parent / "queue.log"),
            "active_logs": sorted(set(active_logs)),
            "active_outputs": sorted(set(active_outputs)),
            "active_results": sorted(set(active_results)),
            "active_progress": sorted(set(active_progress)),
        },
    }


def anomaly_state(snapshot):
    return tuple(
        (item["kind"], item["subject"], item["detail"])
        for item in snapshot["anomalies"]
    )


def luna_view(snapshot):
    counts = {}
    for job in snapshot["jobs"]:
        method_counts = counts.setdefault(
            job["method"],
            {"completed": 0, "pending": 0, "running": 0, "failed": 0},
        )
        method_counts[job["state"]] += 1
    active_jobs = [
        job
        for job in snapshot["jobs"]
        if job["state"] in {"running", "failed"}
    ]
    view = {
        "sampled_utc": snapshot["sampled_utc"],
        "manifest": snapshot["manifest"],
        "status": snapshot["status"],
        "gpu": snapshot["gpu"],
        "method_state_counts": counts,
        "running_or_failed_jobs": active_jobs,
        "anomalies": snapshot["anomalies"],
        "queue_log": snapshot["queue_log"],
        "active_phase_logs": [
            entry for entry in snapshot["logs"] if entry["job"] is not None
        ],
        "failure_logs": snapshot["failure_logs"],
        "paths": snapshot["paths"],
    }

    if not snapshot["anomalies"]:
        for key in ("queue_log", "active_phase_logs", "failure_logs", "paths"):
            view.pop(key)
        view["running_or_failed_jobs"] = [
            {key: job[key] for key in ("id", "state", "gpu", "phase", "pid", "pid_exists", "progress")}
            for job in active_jobs
        ]
    return view


def port_validation_view(snapshot):
    """Add compact PORT-only contract and result paths to a snapshot."""
    view = luna_view(snapshot)
    manifest = read_json(snapshot["manifest"])
    current_jobs = {job["id"]: job for job in snapshot["jobs"]}
    result_paths = []
    for manifest_job in manifest["jobs"]:
        current_job = current_jobs.get(manifest_job["id"], {})
        output = manifest_job.get("output")
        resultpath = manifest_job.get("resultpath")
        result_paths.append(
            {
                "id": manifest_job["id"],
                "family": manifest_job.get("family"),
                "scene": manifest_job.get("scene"),
                "state": current_job.get("state", manifest_job.get("state")),
                "resultpath": resultpath,
                "result_exists": resultpath is not None and Path(resultpath).exists(),
                "config": str(Path(output) / "config.json") if output else None,
                "split": str(Path(output) / "split.json") if output else None,
            }
        )
    protocol = manifest["protocol"]
    selection = protocol["selection"]
    view["profile"] = {
        "name": PROFILE_PORT_VALIDATION,
        "read_only": True,
        "contract": {
            "method": manifest["method"],
            "jobs": len(manifest["jobs"]),
            "families": len(selection),
            "scenes_by_family": {
                family: len(scenes) for family, scenes in selection.items()
            },
        },
        "manifest_protocol": protocol,
        "selection": selection,
        "completion_result_paths": result_paths,
    }
    return view


def port_validation_luna_prompt(snapshot, previous, reason):
    previous_text = "none" if previous is None else json.dumps(
        port_validation_view(previous), indent=2, ensure_ascii=False
    )
    current_text = json.dumps(
        port_validation_view(snapshot), indent=2, ensure_ascii=False
    )
    completion_instruction = (
        "status=completed 时必须只读检查 manifest 声明的每个 job 的 resultpath，以及每个输出目录中存在的 "
        "config.json 和 split.json；逐 job 报告结果文件是否存在、指标是否 finite、配置是否符合 manifest_protocol "
        "中声明的空间分区与 HashGrid 字段，并引用绝对证据路径。"
        if snapshot["status"]["state"] == "completed"
        else
        "队列未完成时只检查当前运行阶段和已声明日志尾部，不把尚未生成的结果当作失败。"
    )
    return (
        "你是 PORT-GS validation 的只读巡检与完成检查执行者。触发原因是 "
        + reason
        + "。这是显式 opt-in 的 PORT-only profile；当前 manifest 的 job selection、训练/测试协议、"
        "空间分区与 HashGrid 参数字段，以及精确 argv、配置字段和实际状态是唯一依据，不要猜测或改写参数。"
        "只处理当前快照 manifest 中的 PORT-GS jobs；不要引用旧 full benchmark、SSS-GS NaN、恢复 socket，"
        "也不要使用 port-full-benchmark tmux socket。"
        "这是只读诊断：允许读取 manifest、status、config、split、result、日志尾部和 GPU 状态，"
        "但严禁修改任何源码、manifest、配置、输出、结果或文档，严禁设置 retry_requested，严禁重跑、"
        "重启、kill、tmux 操作、算法/损失/预算/数据/指标调整或自动修复。进程缺失只能作为待核实证据，"
        "不要因为模型沙箱看不到宿主进程就声称任务退出。异常为空时简短报告健康状态；异常存在时只报告"
        "证据、可能原因和交给主线程的决定。"
        + completion_instruction
        + "请用中文简短报告，列出状态、GPU/进度、六个 scene 的完成度、指标证据路径和仍需主线程处理的事项；"
        "不要声称能够唤醒主线程。\n\n上一份快照：\n"
        + previous_text
        + "\n\n当前真实快照（宿主脚本采集，含声明路径和日志尾部）：\n"
        + current_text
    )


def luna_prompt(snapshot, previous, reason, profile=PROFILE_LEGACY):
    if profile == PROFILE_PORT_VALIDATION:
        return port_validation_luna_prompt(snapshot, previous, reason)
    previous_text = "none" if previous is None else json.dumps(
        luna_view(previous), indent=2, ensure_ascii=False
    )
    current_text = json.dumps(luna_view(snapshot), indent=2, ensure_ascii=False)
    return (
        "你是 PORT-GS 全量队列的运行巡检与有限修复执行者。触发原因是 "
        + reason
        + "。快照由宿主脚本采集，是本次进程和GPU状态的依据。anomalies为空时只依据快照输出简短巡检报告，"
        "不调用工具、不扫描源码、不修改文件、不操作进程；引用SSS历史NaN仍待诊断即可。"
        "只有anomalies非空时才进入以下诊断修复流程。沙箱中的ps和/proc可能隐藏宿主进程，不能据此认定进程已退出。"
        "进程、GPU、tmux的复核必须请求sandbox_permissions=require_escalated的宿主只读检查；"
        "无法取得宿主视图时记录需要主线程处理，不能执行重启。严禁使用kill、tmux -k或kill-session终止健康任务。"
        "你可以在当前 workspace-write 沙箱中检查当前 manifest 声明的路径和失败日志尾部，"
        "只允许修复有明确证据的环境、启动或指标实现错误。严格保持 GPU0/GPU1、seed 0、"
        "训练预算、数据划分、原始测试标定、指标定义和已完成结果。只处理当前 manifest 列出的 PORT-GS、"
        "官方 GS3、官方 SSS-GS、SSD-GS jobs；其他本地项目不属于本队列。SSD-GS 只评价原始测试标定，"
        "不重训、不拟合测试集。不要改训练方法、数值算法、损失、结构、预算、数据或指标口径；不要盲目重跑、"
        "不要使用 nan_to_num 或防御性补丁，不要杀健康任务，不要重启整个队列。数学 NaN、发散、质量退化和"
        "方法问题只做诊断并请求主线程，不能设置 retry。SSS-GS 历史 NaN 待决证据见 "
        "SSS-GS/docs/experiments/nan_20260913.md；当前新 run 暂时 finite 不能宣称已修复。"
        "若且仅若 failed job 已证实是环境/启动/指标实现错误：先保留并归档失败证据；train 失败才归档整训练输出，"
        "render/eval 失败必须保留训练 checkpoint，只归档该失败 phase 的输出和日志。完成最小修复后，才在对应"
        "manifest job 写入 retry_requested:true，交给 scheduler 下一个 job boundary；未修复 job 不得设置。"
        "若协调器退出且仍有 pending jobs，先确认没有任何活动训练子进程，再复用 pane 保存的 python --resume 命令执行"
        "tmux -L port-full-benchmark respawn-pane -t full:queue 恢复；只恢复已修复且 retry_requested 的 job，保留健康任务。"
        "请用中文简短报告证据、实际修复、retry_requested 状态和仍需主线程决定的事项。不要声称能通过"
        "collaboration 工具唤醒主线程；主线程读取 action_required 文件。\n\n上一份快照：\n"
        + previous_text
        + "\n\n当前真实快照（含主要路径和失败日志尾部）：\n"
        + current_text
    )


def luna_command(codex, output_path, profile=PROFILE_LEGACY):
    command = [
        codex,
        "-C",
        str(WORKSPACE),
        "exec",
        "-m",
        "gpt-5.6-luna",
        "-c",
        'model_reasoning_effort="max"',
        "--approve-for-me",
        "--ephemeral",
        "--skip-git-repo-check",
        "-o",
        str(output_path),
    ]
    if profile == PROFILE_PORT_VALIDATION:
        command[command.index("--approve-for-me"): command.index("--approve-for-me") + 1] = [
            "--sandbox",
            "read-only",
        ]
    return command


def ask_luna(prompt, output_path, codex, profile=PROFILE_LEGACY):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        luna_command(codex, output_path, profile) + ["-"],
        check=True,
        input=prompt,
        text=True,
        cwd=WORKSPACE,
    )
    return output_path.read_text()


def last_record_snapshot(path):
    path = Path(path)
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        if line.strip():
            return json.loads(line)["snapshot"]
    return None


def append_record(
    path, snapshot, analysis, model_call, reason, profile=PROFILE_LEGACY
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "sampled_utc": snapshot["sampled_utc"],
                    "snapshot": snapshot,
                    "luna": analysis,
                    "model_call": model_call,
                    "trigger": reason,
                    "profile": profile,
                },
                ensure_ascii=False,
                allow_nan=True,
            )
            + "\n"
        )


def write_action_required(
    path, snapshot, latest_report, model_call, reason, profile=PROFILE_LEGACY
):
    lines = [
        "# PORT-GS monitor action_required",
        "",
        f"- profile: {profile}",
        f"- sampled_utc: {snapshot['sampled_utc']}",
        f"- queue_state: {snapshot['status']['state']}",
        f"- trigger: {reason}",
        f"- latest_report: {latest_report}",
        "",
    ]
    if snapshot["anomalies"]:
        lines.extend(["## 需要主线程审核或处理", ""])
        lines.extend(
            f"- {item['kind']} {item['subject']}: {item['detail']}"
            for item in snapshot["anomalies"]
        )
        lines.extend(
            [
                "",
                "数学 NaN、发散、方法或指标口径决策必须由主线程判断；监控器不会自动改算法、预算或指标。",
            ]
        )
    else:
        lines.extend(["## 当前没有本地异常", "", "继续按固定协议巡检。"])
    if model_call is not None:
        lines.extend(["", "## 最近 Luna 调用", "", json.dumps(model_call, ensure_ascii=False)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def monitor(
    manifest_path,
    record_path,
    latest_report_path,
    action_required_path,
    codex,
    interval=DEFAULT_INTERVAL,
    luna_interval=DEFAULT_LUNA_INTERVAL,
    tail_count=DEFAULT_LOG_TAIL_LINES,
    profile=PROFILE_LEGACY,
):
    previous = last_record_snapshot(record_path)
    last_luna_monotonic = None
    previous_anomaly_state = None
    while True:
        snapshot = compact_snapshot(manifest_path, tail_count)
        current_anomaly_state = anomaly_state(snapshot)
        anomaly_changed = bool(snapshot["anomalies"]) and current_anomaly_state != previous_anomaly_state
        completion_due = (
            snapshot["status"]["state"] == "completed"
            and (previous is None or previous["status"]["state"] != "completed")
        )
        now = time.monotonic()
        interval_due = last_luna_monotonic is None or now - last_luna_monotonic >= luna_interval
        analysis = None
        model_call = None
        reason = None
        if interval_due or anomaly_changed or completion_due:
            reason = (
                "completion"
                if completion_due
                else "initial"
                if last_luna_monotonic is None
                else "anomaly"
                if anomaly_changed
                else "interval"
            )
            last_luna_monotonic = now
            started = utc_now()
            try:
                analysis = ask_luna(
                    luna_prompt(snapshot, previous, reason, profile),
                    latest_report_path,
                    codex,
                    profile,
                )
            except subprocess.CalledProcessError as error:
                model_call = {
                    "reason": reason,
                    "status": "failed",
                    "returncode": error.returncode,
                    "output_path": str(latest_report_path),
                    "profile": profile,
                    "started_utc": started,
                    "finished_utc": utc_now(),
                }
                append_record(record_path, snapshot, None, model_call, reason, profile)
                write_action_required(
                    action_required_path,
                    snapshot,
                    latest_report_path,
                    model_call,
                    reason,
                    profile,
                )
                raise
            model_call = {
                "reason": reason,
                "status": "completed",
                "output_path": str(latest_report_path),
                "profile": profile,
                "started_utc": started,
                "finished_utc": utc_now(),
                "result": analysis,
            }
            write_action_required(
                action_required_path,
                snapshot,
                latest_report_path,
                model_call,
                reason,
                profile,
            )
            previous_anomaly_state = current_anomaly_state
        elif snapshot["anomalies"]:
            previous_anomaly_state = current_anomaly_state
        else:
            previous_anomaly_state = None

        append_record(record_path, snapshot, analysis, model_call, reason, profile)
        previous = snapshot
        print(
            json.dumps(
                {
                    "event": "monitor_poll",
                    "sampled_utc": snapshot["sampled_utc"],
                    "state": snapshot["status"]["state"],
                    "anomalies": len(snapshot["anomalies"]),
                    "luna_called": model_call is not None,
                    "trigger": reason,
                    "record": str(record_path),
                    "latest_report": str(latest_report_path),
                    "action_required": str(action_required_path),
                    "profile": profile,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if snapshot["status"]["state"] == "completed":
            return
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--latest-report", type=Path, default=DEFAULT_LATEST_REPORT)
    parser.add_argument("--action-required", type=Path, default=DEFAULT_ACTION_REQUIRED)
    parser.add_argument("--codex", default="codex")
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=PROFILE_LEGACY,
        help=(
            "monitor prompt/permissions profile; default preserves the historical "
            "full-benchmark behavior"
        ),
    )
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--luna-interval", type=int, default=DEFAULT_LUNA_INTERVAL)
    parser.add_argument("--log-tail-lines", type=int, default=DEFAULT_LOG_TAIL_LINES)
    args = parser.parse_args()
    monitor(
        args.manifest.resolve(),
        args.record.resolve(),
        args.latest_report.resolve(),
        args.action_required.resolve(),
        args.codex,
        args.interval,
        args.luna_interval,
        args.log_tail_lines,
        args.profile,
    )


if __name__ == "__main__":
    main()
