"""Run explicit experiment phases with one serial worker per GPU."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import subprocess
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "runs/full_benchmark_20260913/manifest.json"
COMPLETED_STATES = {"completed", "reused"}
JSON_DECODER = json.JSONDecoder()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def last_json_line(path):
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    return None if not lines else JSON_DECODER.raw_decode(lines[-1].lstrip())[0]


def job_status_record(job):
    state = job["state"]
    if state not in {"pending", "reused"}:
        raise ValueError(f"unsupported manifest job state: {state}")
    reused = state == "reused"
    return {
        "method": job["method"],
        "family": job["family"],
        "scene": job["scene"],
        "state": "completed" if reused else "pending",
        "gpu": None,
        "phase": None,
        "pid": None,
        "output": job["output"],
        "resultpath": job["resultpath"],
        "source_archive": job.get("source_archive"),
        "reused_from": job.get("reused_from"),
        "steps": {
            step["name"]: {
                "state": "reused" if reused else "pending",
                "cwd": step["cwd"],
                "argv": step["argv"],
                "logpath": step["logpath"],
                "output": step["output"],
                "resultpath": step["resultpath"],
            }
            for step in job["steps"]
        },
    }


def initial_status(manifest, manifest_path):
    jobs = {job["id"]: job_status_record(job) for job in manifest["jobs"]}
    return {
        "manifest": str(manifest_path),
        "state": "pending",
        "started_utc": None,
        "finished_utc": None,
        "scheduler_pid": None,
        "updated_utc": utc_now(),
        "workers": {
            str(gpu): {"state": "idle", "job": None, "pid": None}
            for gpu in manifest["worker_gpus"]
        },
        "jobs": jobs,
    }


def resume_status(manifest, manifest_path):
    status_path = Path(manifest["status_path"])
    status = read_json(status_path)
    jobs = status["jobs"]
    for job in manifest["jobs"]:
        if job["id"] not in jobs:
            jobs[job["id"]] = job_status_record(job)
        job_status = jobs[job["id"]]
        pending = False
        failed = job_status["state"] == "failed"
        for phase in job["steps"]:
            phase_status = job_status["steps"][phase["name"]]
            if phase_status["state"] in COMPLETED_STATES:
                continue
            if phase_status["state"] == "failed":
                failed = True
                continue
            phase_status.update(state="pending", pid=None)
            phase_status.pop("started_utc", None)
            phase_status.pop("finished_utc", None)
            phase_status.pop("error", None)
            pending = True
        if failed:
            job_status.update(state="failed", gpu=None, pid=None)
        elif pending:
            job_status.update(state="pending", gpu=None, phase=None, pid=None)
            job_status.pop("finished_utc", None)
            job_status.pop("error", None)
        else:
            job_status.update(state="completed", phase=None, pid=None)
    status["workers"] = {
        str(gpu): {"state": "idle", "job": None, "pid": None}
        for gpu in manifest["worker_gpus"]
    }
    status["state"] = "pending"
    status["started_utc"] = None
    status["finished_utc"] = None
    status.pop("failure", None)
    return status


def scheduler_pid_exists(pid):
    return Path(f"/proc/{pid}").exists()


class Scheduler:
    def __init__(self, manifest, manifest_path, status=None):
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.status_path = Path(manifest["status_path"])
        self.status = initial_status(manifest, manifest_path) if status is None else status
        self.known_ids = {job["id"] for job in manifest["jobs"]}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.scheduler_error = None

    def _save_locked(self):
        self.status["updated_utc"] = utc_now()
        temporary = self.status_path.with_name(self.status_path.name + ".tmp")
        temporary.write_text(json.dumps(self.status, indent=2) + "\n")
        temporary.replace(self.status_path)

    def save(self):
        with self.lock:
            self._save_locked()

    def refresh_jobs(self, jobs):
        with self.lock:
            latest = read_json(self.manifest_path)
            changed = False
            manifest_changed = False
            for job in latest["jobs"]:
                if job["id"] in self.known_ids:
                    if job.get("retry_requested") is True:
                        status = self.job_status(job)
                        if status["state"] == "failed":
                            self.retry_job(job)
                            job.pop("retry_requested")
                            jobs.put(job)
                            changed = True
                            manifest_changed = True
                    continue
                self.known_ids.add(job["id"])
                self.manifest["jobs"].append(job)
                self.status["jobs"][job["id"]] = job_status_record(job)
                if job["state"] == "pending":
                    jobs.put(job)
                changed = True
            if manifest_changed:
                self.manifest_path.write_text(json.dumps(latest, indent=2) + "\n")
            if changed:
                self._save_locked()

    def job_status(self, job):
        return self.status["jobs"][job["id"]]

    def phase_status(self, job, phase):
        return self.job_status(job)["steps"][phase["name"]]

    def retry_job(self, job):
        status = self.job_status(job)
        for phase in job["steps"]:
            phase_status = self.phase_status(job, phase)
            if phase_status["state"] in COMPLETED_STATES:
                continue
            phase_status.update(state="pending", pid=None)
            phase_status.pop("started_utc", None)
            phase_status.pop("finished_utc", None)
            phase_status.pop("error", None)
        status.update(state="pending", gpu=None, phase=None, pid=None)
        status.pop("finished_utc", None)
        status.pop("error", None)

    def start_job(self, gpu, job):
        with self.lock:
            status = self.job_status(job)
            status.update(
                state="running",
                gpu=gpu,
                phase=None,
                pid=None,
                started_utc=utc_now(),
            )
            self.status["workers"][str(gpu)].update(state="running", job=job["id"], pid=None)
            self._save_locked()

    def start_phase(self, gpu, job, phase):
        with self.lock:
            status = self.job_status(job)
            phase_status = self.phase_status(job, phase)
            status["phase"] = phase["name"]
            phase_status.update(state="running", started_utc=utc_now(), pid=None)
            self._save_locked()

    def set_pid(self, gpu, job, phase, pid):
        with self.lock:
            self.job_status(job)["pid"] = pid
            self.phase_status(job, phase)["pid"] = pid
            self.status["workers"][str(gpu)]["pid"] = pid
            self._save_locked()

    def startup_confirmed(self, gpu, job, phase, pid, row):
        event = {"pid": pid, "gpu": gpu, "step": row["step"], "row": row}
        with self.lock:
            self.job_status(job)["startup_confirmed"] = event
            self.phase_status(job, phase)["startup_confirmed"] = event
            self._save_locked()
        print(
            json.dumps(
                {
                    "event": "startup_confirmed",
                    "job": job["id"],
                    "gpu": gpu,
                    "pid": pid,
                    "phase": phase["name"],
                    "step": row["step"],
                }
            ),
            flush=True,
        )

    def finish_phase(self, gpu, job, phase, pid):
        with self.lock:
            phase_status = self.phase_status(job, phase)
            phase_status.update(
                state="completed",
                finished_utc=utc_now(),
                pid=None,
                last_pid=pid,
            )
            self.job_status(job)["pid"] = None
            self._save_locked()
        print(
            json.dumps(
                {
                    "event": "phasefinish",
                    "job": job["id"],
                    "gpu": gpu,
                    "phase": phase["name"],
                    "pid": pid,
                }
            ),
            flush=True,
        )

    def finish_job(self, gpu, job):
        with self.lock:
            status = self.job_status(job)
            status.update(state="completed", phase=None, pid=None, finished_utc=utc_now())
            self.status["workers"][str(gpu)].update(state="idle", job=None, pid=None)
            self._save_locked()
        print(
            json.dumps({"event": "jobfinish", "job": job["id"], "gpu": gpu}),
            flush=True,
        )

    def fail_job(self, gpu, job, error):
        with self.lock:
            status = self.job_status(job)
            phase = status.get("phase")
            if phase is not None:
                phase_status = status["steps"][phase]
                phase_status.update(state="failed", finished_utc=utc_now(), error=repr(error))
            status.update(state="failed", finished_utc=utc_now(), error=repr(error), pid=None)
            self.status["workers"][str(gpu)].update(state="idle", job=None, pid=None)
            self._save_locked()
        print(
            json.dumps(
                {
                    "event": "job_failed",
                    "job": job["id"],
                    "gpu": gpu,
                    "phase": phase,
                    "error": repr(error),
                }
            ),
            flush=True,
        )

    def finish_worker(self, gpu, failed):
        with self.lock:
            worker = self.status["workers"][str(gpu)]
            if not failed and worker["state"] == "running":
                worker.update(state="idle", job=None, pid=None)
            self._save_locked()

    def stop_worker(self, gpu):
        with self.lock:
            self.status["workers"][str(gpu)].update(state="stopped", job=None, pid=None)
            self._save_locked()

    def record_failure(self, gpu, job_id, error):
        self.scheduler_error = error
        with self.lock:
            self.status.update(
                state="failed",
                failure={"gpu": gpu, "job": job_id, "error": repr(error)},
            )
            self._save_locked()

    def run_phase(self, gpu, job, phase):
        self.start_phase(gpu, job, phase)
        logpath = Path(phase["logpath"])
        logpath.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        runtime = self.manifest["launch_environment"]
        environment.update({key: value for key, value in runtime.items() if key != "PATH_prefix"})
        environment["PATH"] = runtime["PATH_prefix"] + os.pathsep + environment["PATH"]
        environment.update(phase.get("env", {}))
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        pid = None
        startup = None
        with logpath.open("a", buffering=1) as log:
            process = subprocess.Popen(
                phase["argv"],
                cwd=phase["cwd"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            pid = process.pid
            self.set_pid(gpu, job, phase, pid)
            for line in process.stdout:
                log.write(line)
                stripped = line.strip()
                if not stripped.startswith("{"):
                    continue
                try:
                    row, _ = JSON_DECODER.raw_decode(stripped)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(row, dict)
                    and phase["name"] == "train"
                    and startup is None
                    and row.get("step", 0) >= 100
                ):
                    startup = row
                    self.startup_confirmed(gpu, job, phase, pid, row)
            returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, phase["argv"])
        if not Path(phase["resultpath"]).exists():
            raise FileNotFoundError(phase["resultpath"])
        self.finish_phase(gpu, job, phase, pid)

    def run_job(self, gpu, job):
        self.start_job(gpu, job)
        for phase in job["steps"]:
            phase_state = self.phase_status(job, phase)["state"]
            if phase_state in COMPLETED_STATES:
                continue
            if phase_state != "pending":
                raise ValueError(f"unsupported status phase state: {phase_state}")
            self.run_phase(gpu, job, phase)
        self.finish_job(gpu, job)


def worker(scheduler, gpu, jobs):
    failed = False
    while not scheduler.stop.is_set():
        try:
            scheduler.refresh_jobs(jobs)
            job = jobs.get_nowait()
        except queue.Empty:
            break
        except Exception as error:
            failed = True
            scheduler.stop_worker(gpu)
            scheduler.record_failure(gpu, None, error)
            scheduler.stop.set()
            break
        try:
            if scheduler.stop.is_set():
                break
            scheduler.run_job(gpu, job)
        except Exception as error:
            failed = True
            scheduler.fail_job(gpu, job, error)
        finally:
            jobs.task_done()
    scheduler.finish_worker(gpu, failed)


def run(manifest_path, resume=False):
    manifest = read_json(manifest_path)
    status = resume_status(manifest, manifest_path) if resume else None
    scheduler = Scheduler(manifest, manifest_path, status=status)
    jobs = queue.Queue()
    for job in manifest["jobs"]:
        job_state = scheduler.job_status(job)["state"]
        if job_state == "pending":
            jobs.put(job)
        elif job_state not in {"pending", "completed", "failed"}:
            raise ValueError(f"unsupported status job state: {job_state}")
    scheduler.status.update(
        state="running",
        started_utc=utc_now(),
        finished_utc=None,
        scheduler_pid=os.getpid(),
    )
    scheduler.save()
    threads = [
        threading.Thread(target=worker, args=(scheduler, gpu, jobs), name=f"gpu-{gpu}")
        for gpu in manifest["worker_gpus"]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if scheduler.scheduler_error is not None:
        scheduler.status.update(
            state="failed",
            finished_utc=utc_now(),
        )
        scheduler.save()
        raise scheduler.scheduler_error
    failed_jobs = [
        {
            "job": job["id"],
            "error": scheduler.job_status(job).get("error"),
        }
        for job in scheduler.manifest["jobs"]
        if scheduler.job_status(job)["state"] == "failed"
    ]
    if failed_jobs:
        scheduler.status.update(state="failed", finished_utc=utc_now())
        scheduler.save()
        print(json.dumps({"event": "queue_failed", "jobs": failed_jobs}), flush=True)
        raise RuntimeError(f"benchmark jobs failed: {json.dumps(failed_jobs)}")
    scheduler.status.update(state="completed", finished_utc=utc_now())
    scheduler.save()
    print(json.dumps({"event": "queue_complete", "status": str(scheduler.status_path)}), flush=True)


def status(manifest_path):
    manifest = read_json(manifest_path)
    status_path = Path(manifest["status_path"])
    current = read_json(status_path) if status_path.exists() else initial_status(manifest, manifest_path)
    display_state = current["state"]
    if display_state == "running" and not scheduler_pid_exists(current["scheduler_pid"]):
        display_state = "interrupted"
    jobs = []
    for job in manifest["jobs"]:
        row = dict(current["jobs"].get(job["id"], job_status_record(job)))
        row["id"] = job["id"]
        row["last_training_progress"] = None
        for phase in job["steps"]:
            if "progress_path" in phase:
                row["last_training_progress"] = last_json_line(phase["progress_path"])
        jobs.append(row)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "status_path": str(status_path),
                "state": display_state,
                "scheduler_pid": current["scheduler_pid"],
                "updated_utc": current["updated_utc"],
                "workers": current["workers"],
                "jobs": jobs,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    if args.status:
        if args.resume:
            parser.error("--status and --resume cannot be used together")
        status(manifest_path)
    else:
        run(manifest_path, resume=args.resume)


if __name__ == "__main__":
    main()
