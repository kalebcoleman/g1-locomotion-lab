#!/usr/bin/env python3
"""Lead's GPU launch coordinator. Serialises job START (flock), checks free GPU memory against an estimate + reserve,
starts the job in its own process group, samples its GPU memory (nvidia-smi) and RSS to <log>.mem.csv, appends to runs.jsonl.
Never touches other processes. Usage: gpu_run.py --name N --est_gb G --log L [--settle_s 90] -- cmd args...
Env: G1R_RESERVE_GB (default 20) = free memory that must remain after the estimate (other users' jobs)."""
import argparse, fcntl, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--est_gb", type=float, required=True)
ap.add_argument("--log", required=True); ap.add_argument("--settle_s", type=int, default=90)
ap.add_argument("--max_wait_s", type=int, default=1800)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args(); cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
reserve = float(os.environ.get("G1R_RESERVE_GB", "20"))

def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
def free_gb(): return float(smi("memory.free")) / 1024
lock = open(os.path.join(ROOT, "logs", "gpu_launch.lock"), "w")
t_wait = time.time()
fcntl.flock(lock, fcntl.LOCK_EX)                      # only one launcher at a time
while free_gb() < a.est_gb + reserve:
    if time.time() - t_wait > a.max_wait_s:
        print(f"GPU_RUN_ABORT {a.name}: free {free_gb():.1f} GB < est {a.est_gb}+reserve {reserve}", flush=True); sys.exit(3)
    time.sleep(10)
os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
logf = open(a.log, "w")
p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
rec = dict(name=a.name, pid=p.pid, cmd=cmd, est_gb=a.est_gb, start=time.strftime("%F %T"), free_gb_at_start=free_gb(), log=a.log)
open(os.path.join(ROOT, "logs", "runs.jsonl"), "a").write(json.dumps(rec) + "\n")
print("GPU_RUN_STARTED", json.dumps(rec), flush=True)
mem_csv = open(a.log + ".mem.csv", "w"); mem_csv.write("t_s,gpu_mem_mib_job,gpu_mem_mib_total_used,gpu_util_pct,rss_mib\n")
t0 = time.time(); peak = 0.0; released = False
while p.poll() is None:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    # job may spawn children; count all processes in our session
    sid = p.pid; mine = 0.0
    for line in out.strip().splitlines():
        try:
            pid, mem = [x.strip() for x in line.split(",")]
            if os.getsid(int(pid)) == sid: mine += float(mem)
        except Exception: pass
    try: rss = int(open(f"/proc/{p.pid}/statm").read().split()[1]) * 4096 / 2**20
    except Exception: rss = 0
    peak = max(peak, mine)
    mem_csv.write(f"{time.time()-t0:.0f},{mine:.0f},{smi('memory.used')},{smi('utilization.gpu')},{rss:.0f}\n"); mem_csv.flush()
    if not released and time.time() - t0 > a.settle_s:
        fcntl.flock(lock, fcntl.LOCK_UN); released = True    # next job may start once this one's startup memory is visible
    time.sleep(2)
if not released: fcntl.flock(lock, fcntl.LOCK_UN)
rec.update(exit=p.returncode, wall_s=time.time() - t0, peak_gpu_mib_job=peak, end=time.strftime("%F %T"))
open(os.path.join(ROOT, "logs", "runs.jsonl"), "a").write(json.dumps(dict(name=a.name, finished=rec)) + "\n")
print("GPU_RUN_DONE", json.dumps(dict(name=a.name, exit=p.returncode, wall_s=round(rec["wall_s"]), peak_gpu_mib_job=peak)), flush=True)
