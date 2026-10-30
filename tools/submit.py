"""Submit one run specification to PBS on NAS.

    python tools/submit.py runs/w39b_smc_retrieval/wasp39_test1.yaml            # a fresh run
    python tools/submit.py runs/w39b_smc_retrieval/wasp39_test1.yaml --resume   # continue its checkpoint
    python tools/submit.py runs/w39b_smc_retrieval/wasp39_test1.yaml --dry-run  # print the qsub line only

The spec is loaded and validated here (run_setup.load_spec, so a bad file fails before
anything is queued), and its input files are checked as the job will resolve them, from
the repo (run_setup.check_inputs: network, case config, atm and stellar-flux files,
observation products). The PBS job is not part of the spec: the qsub command line asks for
one GH200 node on gpu_long, charges the allocation in $PBS_GROUP and sets the walltime
JOB_HEADROOM_S past the spec's governor (config.walltime_seconds); either unset, or a
walltime past the queue's limit, is refused. The run name is the job name, and the job
receives four comma-free values: the spec file, its digest (run_setup.spec_digest: the
job refuses a spec or base edited after submission), the run's output directory and the
resume flag. A fresh submit into a directory that already holds a run is refused;
``--resume`` needs the checkpoint and, when the last job of this run is still queued or
running, chains the new one after it. Each submission is one line of
<out_dir>/jobs/jobs.txt (job id, resume flag, git revision, spec digest, the qsub line);
the run itself copies the spec it ran into <out_dir>. Run it from the activated
environment on cghfe01/cghfe02 (it imports retrieval_framework).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PBS_SCRIPT = REPO / "tools" / "run_nas.pbs"
# the NAS job every run asks for; the allocation is the submitter's ($PBS_GROUP)
QUEUE, QUEUE_WALL_H = "gpu_long", 120      # every run's queue and its wall limit
SELECT = "1:ncpus=72:ngpus=1:mem=400gb:model=gh200"
JOB_HEADROOM_S = 7200       # the PBS wall ends this long after the run's governor
# infrastructure knobs the job script reads; passed through when set in the caller's shell
PASS_THROUGH = ("CONDA_ENV", "VULCAN_SCRATCH")
ACTIVE_STATES = {"Q", "R", "H", "W", "S", "E", "B"}


def _safe(value: str, what: str) -> str:
    if any(c in value for c in ", '\"\\\n"):
        sys.exit(f"{what} {value!r} holds a character qsub -v cannot carry")
    return value


def _last_active_job(jobs_txt: Path) -> str | None:
    """The run's last job id while PBS still holds it, else None. A job PBS no longer
    knows is gone; any other qstat failure stops the submit."""
    ids = [ln.split()[0] for ln in jobs_txt.read_text().splitlines() if ln.strip()] \
        if jobs_txt.exists() else []
    if not ids:
        return None
    q = subprocess.run(["qstat", "-f", "-F", "json", ids[-1]], capture_output=True, text=True)
    if q.returncode != 0:
        if "Unknown Job Id" in q.stderr:
            return None
        sys.exit(f"qstat {ids[-1]} failed (exit {q.returncode}): {q.stderr.strip()}")
    jobs = json.loads(q.stdout).get("Jobs", {})
    states = [j.get("job_state") for j in jobs.values()]
    return ids[-1] if any(st in ACTIVE_STATES for st in states) else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, help="runs/<case>/<name>.yaml")
    ap.add_argument("--resume", action="store_true", help="continue the run's checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="print the qsub command, submit nothing")
    args = ap.parse_args(argv)

    from retrieval_framework.run_setup import check_inputs, check_run_dir, load_spec, spec_digest
    spec = load_spec(args.spec.resolve())
    os.chdir(REPO)                       # the job's working directory: paths resolve as there
    try:
        check_inputs(spec.cfg)
    except (OSError, ValueError) as e:
        sys.exit(f"{spec.path.name}: {e}")
    group = _safe(os.environ.get("PBS_GROUP", ""), "PBS_GROUP")
    if not group:
        sys.exit("set PBS_GROUP to the allocation the job is charged to (qsub -W group_list)")
    if not spec.cfg.walltime_seconds > 0:
        sys.exit(f"{spec.path.name}: config.walltime_seconds is not set: a PBS run must "
                 "checkpoint and exit before the wall")
    h, rest = divmod(math.ceil(spec.cfg.walltime_seconds + JOB_HEADROOM_S), 3600)
    walltime = f"{h:02d}:{rest // 60:02d}:{rest % 60:02d}"
    if h * 3600 + rest > QUEUE_WALL_H * 3600:
        sys.exit(f"walltime {walltime} (the governor + {JOB_HEADROOM_S} s) is past {QUEUE}'s "
                 f"{QUEUE_WALL_H} h: lower config.walltime_seconds")
    out = spec.cfg.out_dir
    jobs = out / "jobs"
    try:
        check_run_dir(out, args.resume)       # the job's own check, before queueing
    except OSError as e:
        sys.exit(str(e))

    attrs = [f"group_list={group}"]
    prev = _last_active_job(jobs / "jobs.txt") if args.resume else None
    if prev:
        attrs.append(f"depend=afterany:{prev}")
    digest = spec_digest(spec)
    env = [f"VR_SPEC={_safe(str(spec.path), 'spec path')}", f"VR_SPEC_DIGEST={digest}",
           f"VR_OUT={_safe(str(out), 'output directory')}",
           f"VR_RESUME={int(args.resume)}"]
    env += [f"{k}={_safe(os.environ[k], k)}" for k in PASS_THROUGH if k in os.environ]
    cmd = ["qsub", "-N", spec.name, "-q", QUEUE,
           "-l", f"select={SELECT}", "-l", f"walltime={walltime}",
           "-j", "oe", "-o", f"{jobs}/", "-W", ",".join(attrs), "-v", ",".join(env),
           str(PBS_SCRIPT)]
    print(shlex.join(cmd))
    if args.dry_run:
        return
    jobs.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)   # PBS_O_WORKDIR = REPO
    if r.returncode:
        sys.exit(r.stderr.strip() or f"qsub exited {r.returncode}")
    jobid = r.stdout.strip()
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip() or "?"
    with (jobs / "jobs.txt").open("a") as f:
        f.write(f"{jobid} resume={int(args.resume)} git={rev} spec={digest} {shlex.join(cmd)}\n")
    print(jobid)


if __name__ == "__main__":
    main()
