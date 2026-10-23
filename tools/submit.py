"""Submit one run specification to PBS on NAS.

    python tools/submit.py runs/w39b_smc_retrieval/v5.yaml            # a fresh run
    python tools/submit.py runs/w39b_smc_retrieval/v5.yaml --resume   # continue its checkpoint
    python tools/submit.py runs/w39b_smc_retrieval/v5.yaml --dry-run  # print the qsub line only

The spec is loaded and validated here (run_setup.load_spec, so a bad file fails before
anything is queued). The PBS job is not part of the spec: the qsub command line asks for
one GH200 node on gpu_long, charges the allocation in $PBS_GROUP and sets the walltime
JOB_HEADROOM_S past the spec's governor (config.walltime_seconds); either unset is
refused. The run name is the job name, and the job receives three comma-free values:
the spec file, the run's output directory and the resume flag. A fresh submit into a
directory that already holds a run is refused; ``--resume`` needs the checkpoint and,
when the last job of this run is still queued or running, chains the new one after it.
Each submission is recorded under <out_dir>/jobs/: the qsub line and the git revision
in jobs.txt and a copy of the spec as <jobid>.yaml. Run it from the activated
environment on cghfe01/cghfe02 (it imports retrieval_framework).
"""
from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PBS_SCRIPT = REPO / "tools" / "run_nas.pbs"
# the NAS job every run asks for; the allocation is the submitter's ($PBS_GROUP)
QUEUE = "gpu_long"
SELECT = "1:ncpus=72:ngpus=1:mem=400gb:model=gh200"
JOB_HEADROOM_S = 7200       # the PBS wall ends this long after the run's governor
# infrastructure knobs the job script reads; passed through when set in the caller's shell
PASS_THROUGH = ("CONDA_ENV", "VULCAN_SCRATCH", "PROFILE_BATCH", "PROFILE_HLO", "NSYS_JAX", "NSYS_BIN")
ACTIVE_STATES = {"Q", "R", "H", "W", "S", "E", "B"}


def _safe(value: str, what: str) -> str:
    if any(c in value for c in ", '\"\\\n"):
        sys.exit(f"{what} {value!r} holds a character qsub -v cannot carry")
    return value


def _last_active_job(jobs_txt: Path) -> str | None:
    ids = [ln.split()[0] for ln in jobs_txt.read_text().splitlines() if ln.strip()] \
        if jobs_txt.exists() else []
    if not ids:
        return None
    q = subprocess.run(["qstat", ids[-1]], capture_output=True, text=True)
    if q.returncode != 0 or not q.stdout.strip():
        return None
    state = q.stdout.strip().splitlines()[-1].split()[-2]
    return ids[-1] if state in ACTIVE_STATES else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, help="runs/<case>/<name>.yaml")
    ap.add_argument("--resume", action="store_true", help="continue the run's checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="print the qsub command, submit nothing")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO / "src"))
    from retrieval_framework.run_setup import load_spec
    spec = load_spec(args.spec)
    group = _safe(os.environ.get("PBS_GROUP", ""), "PBS_GROUP")
    if not group:
        sys.exit("set PBS_GROUP to the allocation the job is charged to (qsub -W group_list)")
    if not spec.cfg.walltime_seconds > 0:
        sys.exit(f"{spec.path.name}: config.walltime_seconds is not set: a PBS run must "
                 "checkpoint and exit before the wall")
    h, rest = divmod(math.ceil(spec.cfg.walltime_seconds + JOB_HEADROOM_S), 3600)
    walltime = f"{h:02d}:{rest // 60:02d}:{rest % 60:02d}"
    out = spec.cfg.out_dir
    jobs = out / "jobs"
    ckpt = out / "nautilus_checkpoint.hdf5"
    if args.resume and not ckpt.exists():
        sys.exit(f"nothing to resume: {ckpt} does not exist")
    if not args.resume and (ckpt.exists() or (out / "anchors").exists()):
        sys.exit(f"{out} already holds a run: --resume to continue it, or copy the spec to a "
                 "new file name for a fresh run")

    attrs = [f"group_list={group}"]
    prev = _last_active_job(jobs / "jobs.txt") if args.resume else None
    if prev:
        attrs.append(f"depend=afterany:{prev}")
    env = [f"VR_SPEC={_safe(str(spec.path), 'spec path')}",
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
    shutil.copy(spec.path, jobs / f"{jobid.split('.')[0]}.yaml")
    with (jobs / "jobs.txt").open("a") as f:
        f.write(f"{jobid} resume={int(args.resume)} git={rev} {shlex.join(cmd)}\n")
    print(jobid)


if __name__ == "__main__":
    main()
