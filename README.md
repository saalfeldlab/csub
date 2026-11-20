# csub

csub submits jobs to an LSF cluster from inside a sandbox, and runs those jobs in a sandbox too.

AI agents that write and run code should work in a constrained container. If such an agent
could call `bsub` directly, its code would run on the cluster outside any sandbox. With csub,
the agent never talks to LSF. It sends a job request to a broker. The broker runs as the user,
checks the request against a policy the agent cannot change, and submits a job that runs the
command inside a rootless podman container with the same image and the same mounts as the
agent.

Jobs can submit jobs the same way. An agent loop can itself run as a job, and a long training
run can checkpoint and queue its own successor.

Humans use the same tool. csub is meant to be the only way sandboxed code reaches the cluster.

## How it works

```
agent sandbox                    LSF host, outside the sandbox        compute node
───────────────────              ─────────────────────────────        ─────────────────────────────────
csub CLI / MCP / Python ─unix──▶ csub-broker ──bsub──▶ LSF ──────────▶ wrapper.sh
                                 (reads policy)                        ├─ csub-broker --serve job.sock
                                                                       └─ podman-run.sh … -- command
                                                                            └─ csub … ─unix─▶ broker ─▶ bsub
```

- **Client.** The `csub` command, the `csub-mcp` MCP server and the `csub` Python package. All
  three use the same library.
- **Broker.** `csub-broker` reads one JSON request and writes one JSON response. It validates
  the request, renders a wrapper script, and calls `bsub`, `bjobs` or `bkill`. It runs only on
  LSF hosts.
- **Transports.** `unix` from a sandbox: a Unix socket served from outside it, by a broker
  (`csub-broker --serve`, started by a job's wrapper or by you) or, on a host where `bsub` does
  not work, by `csub relay`, which forwards each request over ssh to the broker on the submit
  host. Without a sandbox: `local` on an LSF host, `ssh` elsewhere (a key whose only allowed
  command is the broker).
- **Sandbox.** Jobs run in `podman-run.sh` or `sandbox-run.sh` (bwrap) from
  [agentic-sandbox](https://github.com/JaneliaScientificComputingSystems/agentic-sandbox),
  which handles rootless podman under LSF, GPUs, network isolation and cleanup.

## Security model

- The agent sends a structured request, never raw `bsub` or `podman` flags. The broker decides
  image, mounts, network and identity from its own policy.
- Mounts keep their host paths inside the container. Each mount must be a real (non-symlink)
  directory under an allowed root. The user's home directory is never mounted.
- Jobs have no network unless the request lists hosts that the policy allows. Allowed hosts are
  reached through an HTTP(S) allowlist proxy.
- The broker and the wrapper never write to paths the agent can write to. Broker state lives
  in `~/.csub`, which is never mounted. Job output is written from inside the sandbox.
- LSF spools the job script at submit time, so it cannot be changed afterwards. LSF honours a
  `#BSUB` line anywhere in that script, so agent text may not contain one, and paths may not
  contain control characters such as a newline.
- Extra `bsub` flags (`lsf_extra`) are accepted only if the policy lists the flag by name. Flags
  that run a command or read a file outside the sandbox (`-E`, `-L`, `-f`, `-jsdl`, ...) cannot
  be listed.
- Environment variables are set inside the sandbox, never on the `bsub` command line.
- The policy file lives outside every mount and is read on every request.
- A session may not have more than `max_pending_per_session` jobs pending at once (default
  100). The other limits are per job; this is the one that bounds a loop that keeps
  resubmitting itself.
- Checks happen only in the broker, on an LSF host. `csub relay` forwards requests unchanged.

## Using it

### CLI

```sh
csub probe                                         # queues, limits, images, allowed hosts

csub submit --cpus 4 --mem 32G --walltime 30 -- python train.py --epochs 3
csub submit --gpus 1 -q gpu --walltime 120 --name finetune -- python finetune.py
csub submit --allow pypi.org --scratch --walltime 20 -- sh -c 'pip install --user x && python run.py'
csub submit --wait -- python check.py              # submit, wait, print output

csub status                                        # all jobs of this session, with what each used
csub wait 1234 1235 --timeout 600
csub logs 1234 --tail 50
csub kill 1234
```

Size jobs in `--cpus`, `--mem` and `--walltime`. The broker converts them into LSF slots and
picks a CPU queue from the walltime. GPU jobs name a GPU queue.

`csub status` also reports what each job used, so the next one can be sized from it: CPU time,
run time, the most memory it used (`MAXMEM`) and the memory it was given (`MEMLIM`). With
`--json` and in the Python and MCP results these are `cpu_time_s`, `run_time_s`, `max_mem_mb`,
`avg_mem_mb` and `mem_limit_mb`. The figures are LSF's: from `bjobs` while it lists the job, and
afterwards from the summary LSF wrote at the end of the job's output. A figure LSF does not have,
such as anything for a job still pending, is empty.

Exit codes: 0 ok, 1 job failed, 2 invalid request, 3 policy violation, 4 transport error,
5 LSF error, 6 not found, 7 broker misconfigured, 124 timeout.

### Python

```python
import csub

r = csub.submit(command=["python", "train.py"], cpus=8, mem_mb=64000, walltime_min=60)
result = csub.wait([r.job_id], timeout_s=3600)
print(result.jobs[0].status.state, result.jobs[0].stdout_tail)
print(csub.logs(r.job_id).stderr)
```

Checkpoint and resubmit, from inside a job:

```python
import csub, sys

if training_finished():
    sys.exit(0)  # the last successor stops the chain

me = csub.self_job()  # job id, walltime, deadline
csub.submit(  # successor runs when this job ends, however it ends
    command=[sys.executable, "train.py"],
    walltime_min=me.walltime_min,
    depends_on=[{"job_id": me.job_id, "when": "ended"}],
)
while not training_finished():
    train_step()
    if me.seconds_left() < 300:
        save_checkpoint()
        break
```

### MCP

```json
{"mcpServers": {"csub": {"command": "csub-mcp"}}}
```

Tools: `csub_probe`, `csub_submit`, `csub_status`, `csub_wait`, `csub_kill`, `csub_logs`.
Policy rejections come back as tool errors, so the agent can fix its request.

## Job request

| field          | meaning                                                            |
| -------------- | ------------------------------------------------------------------ |
| `command`      | argv list, or a script body with `shell: true`                     |
| `cwd`          | working directory; must be in a read-write mount                   |
| `name`         | job name                                                           |
| `cpus`         | cores (default 1)                                                  |
| `mem_mb`       | memory in MB                                                       |
| `gpus`         | number of GPUs; needs a GPU queue                                  |
| `gpu_mem_gb`   | minimum GPU memory, on queues that mix GPU models                  |
| `walltime_min` | hard runtime limit in minutes                                      |
| `queue`        | optional for CPU jobs, required for GPU jobs                       |
| `image`        | container image; must be allowed by the policy; ignored when the policy uses bwrap |
| `env`          | environment variables; names must be allowed by the policy         |
| `allow_hosts`  | hosts the job may reach over HTTP(S); default none                 |
| `scratch`      | private node-local scratch directory as `TMPDIR`                   |
| `agent`        | run an agent harness in the job, e.g. `claude`; see `[agents]` in the policy |
| `depends_on`   | job ids, or `{job_id, when}` with `when` = `done`, `ended`, `exit` |
| `mounts`       | `{path, mode}`; filled in by the client from `CSUB_MOUNTS`         |
| `session`      | job group; filled in by the client                                 |
| `lsf_extra`    | extra `bsub` flags, one per entry (`-flag` or `-flag value`); only flags the policy lists |
| `array`        | an array job of this many elements, numbered 1 to `array`; see [Array jobs](#array-jobs) |
| `array_concurrent` | with `array`: at most this many elements running at once       |

The broker answers with the job id and the resolved request: queue, slots, walltime, mounts and
an estimated maximum cost.

## Array jobs

`array: N` (CLI: `--array N`) submits one job with N elements, numbered 1 to N. Every element
runs the same command in its own sandbox, exactly as a single job does, with `CSUB_ARRAY_INDEX`
set to its number and `CSUB_ARRAY_SIZE` to N. `array_concurrent: M` (`--array-concurrent M`)
keeps at most M of them running at once.

```
csub submit --array 200 --array-concurrent 20 -- python sweep.py
csub status                      # one line per array: RUN 37/200 done, 2 failed
csub logs 123456 --index 17      # one element's output
csub kill 123456                 # every element
```

- **One job id.** Status reports the array as a whole, with a count per state and the indices
  that failed (the first 100): `RUN` while anything runs, then `EXIT` if any element failed,
  else `DONE`. The exit code is the highest of its elements. Kill and `depends_on` take the
  array's id; a single element cannot be killed or depended on. A dependency on an array
  follows that state: `done` waits for every element to succeed, `exit` for the array to end
  with at least one failed element, `ended` for every element to end.
- **Output** of element `i` is in `<cwd>/.csub/jobs/<job_id>/<i>/stdout` and `stderr`.
- **What it used.** The figures `csub status` reports are for the array as a whole: the highest
  peak memory and the longest run time of any element, the CPU time of all elements together,
  and the memory each element was given. There is no average memory for an array.
- **Limits.** `limits.max_array_size` bounds N (default 20000; 0 forbids array jobs). The cost
  cap (`max_estimated_cost_usd`) applies to the whole array, since an array is one request: the
  elements' estimates are added up. The response gives the estimate for one element and for
  the array. A pending array counts as one job against `max_pending_per_session`.
- **Child jobs.** An element can submit jobs like any other job, array jobs included.

To rerun the elements that failed, submit a second, smaller array whose elements look up which
original index they stand for. Status names the failed indices (the first 100):

```
csub --json status 123456        # "array": {"failed": [3, 17, 42], ...}
printf '3\n17\n42\n' > failed.txt
csub submit --array 3 -- sh -c 'python sweep.py --index "$(sed -n "${CSUB_ARRAY_INDEX}p" failed.txt)"'
```

Element 1 of the new array reruns index 3, element 2 reruns 17, and so on, so the program must
take its index as an argument. The same pattern runs any sparse set of indices. The rerun is a
new job with its own id and output directory. Index lists (`--array 3,17,42`) and rerunning
failed elements in place are not supported.

## Inside a job

Output goes to `<cwd>/.csub/jobs/<job_id>/stdout` and `stderr`. `HOME` is
`<cwd>/.csub/home`. `csub` and `import csub` work in any image that has `python3`, because the
wrapper mounts the installed client into the container.

| variable              | value                                          |
| --------------------- | ---------------------------------------------- |
| `CSUB_JOB_ID`         | this job's id                                  |
| `CSUB_ARRAY_INDEX`    | in an array job: this element's number, from 1 |
| `CSUB_ARRAY_SIZE`     | in an array job: the number of elements        |
| `CSUB_WALLTIME_MIN`   | the job's walltime                             |
| `CSUB_DEADLINE_EPOCH` | when the job will be killed (Unix time)        |
| `CSUB_MOUNTS`         | this job's mounts (the default for child jobs) |
| `CSUB_SESSION`        | session; child jobs join it                    |
| `CSUB_IMAGE`          | this job's image (the default for child jobs)  |

Child jobs cannot exceed their parent: mounts must lie inside the parent's (no read-only to
read-write), hosts must be among the parent's, and the image and session are the parent's. Status
and kill from inside a job see only the parent's session.
| `CSUB_TRANSPORT`      | `unix`                                         |
| `CSUB_SOCKET`         | the per-job broker socket                      |

## Policy

The broker reads `~/.config/csub/broker.toml` on the submit host. Unknown keys are errors.

```toml
[broker]
default_image  = "ghcr.io/example/agent:latest"
allowed_images = ["ghcr.io/example/*:*"]
allowed_roots  = ["/data/lab"]            # mounts must be below these; $HOME is always refused
sandbox        = "podman"                 # or "bwrap" (no image, no rootless podman; GPU jobs need sandbox-run.sh --gpu)
readonly_roots = ["/data/lab/raw"]        # forced read-only
allowed_hosts  = ["pypi.org", "files.pythonhosted.org"]  # globs allowed; "*" also crosses dots
env_allow      = ["OMP_NUM_THREADS"]
lsf_extra_allow = []                      # bsub flags jobs may add, e.g. ["-R", "-sp"] or
                                          # ["-R select\\[\\w+\\]"] (flag, then a regex for its value)

# Agents starting agents: a job may request `agent = "<mode>"`; the launcher then binds that
# harness's CLI, a throwaway copy of the submitter's settings and the credentials file, and the
# listed hosts are added to the job's allow_hosts. Modes: claude, opencode. Omit to refuse.
# Only the bwrap launcher binds the CLI from $HOME; under podman the image must ship it.
[agents]
claude = ["api.anthropic.com", "claude.ai", "platform.claude.com"]

[limits]
max_slots = 64
max_gpus = 4
max_walltime_min = 2880
max_estimated_cost_usd = 50               # per job; 0 = no cap
max_pending_per_session = 100             # pending jobs per session before submit is refused; 0 = no cap
max_array_size = 20000                    # elements per array job; 0 = no array jobs
slot_price_usd_per_hour = 0.05
cpu_short_queue = "short"                 # used when walltime fits its maximum
cpu_default_queue = "long"

[queues.short]
mem_per_slot_mb = 8192
max_walltime_min = 60

[queues.long]
mem_per_slot_mb = 8192
max_walltime_min = 2880

[queues.gpu]
mem_per_slot_mb = 16384
max_walltime_min = 1440
gpu = true
slots_per_gpu = 8                         # more slots than this per GPU are refused
gpu_price_usd_per_hour = 0.5

[lsf]
profile = "/etc/profile.d/lsf.sh"         # sourced before bsub/bjobs/bkill
norc = false                              # true: run LSF commands with bash --norc (skip ~/.bashrc)
project = "mylab"                         # bsub -P: the lab or project jobs are billed to
umask = "022"                             # the job's umask (default 022; "002" for group-writable output)
```

Slots are computed as `max(cpus, ceil(mem_mb / mem_per_slot_mb))`. Every job gets a walltime;
the default is 60 minutes for CPU jobs and 120 for GPU jobs, capped by the queue.

## Client configuration

Environment variables, or the same keys in `~/.config/csub/client.toml`.

| variable          | meaning                                                                               |
| ----------------- | ------------------------------------------------------------------------------------- |
| `CSUB_TRANSPORT`  | `ssh` (default), `local` or `unix`                                                    |
| `CSUB_SOCKET`     | broker or relay socket for the `unix` transport                                       |
| `CSUB_SSH_HOST`   | submit host                                                                           |
| `CSUB_SSH_USER`   | user on the submit host                                                               |
| `CSUB_SSH_KEY`    | private key (default `~/.ssh/csub_ed25519`)                                           |
| `CSUB_SSH_PORT`   | port (default 22)                                                                     |
| `CSUB_SSH_OPTS`   | extra `ssh` options                                                                   |
| `CSUB_MOUNTS`     | the sandbox's mounts, or without a sandbox the directories jobs may use, e.g. `/data/lab/project:rw,/data/ref:ro` |
| `CSUB_SESSION`    | session name (default: derived from the mounts)                                       |
| `CSUB_IMAGE`      | default image for jobs                                                                |
| `CSUB_BROKER_CMD` | broker command for `local`; for `ssh`, the remote command if the key has no forced command |

## Requirements

- An LSF cluster where compute nodes can run `bsub`.
- Rootless podman on the compute nodes and a checkout of agentic-sandbox.
- Python 3.9 or newer on the submit host and compute nodes. The MCP extra needs Python 3.10.

Installation: [`deploy/INSTALL.md`](deploy/INSTALL.md).

## Development

```sh
pixi install
pixi run test                 # unit, integration (fake LSF, fake sandbox), ssh end-to-end
pixi run -e py39 test-py39    # on Python 3.9
pixi run lint
```