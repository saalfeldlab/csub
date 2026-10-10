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
agent container                 submit host                       compute node
───────────────────             ──────────────────────            ─────────────────────────────────
csub CLI / MCP / Python ──ssh──▶ csub-broker ──bsub──▶ LSF ──────▶ wrapper.sh
                                 (forced command,                   ├─ csub-broker --serve job.sock
                                  reads policy)                     └─ podman-run.sh … -- command
                                                                         └─ csub … ──unix socket──▶ broker ─▶ bsub
```

- **Client.** The `csub` command, the `csub-mcp` MCP server and the `csub` Python package. All
  three use the same library.
- **Broker.** `csub-broker` reads one JSON request and writes one JSON response. It validates
  the request, renders a wrapper script, and calls `bsub`, `bjobs` or `bkill`.
- **Transports.** `ssh` from the agent container (a key whose only allowed command is the
  broker), `local` for a person on the submit host, and `unix` from inside a job (a per-job
  broker started by the wrapper; its socket is mounted into the job's container).
- **Sandbox.** Jobs run in `podman-run.sh` from
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
- LSF spools the job script at submit time, so it cannot be changed afterwards. Agent text may
  not contain `#BSUB` lines.
- Environment variables are set inside the sandbox, never on the `bsub` command line.
- The policy file lives outside every mount and is read on every request.

## Using it

### CLI

```sh
csub probe                                         # queues, limits, images, allowed hosts

csub submit --cpus 4 --mem 32G --walltime 30 -- python train.py --epochs 3
csub submit --gpus 1 -q gpu --walltime 120 --name finetune -- python finetune.py
csub submit --allow pypi.org --scratch --walltime 20 -- sh -c 'pip install --user x && python run.py'
csub submit --wait -- python check.py              # submit, wait, print output

csub status                                        # all jobs of this session
csub wait 1234 1235 --timeout 600
csub logs 1234 --tail 50
csub kill 1234
```

Size jobs in `--cpus`, `--mem` and `--walltime`. The broker converts them into LSF slots and
picks a CPU queue from the walltime. GPU jobs name a GPU queue.

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
| `image`        | container image; must be allowed by the policy; ignored under bwrap |
| `env`          | environment variables; names must be allowed by the policy         |
| `allow_hosts`  | hosts the job may reach over HTTP(S); default none                 |
| `scratch`      | private node-local scratch directory as `TMPDIR`                   |
| `claude`       | run Claude Code in the job (CLI, settings, credentials; adds the API hosts) |
| `depends_on`   | job ids, or `{job_id, when}` with `when` = `done`, `ended`, `exit` |
| `mounts`       | `{path, mode}`; filled in by the client from `CSUB_MOUNTS`         |
| `session`      | job group; filled in by the client                                 |

The broker answers with the job id and the resolved request: queue, slots, walltime, mounts and
an estimated maximum cost.

## Inside a job

Output goes to `<cwd>/.csub/jobs/<job_id>/stdout` and `stderr`. `HOME` is
`<cwd>/.csub/home`. `csub` and `import csub` work in any image that has `python3`, because the
wrapper mounts the installed client into the container.

| variable              | value                                          |
| --------------------- | ---------------------------------------------- |
| `CSUB_JOB_ID`         | this job's id                                  |
| `CSUB_WALLTIME_MIN`   | the job's walltime                             |
| `CSUB_DEADLINE_EPOCH` | when the job will be killed (Unix time)        |
| `CSUB_MOUNTS`         | this job's mounts (the default for child jobs) |
| `CSUB_SESSION`        | session; child jobs join it                    |
| `CSUB_IMAGE`          | this job's image (the default for child jobs)  |
| `CSUB_TRANSPORT`      | `unix`                                         |
| `CSUB_SOCKET`         | the per-job broker socket                      |

## Policy

The broker reads `~/.config/csub/broker.toml` on the submit host. Unknown keys are errors.

```toml
[broker]
default_image  = "ghcr.io/example/agent:latest"
allowed_images = ["ghcr.io/example/*:*"]
allowed_roots  = ["/data/lab"]            # mounts must be below these; $HOME is always refused
allow_claude   = false                    # true: jobs may run Claude Code with the submitter's login
sandbox        = "podman"                 # or "bwrap" (no image, no rootless podman, no GPU jobs)
readonly_roots = ["/data/lab/raw"]        # forced read-only
allowed_hosts  = ["pypi.org", "files.pythonhosted.org"]
env_allow      = ["OMP_NUM_THREADS"]

[limits]
max_slots = 64
max_gpus = 4
max_walltime_min = 2880
max_estimated_cost_usd = 50               # per job; 0 = no cap
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
```

Slots are computed as `max(cpus, ceil(mem_mb / mem_per_slot_mb))`. Every job gets a walltime;
the default is 60 minutes for CPU jobs and 120 for GPU jobs, capped by the queue.

## Client configuration

Environment variables, or the same keys in `~/.config/csub/client.toml`.

| variable          | meaning                                                         |
| ----------------- | --------------------------------------------------------------- |
| `CSUB_TRANSPORT`  | `ssh` (default), `local` or `unix`                              |
| `CSUB_SSH_HOST`   | submit host                                                     |
| `CSUB_SSH_USER`   | user on the submit host                                         |
| `CSUB_SSH_KEY`    | private key (default `~/.ssh/csub_ed25519`)                     |
| `CSUB_SSH_PORT`   | port (default 22)                                               |
| `CSUB_SSH_OPTS`   | extra `ssh` options                                             |
| `CSUB_MOUNTS`     | the container's mounts, e.g. `/data/lab/project:rw,/data/ref:ro` |
| `CSUB_SESSION`    | session name (default: derived from the mounts)                 |
| `CSUB_IMAGE`      | default image for jobs                                          |
| `CSUB_BROKER_CMD` | broker command for the `local` transport                        |

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