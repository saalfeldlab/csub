# Installing csub

Replace `submit.example.org`, `/data/lab` and the queue names with your own.

Steps 1–3 install the broker on the submit host and test it; step 4 connects the agent to it.

## Prerequisites

- Python 3.9 or newer on the submit host. The same interpreter must exist on the compute nodes,
  because each job runs a copy of the broker on its node.
- Compute nodes can run `bsub`, so jobs can submit jobs.
- Your home directory is shared between the submit host and the compute nodes.
- Rootless podman works for your account on the compute nodes. Follow the one-time setup in the
  [agentic-sandbox README](https://github.com/JaneliaScientificComputingSystems/agentic-sandbox).

## 1. Install the broker on the submit host

On the submit host, install csub from a wheel or from the repository.

From a wheel, if you have one (to build one, run `python3 -m pip wheel --no-deps -w dist .` in a
checkout and copy `dist/*.whl` to the submit host):

```sh
pip3 install --user csub-0.1.0-py3-none-any.whl
```

From the repository (pip builds the package, which needs network access to fetch the build
backend):

```sh
git clone https://github.com/saalfeldlab/csub ~/csub
pip3 install --user ~/csub
```

Either way, also clone agentic-sandbox:

```sh
git clone https://github.com/JaneliaScientificComputingSystems/agentic-sandbox \
    ~/.local/share/csub/agentic-sandbox
```

This installs `~/.local/bin/csub` and `~/.local/bin/csub-broker`. The broker finds
agentic-sandbox at this path, the default of the policy key `sandbox_scripts_dir`.

## 2. Write the policy

```sh
mkdir -p ~/.config/csub
cat > ~/.config/csub/broker.toml <<'EOF'
[broker]
default_image  = "ghcr.io/janeliascientificcomputingsystems/agentic-sandbox-lite:latest"
allowed_images = ["ghcr.io/janeliascientificcomputingsystems/agentic-sandbox-*:*"]
allowed_roots  = ["/data/lab"]
allowed_hosts  = ["pypi.org", "files.pythonhosted.org"]
env_allow      = ["OMP_NUM_THREADS"]

[limits]
max_slots = 64
max_gpus = 4
max_walltime_min = 2880
max_estimated_cost_usd = 50
max_pending_per_session = 100
slot_price_usd_per_hour = 0.05
cpu_short_queue = "short"
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
slots_per_gpu = 8
gpu_price_usd_per_hour = 0.5

[lsf]
profile = "/etc/profile.d/lsf.sh"   # the file that sets up LSF in a login shell; "" if none
EOF
chmod 600 ~/.config/csub/broker.toml
```

Check it:

```sh
echo '{"protocol":1,"op":"probe"}' | ~/.local/bin/csub-broker
```

The answer contains `"ok":true`. Otherwise it names the policy line that is wrong. The
[policy section of the README](../README.md#policy) lists optional keys, such as `sandbox`,
`readonly_roots`, `project` and `norc`.

## 3. Smoke test

On the submit host, in a directory under an allowed root:

```sh
cd /data/lab/project
~/csub/deploy/smoke.sh
```

`smoke.sh` is in the repository's `deploy/` directory; if you installed the wheel, copy it from
the checkout you built it in.

It runs a CPU job, a GPU job, a nested submission and two requests that must be rejected. The
GPU job uses the policy's GPU queue with the shortest walltime limit; set
`CSUB_SMOKE_GPU_QUEUE` to pick another.

## 4. Connect the agent

How csub reaches the broker depends on two things: whether `bsub` works on the machine where
the agent runs, and whether the agent runs in a sandbox you start yourself.

|                                           | without a sandbox      | in a sandbox you start                   |
| ----------------------------------------- | ---------------------- | ---------------------------------------- |
| `bsub` works (submit host, compute node)  | `CSUB_TRANSPORT=local` | socket from `csub-broker --serve`        |
| `bsub` doesn't work (workstation, laptop) | ssh to the broker      | socket from `csub relay`, which uses ssh |

`CSUB_TRANSPORT=local` needs no further setup. ssh is set up in
[4.2](#42-ssh-access-to-the-broker), sandboxes in [4.3](#43-a-sandbox-you-start-yourself). An
agent submitted as a csub job ([4.1](#41-the-agent-as-a-csub-job)) needs none of this: the job
provides both the sandbox and the socket.

### 4.1 The agent as a csub job

From a shell on the submit host, outside any sandbox, submit the command that starts the agent.
For example, a Python agent loop that may run for up to ten hours:

```sh
cd /data/lab/project
CSUB_TRANSPORT=local CSUB_MOUNTS="$PWD:rw" csub submit --walltime 600 \
    --allow api.anthropic.com --env ANTHROPIC_API_KEY -- python agent_loop.py
```

For this, add `api.anthropic.com` to the policy's `allowed_hosts` and `ANTHROPIC_API_KEY` to
`env_allow`. `--env` without a value passes on the key from your shell, so it never appears on
the command line. The agent and its dependencies must be in the image or in a mount.

The job's wrapper starts a broker outside the container and binds its socket in, so the agent
submits child jobs with plain `csub`, without ssh or network access to the submit host. With
`inject_client = true` (the default) the wrapper also binds the broker's csub package and puts
`csub` on `PATH`. That needs `python3` 3.9 or newer in the image; both agentic-sandbox images
have it.

### 4.2 ssh access to the broker

The client runs `csub-broker` on the submit host over ssh, with a key that can run nothing else.
This is the client's default transport. Set it up on the machine where the agent runs:

```sh
ssh-keygen -t ed25519 -N "" -f ~/.ssh/csub_ed25519 -C csub
scp ~/.ssh/csub_ed25519.pub submit.example.org:csub_ed25519.pub
ssh submit.example.org 'echo "restrict,command=\"$HOME/.local/bin/csub-broker\" $(cat csub_ed25519.pub)" >> ~/.ssh/authorized_keys && rm csub_ed25519.pub'
mkdir -p ~/.config/csub
cat > ~/.config/csub/client.toml <<'EOF'
ssh_host = "submit.example.org"
ssh_key  = "~/.ssh/csub_ed25519"
EOF
```

If this machine does not share your home directory with the submit host, install the client,
from the repository or the wheel as in step 1 (Python 3.10 or newer for MCP; drop `[mcp]` for
the CLI alone):

```sh
git clone https://github.com/saalfeldlab/csub ~/csub
python3 -m pip install --user "$HOME/csub[mcp]"
# or: python3 -m pip install --user "csub-0.1.0-py3-none-any.whl[mcp]"
```

Check the connection with `csub probe`.

The MCP server is `{"mcpServers": {"csub": {"command": "csub-mcp"}}}`.

#### Without the cluster filesystem

On a machine that does not mount the cluster's directories, such as a laptop, paths are the
cluster's. `CSUB_MOUNTS` and `--cwd` name directories on the cluster, and `--cwd` is required
because the client's own working directory does not exist there:

```sh
CSUB_MOUNTS=/data/lab/project:rw csub submit --cwd /data/lab/project --walltime 10 -- hostname
```

`probe`, `submit`, `status`, `wait` and `kill` work. Job output is read from the filesystem, so
`csub logs` fails and `csub wait` shows none; it is in `/data/lab/project/.csub/jobs/<id>/` on
the cluster. The same holds for an agent in a sandbox on such a machine.

### 4.3 A sandbox you start yourself

For an agent you launch with agentic-sandbox's `sandbox-run.sh` (bwrap) or `podman-run.sh`. A
process outside the sandbox serves a Unix socket; the sandbox gets the socket read-only and
`CSUB_TRANSPORT=unix`. No ssh client, key or network allowance is needed inside the sandbox.

Run everything in this section in one bash shell, starting with:

```bash
CSUB=~/.local/share/csub        # client venv and agentic-sandbox checkout
D=$(mktemp -d /tmp/csub.XXXX)   # socket directory; socket paths are limited to ~100 bytes
PYTHON_BINDS=()
cd /data/lab/project
export CSUB_MOUNTS="$PWD:rw"
```

`CSUB_MOUNTS` lists the project directories jobs may use; on a machine without the cluster
filesystem these are cluster paths ([4.2](#without-the-cluster-filesystem)). If this machine
does not share your home directory with the submit host, clone agentic-sandbox to
`$CSUB/agentic-sandbox` as in step 1.

#### Install the client

Install the client into its own venv rather than binding all of `~/.local` into the sandbox. A
venv is not self-contained: its base interpreter must be visible in the sandbox too. The
commands install from the checkout `~/csub`; with a wheel, install the wheel file instead.
Choose one:

- **System Python, CLI only, bwrap only.** bwrap shows the host's `/usr`; a podman image has
  its own. RHEL 9's Python 3.9 is too old for the MCP extra.

  ```bash
  /usr/bin/python3 -m venv "$CSUB/venv"
  "$CSUB"/venv/bin/python3 -m pip install ~/csub
  ```

- **A separate Python 3.10 or newer, with MCP, bwrap or podman.** For example a conda or pixi
  environment. Its prefix is bound read-only into the sandbox; for podman it must run in the
  image.

  ```bash
  PYTHON_PREFIX=/absolute/path/to/python-environment
  "$PYTHON_PREFIX"/bin/python3 -m venv "$CSUB/venv"
  "$CSUB"/venv/bin/python3 -m pip install "$HOME/csub[mcp]"
  PYTHON_BINDS=(--ro "$PYTHON_PREFIX")
  ```

#### Serve the socket

Run one of these, and stop it when the session ends:

```bash
~/.local/bin/csub-broker --serve "$D/sock" &   # where bsub works
~/.local/bin/csub relay "$D/sock" &            # elsewhere, with the ssh setup from 4.2
```

`csub relay` forwards each request unchanged to the broker on the submit host, which does all
checks. The ssh key stays in `~/.ssh`, outside the sandbox. If the container runtime cannot pass
a host Unix socket in (for example when containers run in a VM, as with Docker Desktop on
macOS), give the container the key from 4.2 and network access to port 22 on the submit host
instead.

#### With bwrap (`sandbox-run.sh`)

bwrap binds the current directory and passes exported variables with `--env`. Check the
connection:

```bash
export CSUB_TRANSPORT=unix CSUB_SOCKET="$D/sock"   # after starting a relay: it must not see these
"$CSUB"/agentic-sandbox/scripts/sandbox-run.sh --ro "$D/sock" --ro "$CSUB" "${PYTHON_BINDS[@]}" \
    --env CSUB_TRANSPORT --env CSUB_SOCKET --env CSUB_MOUNTS -- "$CSUB"/venv/bin/csub probe
```

Then start the agent with the same flags. For Claude Code with the MCP client (the three
`--allow` hosts are the ones Claude Code needs):

```bash
claude mcp add --scope user csub -- "$CSUB"/venv/bin/csub-mcp
"$CSUB"/agentic-sandbox/scripts/sandbox-run.sh --claude \
    --allow api.anthropic.com --allow claude.ai --allow platform.claude.com \
    --ro "$D/sock" --ro "$CSUB" "${PYTHON_BINDS[@]}" \
    --env CSUB_TRANSPORT --env CSUB_SOCKET --env CSUB_MOUNTS -- claude
```

#### With podman (`podman-run.sh`)

Unlike bwrap, podman binds nothing by default and masks no credentials: bind only what is
needed, never your home directory. `--keep-id` makes the container user and home match yours
and needs the subuid setup from the agentic-sandbox README. There is no `--env`, so variables go
after `--`, and the working directory is the image's. The image needs `python3`, which
`podman-run.sh` uses for `--allow`; both agentic-sandbox images have it. Check the connection:

```bash
IMAGE=ghcr.io/janeliascientificcomputingsystems/agentic-sandbox-lite:latest
"$CSUB"/agentic-sandbox/scripts/podman-run.sh --image "$IMAGE" --keep-id \
    --rw "$PWD" --ro "$D/sock" --ro "$CSUB" "${PYTHON_BINDS[@]}" -- \
    env CSUB_TRANSPORT=unix CSUB_SOCKET="$D/sock" CSUB_MOUNTS="$CSUB_MOUNTS" CSUB_IMAGE="$IMAGE" \
    "$CSUB"/venv/bin/csub probe
```

Then start the agent with the same flags. For Claude Code with the MCP client:

```bash
claude mcp add --scope user csub -- "$CSUB"/venv/bin/csub-mcp
"$CSUB"/agentic-sandbox/scripts/podman-run.sh --image "$IMAGE" --keep-id --claude \
    --allow api.anthropic.com --allow claude.ai --allow platform.claude.com \
    --rw "$PWD" --ro "$D/sock" --ro "$CSUB" "${PYTHON_BINDS[@]}" -- \
    env CSUB_TRANSPORT=unix CSUB_SOCKET="$D/sock" CSUB_MOUNTS="$CSUB_MOUNTS" CSUB_IMAGE="$IMAGE" \
    sh -c 'cd "$1" && exec claude' sh "$PWD"
```

#### Other agents

`--claude` copies `~/.claude.json`, where `claude mcp add --scope user` registered the server,
into the sandbox. Other MCP clients take the server with its full path, since no venv is
activated:

```json
{"mcpServers": {"csub": {"command": "/home/you/.local/share/csub/venv/bin/csub-mcp"}}}
```

With the CLI-only client, the agent calls `"$CSUB"/venv/bin/csub`.

## Tuning

Every request over ssh starts your login shell on the submit host, and bash reads `~/.bashrc`
even for a non-interactive command. Anything slow in there (a `conda` hook, `nvm`) is paid on
every `csub` call. Keep such lines behind an interactive guard, e.g. `[[ $- == *i* ]] || return`.
Setting `norc = true` in the policy's `[lsf]` table skips `~/.bashrc` for the broker's own
`bsub`/`bjobs`/`bkill` calls, but not for the login shell sshd starts.
