# Installing csub

Replace `submit.example.org`, `/data/lab` and the queue names with your own.

All paths below start with steps 1–3 (build, install the broker, and write its policy):

- **[Path A: Submit the agent as a csub job](#path-a-submit-the-agent-as-a-csub-job).** Start
  from a trusted shell on the submit host. csub provides the job's sandbox and broker connection.
- **[Path B: Start a standalone agent sandbox](#path-b-start-a-standalone-agent-sandbox-optional).**
  Use this optional setup when the agent runs in a sandbox you launch yourself, beside a
  host-side broker. On a machine without LSF the broker runs its LSF commands over SSH.
- **[Path C: Client on a machine without the shared filesystem](#path-c-client-on-a-machine-without-the-shared-filesystem).**
  A laptop. No broker can run there; the client reaches the broker on the submit host over
  SSH with a key that can run nothing else.

## Prerequisites

- Python 3.9 or newer on the submit host. The same interpreter must exist on the compute nodes,
  because each job restarts the broker there.
- Your home directory is shared between the submit host and the compute nodes.
- Rootless podman works for your account on the compute nodes. Follow the one-time setup in the
  [agentic-sandbox README](https://github.com/JaneliaScientificComputingSystems/agentic-sandbox).

## 1. Build the package

In a checkout of this repository:

```sh
python3 -m pip wheel --no-deps -w dist .
scp dist/csub-0.1.0-py3-none-any.whl submit.example.org:
```

## 2. Install the broker on the submit host

```sh
ssh submit.example.org
pip3 install --user csub-0.1.0-py3-none-any.whl
git clone https://github.com/JaneliaScientificComputingSystems/agentic-sandbox \
    ~/.local/share/csub/agentic-sandbox
```

This installs `~/.local/bin/csub` and `~/.local/bin/csub-broker`.

Every request over ssh starts your login shell on the submit host, and bash reads `~/.bashrc`
even for a non-interactive command. Anything slow in there (a `conda` hook, `nvm`) is paid on
every `csub` call. Keep such lines behind an interactive guard, e.g. `[[ $- == *i* ]] || return`.
Setting `norc = true` in the policy's `[lsf]` table skips `~/.bashrc` for the broker's own
`bsub`/`bjobs`/`bkill` calls, but not for the login shell sshd starts.

## 3. Write the policy

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

The answer contains `"ok":true`. Otherwise it names the policy line that is wrong.

## 4. Choose how to run the agent

### Path A: Submit the agent as a csub job

This is the simpler path when the agent itself can run as a cluster job. From a trusted shell
on the submit host, outside any standalone agent sandbox, first check submission with a small
command:

```sh
cd /data/lab/project
CSUB_TRANSPORT=local CSUB_MOUNTS="$PWD:rw" csub submit --walltime 10 -- hostname
```

The local transport runs the broker directly on the submit host. When the job starts, its
wrapper starts a per-job broker outside the container, mounts that broker's Unix socket into
the container, and sets `CSUB_TRANSPORT=unix` and `CSUB_SOCKET`. The agent can then call csub,
including submitting child jobs, without SSH or network access to the submission host.

Replace `hostname` with your agent's command and choose an appropriate walltime. With the
policy's default `inject_client = true`, the wrapper binds the broker's own csub package into
the container read-only and puts a `csub` command on `PATH`, so the image does not need csub
installed. It does need `python3` (3.9 or newer) on `PATH`, since that command runs
`python3 -m csub.cli`; the default `agentic-sandbox-lite` image lacks it. The job still needs the agent's executable and dependencies, credentials, and any
API hosts permitted by the broker policy and requested with `--allow`. Those are agent-specific
requirements; nothing from Path B is needed for this workflow.

Skip Path B and continue to the [smoke test](#5-smoke-test).

### Path B: Start a standalone agent sandbox (optional)

Use this path if you launch the agent yourself with agentic-sandbox's `sandbox-run.sh` or
`podman-run.sh`, rather than submitting it as a csub job. These launchers do not start a csub
broker for you. How the client reaches one depends on where the sandbox runs:

- Start a broker outside the sandbox and bind its socket in. This is the same mechanism jobs
  use. No network allowance for csub and no SSH client, key or credential inside the sandbox.
  See [Sandbox beside a host-side broker](#sandbox-beside-a-host-side-broker).
- **On a machine without LSF** (a workstation that mounts the cluster filesystem), the same
  broker sends its `bsub`, `bjobs` and `bkill` calls over SSH to the submit host. One policy
  key enables that; see [Machine without LSF](#machine-without-lsf).

Both need the csub client inside the sandbox, so start with the install step. The launch
commands distinguish bwrap and Podman: their environment flags, Python runtimes, and
home-directory handling are not interchangeable. Unlike bwrap, `podman-run.sh` does not mask
credential paths; bind only the specific directories needed, never the entire home directory.

#### Install the client

Run this on the machine that starts the agent. If its home is not shared with the submit host,
also clone agentic-sandbox there at the path used in step 2.

Install the client into its own venv, rather than binding all of `~/.local` and exposing other
tools' state. The venv is not a self-contained Python installation: the base interpreter,
standard library, and shared libraries it uses must also be visible inside the sandbox. In
bwrap, system Python under `/usr` is visible automatically; a conda or pixi environment under
your home directory is not, unless its runtime directory is also bound read-only.

Choose one of these routes for a new venv. The following setup and launch examples use Bash;
`PYTHON_BINDS` carries any extra runtime mount into each launch.

**CLI-only client with the system Python (bwrap or a plain machine).** On RHEL 9 and similar
distributions, `/usr/bin/python3` is Python 3.9. That supports csub, but
not the MCP extra. Install without `[mcp]`:

```bash
CSUB=~/.local/share/csub
PYTHON_BINDS=()
/usr/bin/python3 -m venv "$CSUB/venv"
"$CSUB"/venv/bin/python3 -m pip install /path/to/csub
```

**MCP client with a separately installed Python 3.10+.** Use an existing dedicated Python
runtime, for example a conda or pixi environment; this does not assume a newer system Python
is installed. Replace `PYTHON_PREFIX` with that environment's absolute path:

```bash
CSUB=~/.local/share/csub
PYTHON_PREFIX=/absolute/path/to/python-environment
"$PYTHON_PREFIX/bin/python3" --version   # must be 3.10 or newer
"$PYTHON_PREFIX/bin/python3" -m venv "$CSUB/venv"
"$CSUB"/venv/bin/python3 -m pip install '/path/to/csub[mcp]'
PYTHON_BINDS=(--ro "$PYTHON_PREFIX")
```

Bind the entire runtime prefix (including `bin` and `lib`), not just its Python executable.
If it has symlinks or runtime dependencies outside that prefix and the system directories,
bind those specific directories too. Include `"${PYTHON_BINDS[@]}"` on every sandbox launch,
including launches of the agent that starts the MCP server.

**Podman:** the host's `/usr`-based venv will not run inside a different image, and binding
the host's `/usr` over the image's toolchain is not an option. Use the separately bound
runtime route with a runtime compatible with your image, or install the client inside the
image and adjust its paths in the commands below. Independently of where the client comes
from, the image needs `python3` on its `PATH`: `podman-run.sh` starts its proxy relay with it
whenever any `--allow` is given, and csub jobs run the injected client with it. The default
`agentic-sandbox-lite` image has no `python3` (and no `ssh`); `agentic-sandbox-gpu` has both.

#### Sandbox beside a host-side broker

`csub-broker --serve SOCKET` answers requests on a Unix socket until killed; the job wrapper
uses the same mode. Start one outside the sandbox, bind the socket in read-only, and point the
client at it. Connecting needs no write access to the socket's directory, and a read-only bind
keeps the agent from unlinking or replacing the socket. Keep the path short: Unix socket paths
are limited to about 100 bytes.

`CSUB_MOUNTS` names the project paths the agent may use in jobs. The client, runtime and
socket directories are support files and do not belong in it.

**bwrap (`sandbox-run.sh`):** the launcher binds the current directory automatically, and
`--env` passes exported variables through.

```bash
D=$(mktemp -d /tmp/csub.XXXX)
~/.local/bin/csub-broker --serve "$D/sock" &
cd /data/lab/project
export CSUB_TRANSPORT=unix CSUB_SOCKET="$D/sock" CSUB_MOUNTS="$PWD:rw"
"$CSUB"/agentic-sandbox/scripts/sandbox-run.sh --ro "$D/sock" --ro "$CSUB" "${PYTHON_BINDS[@]}" \
    --env CSUB_TRANSPORT --env CSUB_SOCKET --env CSUB_MOUNTS -- "$CSUB"/venv/bin/csub probe
```

- `--ro "$D/sock"` brings in the broker socket and nothing else. The socket is reachable from
  the sandbox and from host processes running as you, the same as `csub-broker` itself.
- `--ro ~/.local/share/csub` brings in the client venv and the sandbox checkout from step 2,
  without exposing the rest of `~/.local`.
- No `--allow` is needed for csub. Kill the broker when the session ends.

**Podman (`podman-run.sh`):** the image must be able to run the client (see the install step).
Use `--keep-id` so the container user and home match the host; this requires the subordinate
UID/GID setup described in the agentic-sandbox README, and dropping the flag means adapting
the home paths. Podman does not bind the project automatically, so it also needs `--rw "$PWD"`,
and it has no `--env` option, so the variables go after `--`:

```bash
CSUB_IMAGE=your-compatible-agent-image
D=$(mktemp -d /tmp/csub.XXXX)
~/.local/bin/csub-broker --serve "$D/sock" &
cd /data/lab/project
export CSUB_MOUNTS="$PWD:rw"
"$CSUB"/agentic-sandbox/scripts/podman-run.sh --image "$CSUB_IMAGE" --keep-id \
    --rw "$PWD" --ro "$D/sock" --ro "$CSUB" "${PYTHON_BINDS[@]}" -- \
    env CSUB_TRANSPORT=unix CSUB_SOCKET="$D/sock" CSUB_MOUNTS="$CSUB_MOUNTS" \
    "$CSUB"/venv/bin/csub probe
```

#### Start the agent

Once `csub probe` answers, add the same flags to the agent launch you already use. The agent
itself is the launcher's business: its `--claude` (or `--opencode`) mode binds the CLI and its
credentials, and the model endpoints need their own `--allow` entries, exactly as in the
agentic-sandbox README. With Claude Code, the MCP client installed, and bwrap:

```bash
claude mcp add --scope user csub -- "$CSUB"/venv/bin/csub-mcp
"$CSUB"/agentic-sandbox/scripts/sandbox-run.sh --claude \
    --allow api.anthropic.com --allow claude.ai --allow platform.claude.com \
    --ro "$D/sock" --ro "$CSUB" "${PYTHON_BINDS[@]}" \
    --env CSUB_TRANSPORT --env CSUB_SOCKET --env CSUB_MOUNTS -- claude
```

For Podman, use the same MCP registration and keep the image, identity, project, socket and
runtime flags from its probe command. The `--allow` entries make the launcher start its proxy
relay inside the container, which is why the image needs `python3` on `PATH`. Explicitly change to the mounted project before starting
the agent: unlike bwrap, this launcher leaves the working directory at the image's default.

```bash
"$CSUB"/agentic-sandbox/scripts/podman-run.sh --image "$CSUB_IMAGE" --keep-id --claude \
    --rw "$PWD" --ro "$D/sock" --ro "$CSUB" "${PYTHON_BINDS[@]}" \
    --allow api.anthropic.com --allow claude.ai --allow platform.claude.com -- \
    env CSUB_TRANSPORT=unix CSUB_SOCKET="$D/sock" CSUB_MOUNTS="$CSUB_MOUNTS" \
    sh -c 'cd "$1" && exec claude' sh "$PWD"
```

`--scope user` writes the server into `~/.claude.json`, which either launcher's `--claude`
mode copies into the sandbox; a project-scoped `.mcp.json` in the bound working directory
works too. The command must be the venv's full path; no venv activation is assumed. Other
MCP clients take the same server as JSON, with the path spelled out:

```json
{"mcpServers": {"csub": {"command": "/home/you/.local/share/csub/venv/bin/csub-mcp"}}}
```

With the CLI-only client, skip MCP registration and have the agent call the venv's `csub`
executable instead.

#### Machine without LSF

Where `bsub` is missing, the broker still runs on the machine that starts the agent and sends
its `bsub`, `bjobs` and `bkill` calls over SSH to the submit host. Add `submit_host` to the
policy's `[lsf]` table:

```toml
[lsf]
profile  = "/etc/profile.d/lsf.sh"   # sourced on the submit host
submit_host = "submit.example.org"
```

The key is ignored on the submit host itself and inside LSF jobs, where the per-job brokers
run, so a policy file in a shared home serves all three. Whether LSF binaries happen to be
visible on the machine does not matter; on a shared `/misc` or `/opt` they often are, and LSF
still rejects submissions from a host outside the cluster. Nothing csub-specific is needed on
the SSH side: no forced command and no second key. Requirements:

- `ssh submit.example.org true` succeeds without a prompt for the account that starts the
  broker, through `~/.ssh/config`, an agent, or a key already in place. The broker runs outside
  the sandbox, so `~/.ssh` stays unmounted and the agent never sees a credential. One
  multiplexed connection is reused for every call.
- The machine sees the cluster filesystem at the cluster's paths: the home directory (policy,
  broker state, the installed csub package the wrapper mounts into jobs, and the agentic-sandbox
  checkout) and every project path in `CSUB_MOUNTS`. Each job restarts the broker on a compute
  node from the same install path, so install csub into the shared home as in step 2, not into
  a machine-local prefix. A laptop without these mounts cannot run the broker.

Then start `csub-broker --serve` and launch the sandbox exactly as in
[Sandbox beside a host-side broker](#sandbox-beside-a-host-side-broker). Nothing changes inside
the sandbox. Without a sandbox, `CSUB_TRANSPORT=local` on the same machine uses the same broker.

### Path C: Client on a machine without the shared filesystem

A laptop has neither `bsub` nor the cluster filesystem, so no broker can run on it. The client
reaches the broker on the submit host over SSH with a key that can run the broker and nothing
else. This is the client's `ssh` transport, the default.

Create the key on the laptop and, on the submit host, restrict it to the broker:

```sh
ssh-keygen -t ed25519 -N "" -f ~/.ssh/csub_ed25519 -C csub-agent
scp ~/.ssh/csub_ed25519.pub submit.example.org:csub_ed25519.pub
ssh submit.example.org 'echo "restrict,command=\"$HOME/.local/bin/csub-broker\" $(cat csub_ed25519.pub)" >> ~/.ssh/authorized_keys && rm csub_ed25519.pub'
```

Install the client and point it at the submit host:

```sh
python3 -m pip install '/path/to/csub[mcp]'   # Python 3.10 or newer; drop [mcp] for the CLI alone
mkdir -p ~/.config/csub
cat > ~/.config/csub/client.toml <<'EOF'
ssh_host = "submit.example.org"
ssh_key  = "~/.ssh/csub_ed25519"
EOF
CSUB_MOUNTS=/data/lab/project:rw csub probe
```

Paths are the cluster's. `CSUB_MOUNTS` and `--cwd` name directories on the submit host, and the
broker checks them there. The client defaults a job's working directory to its own, which does
not exist on the cluster, so pass `--cwd`:

```sh
CSUB_MOUNTS=/data/lab/project:rw csub submit --cwd /data/lab/project --walltime 10 -- hostname
```

`probe`, `submit`, `status` and `kill` work this way. `csub logs` reads job output from the
filesystem and does not; look in `/data/lab/project/.csub/jobs/<id>/` on the submit host
instead. The MCP server is registered as in Path B, with `csub-mcp` on the laptop's `PATH`:

```json
{"mcpServers": {"csub": {"command": "csub-mcp"}}}
```

If the agent runs in a container on the laptop, that container needs the key and port 22 to the
submit host, and nothing else for csub.

## 5. Smoke test

Run this from the submit host for either path. It does not require Path B's setup:

```sh
cd /data/lab/project
CSUB_TRANSPORT=local /path/to/csub/deploy/smoke.sh
```

It runs a CPU job, a GPU job, a nested submission and two requests that must be rejected. The
GPU job uses the policy's GPU queue with the shortest walltime limit; set
`CSUB_SMOKE_GPU_QUEUE` to pick another.
