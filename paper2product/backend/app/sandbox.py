"""Stage E - controlled execution.

DockerSandbox  : real isolation (no network, read-only root, dropped capabilities, resource limits,
                 non-root user, no docker socket, no secrets). Used automatically when Docker works.
LocalSandbox   : development fallback. Scrubbed env + rlimits + timeout + process-group kill.
                 It is NOT a security boundary (no network or filesystem isolation) and is labelled as such
                 in every result, report and manifest.
"""
from __future__ import annotations
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict

IS_WINDOWS = os.name == "nt"
if not IS_WINDOWS:                      # `resource` does not exist on Windows
    import resource

from .security import safe_relpath, SecurityError


@dataclass
class SandboxResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    truncated: bool = False
    mode: str = "local"
    isolated: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


import re as _re
_ARG_RX = _re.compile(r"^[A-Za-z0-9_./-]{1,100}$")


def _check_arg(a: str) -> None:
    if not _ARG_RX.match(a) or ".." in a or a.startswith("/"):
        raise SecurityError(f"unsafe argument: {a!r}")


def validate_python_args(args: list[str]) -> list[str]:
    """Only two invocation shapes are allowed: `-m unittest ...` or `<relative script>.py ...`."""
    if not args:
        raise SecurityError("empty command")
    if args[0] == "-m":
        if len(args) < 2 or args[1] != "unittest":
            raise SecurityError("only 'python -m unittest' is allowed")
        for a in args[2:]:
            if a.startswith("-") and a in ("-s", "-t", "-v", "-p"):
                continue
            _check_arg(a)
        return list(args)
    script = safe_relpath(args[0])
    if not script.endswith(".py"):
        raise SecurityError("only .py scripts may be executed")
    for a in args[1:]:
        _check_arg(a)
    return [script] + list(args[1:])


def _clip(b: bytes, limit: int) -> tuple[str, bool]:
    t = len(b) > limit
    return b[:limit].decode("utf-8", errors="replace") + ("\n[output truncated]" if t else ""), t


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and everything it started (works on Windows and POSIX)."""
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


class LocalSandbox:
    mode = "local"
    isolated = False
    description = ("LOCAL DEVELOPMENT MODE - NOT SECURE: subprocess with a timeout, output cap and a scrubbed environment "\
                   "(plus memory/CPU limits on Linux/macOS), but no network or filesystem isolation. "\
                   "Use Docker for untrusted code.")

    @staticmethod
    def _env(workdir: str) -> dict:
        env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        if IS_WINDOWS:
            # Windows needs these for Python/NumPy DLL loading and temp files; nothing secret is passed through.
            for k in ("SYSTEMROOT", "SystemDrive", "PATH", "PATHEXT", "COMSPEC", "WINDIR", "NUMBER_OF_PROCESSORS"):
                if os.environ.get(k):
                    env[k] = os.environ[k]
            env["TEMP"] = env["TMP"] = workdir
            env["USERPROFILE"] = env["HOME"] = workdir
        else:
            env.update({"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": workdir, "LANG": "C.UTF-8"})
        return env

    def run_python(self, workdir: str, args: list[str], timeout_s: int, memory_mb: int, max_output_kb: int) -> SandboxResult:
        args = validate_python_args(args)
        argv = [sys.executable, "-E", "-s", "-X", "utf8"] + args
        env = self._env(workdir)

        def limits() -> None:               # POSIX only
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 5, timeout_s + 5))
            mem = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        popen_kw: dict = {}
        if IS_WINDOWS:
            popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            popen_kw["preexec_fn"] = limits

        tmp = tempfile.mkdtemp(prefix="p2p_out_")
        out_path, err_path = os.path.join(tmp, "out"), os.path.join(tmp, "err")
        start = time.time()
        timed_out = False
        try:
            with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
                proc = subprocess.Popen(argv, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                                        stdout=fo, stderr=fe, **popen_kw)
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_tree(proc)
                    proc.wait()
            limit = max_output_kb * 1024
            with open(out_path, "rb") as f:
                out, t1 = _clip(f.read(limit + 1), limit)
            with open(err_path, "rb") as f:
                err, t2 = _clip(f.read(limit + 1), limit)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        code = proc.returncode if not timed_out else -9
        return SandboxResult(argv=["python"] + args, exit_code=code, stdout=out, stderr=err,
                             duration_s=round(time.time() - start, 3), timed_out=timed_out,
                             truncated=t1 or t2, mode="local", isolated=False, note=self.description)


SANDBOX_DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir numpy scikit-learn pandas scipy
"""
DEFAULT_IMAGE = "paper2product-sandbox:1"


class DockerSandbox:
    mode = "docker"
    isolated = True
    description = "Docker isolation: --network none, read-only rootfs, cap-drop ALL, non-root, resource limits."

    def __init__(self, image: str = DEFAULT_IMAGE):
        self.image = image

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        try:
            return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
        except Exception:
            return False

    def image_ready(self) -> bool:
        try:
            return subprocess.run(["docker", "image", "inspect", self.image], capture_output=True, timeout=30).returncode == 0
        except Exception:
            return False

    def ensure_image(self) -> bool:
        """The sandbox runs with NO network, so the image must already contain numpy/scikit-learn.
        Our own image is built once (needs internet only for that one-time build). Other images are used as-is."""
        if self.image_ready():
            return True
        if self.image != DEFAULT_IMAGE:
            try:
                return subprocess.run(["docker", "pull", self.image], capture_output=True, timeout=600).returncode == 0
            except Exception:
                return False
        d = tempfile.mkdtemp(prefix="p2p_img_")
        try:
            with open(os.path.join(d, "Dockerfile"), "w", encoding="utf-8") as f:
                f.write(SANDBOX_DOCKERFILE)
            r = subprocess.run(["docker", "build", "-t", self.image, d], capture_output=True, timeout=1200)
            return r.returncode == 0
        except Exception:
            return False
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def build_command(self, workdir: str, args: list[str], timeout_s: int, memory_mb: int, name: str) -> list[str]:
        args = validate_python_args(args)
        src = os.path.realpath(workdir)
        if "," in src or '"' in src:
            raise SecurityError("work directory path contains characters that are unsafe for a Docker mount")
        return ["docker", "run", "--rm", "--name", name,
                "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m,noexec",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "128", "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
                "--cpus", "1.0", "--user", "65534:65534",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "HOME=/tmp",
                "-e", "PYTHONUTF8=1",
                "--mount", f"type=bind,source={src},target=/work",     # --mount also handles Windows paths (C:\\...)
                "-w", "/work",
                self.image, "python"] + args

    @staticmethod
    def _make_writable(workdir: str) -> None:
        """The container runs as an unprivileged user (65534); on Linux it must be able to write artifacts/."""
        if IS_WINDOWS:
            return
        for dp, dn, fn in os.walk(workdir):
            try:
                os.chmod(dp, 0o777)
                for f in fn:
                    os.chmod(os.path.join(dp, f), 0o666)
            except OSError:
                pass

    def run_python(self, workdir: str, args: list[str], timeout_s: int, memory_mb: int, max_output_kb: int) -> SandboxResult:
        name = f"p2p-{uuid.uuid4().hex[:10]}"
        cmd = self.build_command(workdir, args, timeout_s, memory_mb, name)
        self._make_writable(workdir)
        start = time.time()
        timed_out = False
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout_s + 10)
            out, err, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            subprocess.run(["docker", "kill", name], capture_output=True)
            out, err, code = e.stdout or b"", e.stderr or b"", -9
        limit = max_output_kb * 1024
        o, t1 = _clip(out, limit)
        er, t2 = _clip(err, limit)
        return SandboxResult(argv=["python"] + list(args), exit_code=code, stdout=o, stderr=er,
                             duration_s=round(time.time() - start, 3), timed_out=timed_out,
                             truncated=t1 or t2, mode="docker", isolated=True, note=self.description)


class AutoSandbox:
    """Docker is installed but the sandbox image is not built yet: build it in the background (one-time, needs
    internet) and use the clearly-labelled local sandbox until it is ready. Never blocks app startup."""

    def __init__(self, image: str = DEFAULT_IMAGE):
        import threading
        self.local, self.docker = LocalSandbox(), DockerSandbox(image)
        self.build_state = "building"          # building | ready | failed
        threading.Thread(target=self._build, daemon=True, name="p2p-image-build").start()

    def _build(self) -> None:
        self.build_state = "ready" if self.docker.ensure_image() else "failed"

    def current(self):
        return self.docker if self.build_state == "ready" else self.local

    @property
    def mode(self) -> str:
        return self.current().mode

    @property
    def isolated(self) -> bool:
        return self.current().isolated

    @property
    def description(self) -> str:
        c = self.current()
        if c is self.docker:
            return c.description
        why = {"building": "the Docker sandbox image is still being built in the background (first run only)",
               "failed": "the Docker sandbox image could not be built (internet needed once; run `docker build` or retry)"}
        return c.description + " [" + why.get(self.build_state, "") + "]"

    def run_python(self, *a, **kw) -> SandboxResult:
        return self.current().run_python(*a, **kw)


def get_sandbox(mode: str = "auto", image: str = DEFAULT_IMAGE):
    """auto  : Docker if it works (building the image in the background if needed), otherwise the
               clearly-labelled local dev sandbox.
       docker: Docker or an error - never silently downgraded.
       local : local dev sandbox (NOT secure)."""
    if mode == "local":
        return LocalSandbox()
    if mode == "docker":
        if not DockerSandbox.available():
            raise RuntimeError("P2P_SANDBOX=docker but Docker is not available (is Docker Desktop running?)")
        sb = DockerSandbox(image)
        if not sb.ensure_image():
            raise RuntimeError(f"Docker sandbox image '{image}' is not available and could not be built")
        return sb
    if DockerSandbox.available():
        sb = DockerSandbox(image)
        return sb if sb.image_ready() else AutoSandbox(image)
    return LocalSandbox()
