"""Hardware detection and automated acceleration bootstrap for N-ATLAS."""

import importlib
import os
import platform
import shutil
import subprocess
import sys
from typing import Literal

from rich.console import Console
from rich.prompt import Confirm

console = Console()

AccelerationBackend = Literal["cuda", "metal", "cpu"]


def resolve_cuda_dlls() -> None:
    """Ensure CUDA and llama_cpp shared libraries are discoverable by Windows ctypes."""
    if sys.platform != "win32":
        return

    # 1. Discover nvidia package DLL directories (cublas, cuda_runtime, cuda_nvrtc)
    for pkg_name in ["nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc"]:
        try:
            mod = importlib.import_module(pkg_name)
            paths = list(getattr(mod, "__path__", []))
            if paths:
                bin_dir = os.path.join(paths[0], "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)
                    current_path = os.environ.get("PATH", "")
                    if bin_dir not in current_path:
                        os.environ["PATH"] = bin_dir + os.pathsep + current_path
        except (ImportError, AttributeError, OSError):
            pass

    # 2. Discover llama_cpp/lib directory
    try:
        import site

        for s in site.getsitepackages():
            llama_lib = os.path.join(s, "llama_cpp", "lib")
            if os.path.isdir(llama_lib):
                os.add_dll_directory(llama_lib)
                current_path = os.environ.get("PATH", "")
                if llama_lib not in current_path:
                    os.environ["PATH"] = llama_lib + os.pathsep + current_path
    except (ImportError, AttributeError, OSError):
        pass


def detect_gpu_acceleration() -> tuple[AccelerationBackend, str]:
    """Detect available hardware accelerator for local GGUF inference."""
    # 1. Check for Apple Silicon Metal
    if sys.platform == "darwin" and platform.machine() in {"arm64", "aarch64"}:
        return "metal", "Apple Silicon (Metal MPS acceleration)"

    # 2. Check for NVIDIA GPU via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split("\n")[0]
                return "cuda", f"NVIDIA GPU detected ({gpu_info})"
        except (subprocess.SubprocessError, OSError):
            pass

    # 3. Fallback to CPU
    cpu_info = platform.processor() or "Generic CPU"
    return "cpu", f"CPU inference ({cpu_info})"


def get_llama_cpp_install_args(backend: AccelerationBackend) -> list[str]:
    """Get CLI installation arguments for the optimal pre-compiled wheel and CUDA runtimes."""
    base_installer = "uv" if shutil.which("uv") else sys.executable
    is_uv = base_installer == "uv"

    args: list[str] = [base_installer]
    if is_uv:
        args.extend(["pip", "install", "--python", sys.executable])
    else:
        args.extend(["-m", "pip", "install"])

    if backend == "cuda":
        # Target CUDA 12.4 pre-compiled wheels + runtime DLL packages for Windows
        args.extend(
            [
                "llama-cpp-python",
                "nvidia-cuda-runtime-cu12",
                "nvidia-cublas-cu12",
                "--extra-index-url",
                "https://abetlen.github.io/llama-cpp-python/whl/cu124",
            ]
        )
    else:
        # Pre-compiled CPU / Metal wheels
        args.extend(
            [
                "llama-cpp-python",
                "--extra-index-url",
                "https://abetlen.github.io/llama-cpp-python/whl/cpu",
            ]
        )

    return args


def is_llama_cpp_installed() -> bool:
    """Check if llama_cpp can be imported in the active environment."""
    resolve_cuda_dlls()
    try:
        importlib.import_module("llama_cpp")
        return True
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
        return False


def install_llama_cpp(backend: AccelerationBackend) -> bool:
    """Install the pre-compiled binary wheel matching the detected backend."""
    cmd = get_llama_cpp_install_args(backend)
    console.print(f"[cyan]Executing: {' '.join(cmd)}[/cyan]")
    try:
        res = subprocess.run(cmd, check=False)
        return res.returncode == 0 and is_llama_cpp_installed()
    except (subprocess.SubprocessError, OSError) as exc:
        console.print(f"[bold red]Installation command failed: {exc}[/bold red]")
        return False


def ensure_llama_cpp(auto_yes: bool = False) -> bool:
    """Verify llama-cpp-python installation or prompt to auto-install binary wheel."""
    if is_llama_cpp_installed():
        return True

    backend, details = detect_gpu_acceleration()
    console.print(
        "\n[bold yellow]Local GGUF acceleration library (llama-cpp-python) is missing.[/bold yellow]"
    )
    console.print(f"[bold green]Detected Hardware:[/bold green] {details}")

    if backend == "cuda":
        console.print(
            "[dim]A pre-compiled CUDA 12.4 binary wheel with bundled runtime libraries is available. No Visual Studio C++ compilers required.[/dim]"
        )

    should_install = auto_yes
    if not should_install and sys.stdin.isatty():
        should_install = Confirm.ask(
            f"Would you like to auto-install the pre-compiled {backend.upper()} binary wheel now?",
            default=True,
        )

    if not should_install:
        console.print(
            "[yellow]Skipping auto-installation. Server will run in mock mode for LLM generation.[/yellow]"
        )
        return False

    console.print(
        f"[bold cyan]Installing llama-cpp-python for {backend.upper()}...[/bold cyan]"
    )
    success = install_llama_cpp(backend)
    if success:
        console.print(
            "[bold green]llama-cpp-python installed successfully![/bold green]"
        )
        return True

    console.print(
        "[bold red]Installation was not completed. Falling back to mock mode.[/bold red]"
    )
    return False
