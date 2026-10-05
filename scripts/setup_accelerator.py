"""Create an isolated Python 3.11 / Torch 2.10 runtime for notebook training."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu",action="store_true")
    args=parser.parse_args()
    uv=shutil.which("uv")
    if uv is None:
        subprocess.run([sys.executable,"-m","pip","install","uv"],check=True)
        uv=shutil.which("uv")
    if uv is None: raise RuntimeError("uv not on PATH after installation")
    environment=ROOT/(".venv-gpu" if os.name=="nt" else ".venv-colab")
    python=environment/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    if not python.exists(): subprocess.run([uv,"venv","--python","3.11",str(environment)],check=True,cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="ppg-runtime-") as folder:
        requirements=Path(folder)/"requirements.txt"
        subprocess.run([uv,"export","--locked","--no-hashes","--no-dev","--no-emit-package","torch",
                        "--format","requirements-txt","--output-file",str(requirements)],check=True,cwd=ROOT)
        wheel="torch==2.10.0" if args.cpu else "torch==2.10.0+cu126"
        index="https://download.pytorch.org/whl/cpu" if args.cpu else "https://download.pytorch.org/whl/cu126"
        subprocess.run([uv,"pip","install","--python",str(python),"-r",str(requirements),wheel,
                        "--extra-index-url",index,"--index-strategy","unsafe-best-match"],check=True,cwd=ROOT)
    subprocess.run([uv,"pip","install","--python",str(python),
                    "-r",str(ROOT/"requirements-notebook.txt")],check=True,cwd=ROOT)
    subprocess.run([str(python),"-c",
        "import sys,torch,json,ipykernel,nbformat,nbclient; print(json.dumps({'python':sys.version,'torch':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"],
        check=True,cwd=ROOT)
    print(f"Runtime ready: {python}; pinned CPU environment unchanged")
    return 0

if __name__=="__main__": raise SystemExit(main())
