#!/bin/bash
#SBATCH --partition=hpc4-el7-3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=temptrans
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# 0) Рабочая директория проекта (там лежат venv/ и temtra/temptrans)
cd "${SLURM_SUBMIT_DIR}"

# 1) Python из venv (НЕ используем activate)
PY="${SLURM_SUBMIT_DIR}/venv/bin/python"

# 2) OpenSSL для модуля ssl (ваш кастомный OpenSSL 1.1.1w)
export LD_LIBRARY_PATH="/s/ls4/users/bur/opt/openssl-1.1.1w/lib:${LD_LIBRARY_PATH:-}"

# 3) libffi (обычно не нужно, но безопасно для _ctypes если потребуется)
export LD_LIBRARY_PATH="/s/ls4/users/bur/opt/libffi-3.2.1/lib64:${LD_LIBRARY_PATH}"

# 4) xTB
export XTB_HOME="/s/ls4/users/bur/opt/xtb"
export PATH="$XTB_HOME/bin:$PATH"
export XTBPATH="$XTB_HOME/share/xtb:${XTBPATH:-}"

# 5) Потоки для numpy/scipy (чтобы не было оверсабскрипшна)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# 6) Диагностика окружения (оставьте хотя бы на первые прогоны)
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "HOME=$HOME"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PY=$PY"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "xtb=$(command -v xtb || echo NOTFOUND)"

ls -l "$PY"
"$PY" -V
"$PY" -c "import ssl; print('ssl', ssl.OPENSSL_VERSION)"
"$PY" -c "import _ctypes; print('ctypes OK')"
"$PY" -c "import numpy, scipy, rdkit; print('py deps OK')"

xtb --version

# 7) Запуск (экспортируем окружение в шаг srun)
srun --export=ALL "$PY" -u -m temtra.sample
