"""Sample device/process VRAM around a fresh C++ process (no warmup/reset).

Sampling can miss short spikes. Device memory includes unrelated processes;
process memory is reported only when nvidia-smi supports it (WSL may not).
Allocator peaks are separately reported by the CUDA-enabled C++ executable.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

def read_memory(gpu, pid=None):
    total = subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if total.returncode:
        raise RuntimeError(total.stderr.strip())
    text = total.stdout.strip()
    device_memory = float(text) if text.replace('.', '', 1).isdigit() else None
    own = None
    if pid is not None:
        procs = subprocess.run(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid,used_gpu_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if procs.returncode == 0:
            for line in procs.stdout.splitlines():
                parts = [item.strip() for item in line.split(',')]
                if len(parts) == 2 and parts[0] == str(pid) and parts[1].replace('.', '', 1).isdigit():
                    own = float(parts[1])
    return device_memory, own

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='0', help='Physical nvidia-smi GPU index/UUID')
    p.add_argument('--interval', type=float, default=0.1)
    p.add_argument('--csv', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER, help='-- executable arguments')
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or a.interval <= 0:
        p.error('Provide a command and a positive interval')
    if not shutil.which('nvidia-smi'):
        p.error('nvidia-smi is unavailable; GPU measurement was not performed')
    initial, _ = read_memory(a.gpu)
    for path in (a.csv, a.log):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    start = time.monotonic()
    with open(a.log, 'w') as log, open(a.csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['elapsed_seconds', 'device_used_mib', 'process_used_mib'])
        writer.writerow([0, initial, None])
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                try:
                    total, own = read_memory(a.gpu, proc.pid)
                    row = (time.monotonic() - start, total, own)
                    rows.append(row)
                    writer.writerow(row)
                    file.flush()
                except RuntimeError as error:
                    errors.append(str(error))
                time.sleep(a.interval)
            status = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    peaks = lambda index: max((r[index] for r in rows if r[index] is not None), default=None)
    print(json.dumps({'exit_code': status, 'device_initial_mib': initial,
                      'sampled_device_peak_mib': max(initial, peaks(1)) if initial is not None and peaks(1) is not None else peaks(1),
                      'sampled_process_peak_mib': peaks(2), 'samples': len(rows),
                      'measurement_errors': errors,
                      'note': 'Sampled peaks may miss short spikes; null means unavailable. Device usage includes other processes.'}, indent=2))
    sys.exit(status if status >= 0 else 128 - status)
