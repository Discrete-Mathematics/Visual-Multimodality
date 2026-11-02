#!/usr/bin/env python3
"""Log free disk space and mail when the data disk drops below 8 GB."""

import subprocess
import time
from pathlib import Path

LOG = Path("/root/autodl-tmp/MineSeg/v2/runs/relay_retrain/disk_monitor.log")
MAIL = ["/usr/bin/python3", "/root/autodl-tmp/MineSeg/mail/report_progress.py", "--relay-only", "--disk-alert"]
THRESHOLD_GB = 8.0
INTERVAL = 300
MAIL_GAP = 3600


def free_gb(path):
    import os
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / (1024 ** 3)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    last_mail = 0.0
    while True:
        free_root = free_gb("/")
        free_data = free_gb("/root/autodl-tmp")
        line = time.strftime("%Y-%m-%d %H:%M:%S") + f"  / {free_root:.2f} GB free   /root/autodl-tmp {free_data:.2f} GB free"
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        if free_data < THRESHOLD_GB and time.time() - last_mail > MAIL_GAP:
            subprocess.run(MAIL, check=False)
            last_mail = time.time()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
