#!/usr/bin/env python3
"""Send one MineSeg progress report through the QQ mailbox SMTP service."""

import json
import os
import smtplib
import subprocess
import time
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path

LOG = Path("/root/autodl-tmp/MineSeg/mail/send.log")
SMTP_ENV = Path("/root/autodl-tmp/MineSeg/mail/smtp.env")
RUN_ROOTS = [
    Path("/root/autodl-tmp/MineSeg/v3/runs"),
    Path("/root/autodl-tmp/MineSeg/v2/runs"),
    Path("/root/autodl-tmp/MineSeg/runs"),
    Path("/root/autodl-tmp/SelectiveMine_best_20261024/results"),
]
CLASS_NAMES = [
    "person",
    "cable",
    "tube",
    "indicator",
    "electrical equipment",
    "electronic equipment",
    "mining equipment",
    "rail area",
    "support equipment",
    "door",
    "tools and materials",
    "rescue equipment",
    "container",
    "metal fixture",
    "anchoring equipment",
]


def log(line):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {line}\n")


def pct(value):
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}%"


def num(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def read_last_record(path):
    if path.suffix == ".jsonl":
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - 400_000))
            blob = handle.read().decode("utf-8", "ignore")
        lines = [line for line in blob.splitlines() if line.strip().startswith("{")]
        if not lines:
            return None
        return json.loads(lines[-1])
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return None


def best_miou(path):
    best = None
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = row.get("holdout_miou", row.get("val_miou", row.get("miou")))
                if value is not None:
                    best = value if best is None else max(best, value)
        return best
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("holdout_miou", row.get("val_miou", row.get("miou")))
        if value is not None:
            best = value if best is None else max(best, value)
    return best


def load_config(run_dir):
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def discover_run_dirs():
    found = {}
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        markers = list(root.rglob("history.jsonl")) + list(root.rglob("history.json")) + list(root.rglob("config.json"))
        for path in markers:
            if path.name == "history.json" and path.with_suffix(".jsonl").exists():
                continue
            run_dir = path.parent.resolve()
            found[run_dir] = max(found.get(run_dir, 0), path.stat().st_mtime)
    return [item[0] for item in sorted(found.items(), key=lambda item: item[1], reverse=True)]


def history_of(run_dir):
    jsonl = run_dir / "history.jsonl"
    if jsonl.is_file():
        return jsonl
    history = run_dir / "history.json"
    if history.is_file():
        return history
    return None


def outdir_of(command):
    parts = command.split()
    if "--outdir" not in parts:
        return ""
    index = parts.index("--outdir")
    if index + 1 >= len(parts):
        return ""
    return str(Path(parts[index + 1]).resolve())


def flatten_record(row):
    """Accept the v2 history schema and the SelectiveMine metrics schema."""
    if not isinstance(row, dict):
        return {}
    flat = dict(row)
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("all"), dict):
        summary = metrics["all"]
        flat.setdefault("holdout_miou", summary.get("miou"))
        flat.setdefault("holdout_mpa", summary.get("mpa"))
        flat.setdefault("holdout_pa", summary.get("pa"))
        class_iou = summary.get("class_iou")
        if isinstance(class_iou, dict) and "worst" not in flat:
            ranked = sorted(class_iou.items(), key=lambda item: item[1])[:3]
            flat["worst"] = [{"name": name, "iou": value} for name, value in ranked]
    if flat.get("seconds") is None and row.get("epoch_seconds") is not None:
        flat["seconds"] = row["epoch_seconds"]
    return flat


def worst_text(row):
    worst = row.get("worst")
    if isinstance(worst, list) and worst:
        parts = []
        for item in worst[:3]:
            if isinstance(item, dict):
                parts.append(f"{item.get('name', '?')} {pct(item.get('iou'))}")
        if parts:
            return "，".join(parts)
    iou = row.get("val_iou") or row.get("iou")
    if isinstance(iou, list) and iou:
        ranked = sorted(enumerate(iou), key=lambda item: item[1])[:3]
        parts = []
        for index, value in ranked:
            name = CLASS_NAMES[index] if index < len(CLASS_NAMES) else f"class {index}"
            parts.append(f"{name} {pct(value)}")
        return "，".join(parts)
    return "-"


def train_processes():
    rows = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        cmd_path = Path(f"/proc/{pid}/cmdline")
        try:
            raw = cmd_path.read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()
        except OSError:
            continue
        if "python" not in raw:
            continue
        if not any(name in raw for name in ("train.py", "train_selective.py", "train_debf.py", "finetune_selective.py", "stage2_selective.py", "evaluate.py")):
            continue
        if "spawn_main" in raw or "resource_tracker" in raw:
            continue
        elapsed = "-"
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            start_ticks = int(stat[21])
            hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            uptime = float(Path("/proc/uptime").read_text().split()[0])
            seconds = max(0, int(uptime - start_ticks / hz))
            hours, rem = divmod(seconds, 3600)
            minutes, secs = divmod(rem, 60)
            elapsed = f"{hours}小时{minutes}分{secs}秒" if hours else f"{minutes}分{secs}秒"
        except (OSError, ValueError, IndexError):
            pass
        rows.append((pid, elapsed, raw))
    return rows


def gpu_lines():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"无法读取 GPU：{exc}"]
    if result.returncode != 0:
        return [result.stderr.strip() or "nvidia-smi 失败"]
    lines = []
    for row in result.stdout.strip().splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 5:
            lines.append(row)
            continue
        name, util, used, total, temp = parts[:5]
        lines.append(f"{name}  利用率 {util}%  显存 {used}/{total} MiB  温度 {temp}°C")
    return lines or ["没有检测到 GPU"]


def disk_lines():
    result = subprocess.run(["df", "-h", "/", "/root/autodl-tmp"], check=False, capture_output=True, text=True)
    lines = []
    for row in result.stdout.splitlines()[1:]:
        parts = row.split()
        if len(parts) >= 6:
            lines.append(f"{parts[5]}  已用 {parts[2]} / {parts[1]}（{parts[4]}）  剩余 {parts[3]}")
    return lines or ["无法读取磁盘"]


def describe_run(run_dir, live_outdirs):
    history = history_of(run_dir)
    config = load_config(run_dir)
    args = config.get("args") or {}
    running = str(run_dir) in live_outdirs
    model = args.get("model")
    variant = args.get("variant")
    try:
        title = str(run_dir.relative_to("/root/autodl-tmp"))
    except ValueError:
        title = run_dir.name
    if model:
        title = f"{title}（{model}-{variant or '?'}）"
    status = "进行中" if running else "已停止"
    if history is None:
        height = args.get("height")
        width = args.get("width")
        size = f"  分辨率 {height}x{width}" if height and width else ""
        return f"{title}  [{status}]\n  刚启动，还没有完成第一个 epoch。{size}"
    try:
        last = flatten_record(read_last_record(history))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{title}: 读取失败 {exc}"
    if not last:
        return f"{title}  [{status}]\n  还没有记录"
    epochs = args.get("epochs")
    epoch = last.get("epoch")
    progress = ""
    if epoch and epochs:
        progress = f"{epoch}/{epochs}（{epoch / epochs * 100:.1f}%）"
    elif epoch:
        progress = str(epoch)
    miou = last.get("holdout_miou", last.get("val_miou", last.get("miou")))
    mpa = last.get("holdout_mpa", last.get("mpa"))
    pa = last.get("holdout_pa", last.get("val_pixel_acc", last.get("pa")))
    best = last.get("best")
    if best is None:
        best = best_miou(history)
    seconds = last.get("seconds")
    eta = ""
    if running and seconds and epoch and epochs and epoch < epochs:
        remain = float(seconds) * (epochs - epoch)
        eta = f"  按最近一轮耗时估计还要 {remain / 3600:.1f} 小时"
    return (
        f"{title}  [{status}]\n"
        f"  进度 epoch {progress or '-'}  loss {num(last.get('loss'))}\n"
        f"  最新 mIoU {pct(miou)}  MPA {pct(mpa)}  PA {pct(pa)}  历史最好 mIoU {pct(best)}\n"
        f"  最近一轮 {num(seconds, 1)} 秒  最差类别：{worst_text(last)}{eta}\n"
        f"  训练/验证图像 {config.get('fit_images', config.get('train_images', '-'))}"
        f" / {config.get('holdout_images', '-')}  记录文件 {history}"
    )


def build_report():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    processes = train_processes()
    live_outdirs = {outdir_of(command) for _, _, command in processes if outdir_of(command)}
    run_dirs = discover_run_dirs()
    run_dirs.sort(key=lambda path: (str(path) in live_outdirs, path.stat().st_mtime), reverse=True)
    headline = "当前没有正在运行的训练"
    for _, _, command in processes:
        outdir = outdir_of(command)
        name = Path(outdir).name if outdir else "train.py"
        headline = f"{name} 正在训练"
        if outdir:
            history = history_of(Path(outdir))
            if history:
                try:
                    last = flatten_record(read_last_record(history))
                except (OSError, json.JSONDecodeError):
                    last = None
                if last:
                    miou = last.get("holdout_miou", last.get("val_miou", last.get("miou")))
                    headline = f"{name} epoch {last.get('epoch')} mIoU {pct(miou)}"
        break
    subject = f"MineSeg 实验进展 {now[5:16]} {headline}"
    lines = [
        "MineSeg 实验进展",
        f"时间：{now}",
        f"主机：{os.uname().nodename}  区域：{os.environ.get('AutoDLRegion', '-')}",
        "",
        "【GPU】",
        *gpu_lines(),
        "",
        "【正在运行的训练】",
    ]
    if processes:
        for pid, elapsed, command in processes:
            lines.append(f"pid {pid}  已运行 {elapsed}")
            lines.append(f"  {command}")
    else:
        lines.append("当前没有 train.py 进程。下面是磁盘上最近一次实验的结果。")
    lines.append("")
    lines.append("【实验数据】")
    if run_dirs:
        for run_dir in run_dirs[:8]:
            lines.append(describe_run(run_dir, live_outdirs))
            lines.append("")
    else:
        lines.append("没有找到实验目录。")
    lines.extend(["【磁盘】", *disk_lines(), "", "这封邮件由服务器每小时自动发送。"])
    return subject, "\n".join(lines) + "\n"


def load_smtp():
    if not SMTP_ENV.is_file():
        raise RuntimeError(f"missing {SMTP_ENV}")
    values = {}
    for line in SMTP_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        if not values.get(key):
            raise RuntimeError(f"{SMTP_ENV} is missing {key}")
    values.setdefault("SMTP_PORT", "465")
    values.setdefault("MAIL_TO", values["SMTP_USER"])
    return values


def send_with_smtp(subject, body):
    cfg = load_smtp()
    user = cfg["SMTP_USER"]
    recipient = cfg["MAIL_TO"]
    host = cfg["SMTP_HOST"]
    port = int(cfg["SMTP_PORT"])
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = Header(subject, "utf-8")
    message["From"] = formataddr((str(Header("MineSeg 实验监控", "utf-8")), user))
    message["To"] = recipient
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="qq.com")
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, timeout=30)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
        server.ehlo("qq.com")
        server.starttls()
        server.ehlo("qq.com")
    try:
        server.login(user, cfg["SMTP_PASSWORD"])
        refused = server.sendmail(user, [recipient], message.as_string())
    finally:
        try:
            server.quit()
        except smtplib.SMTPServerDisconnected:
            pass
    if refused:
        raise RuntimeError(f"refused {refused}")
    return f"{host}:{port} accepted the message for {recipient}"


def main():
    subject, body = build_report()
    detail = send_with_smtp(subject, body)
    log(f"OK smtp to {load_smtp()['MAIL_TO']}: {subject} | {detail}")
    print(subject)
    print(body)
    print(detail)


if __name__ == "__main__":
    main()
