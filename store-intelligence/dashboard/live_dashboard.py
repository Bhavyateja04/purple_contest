"""
Terminal live dashboard — shows store metrics updating in real time.
Uses the `rich` library for a beautiful terminal UI.

Usage:
    python dashboard/live_dashboard.py [--api http://localhost:8000] [--store STORE_BLR_002]
"""

import argparse
import json
import time
import sys
from datetime import datetime
from typing import Optional

try:
    import requests
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.progress import BarColumn, Progress, SpinnerColumn
    from rich import box
except ImportError:
    print("Install dependencies: pip install rich requests")
    sys.exit(1)

console = Console()


def fetch(url: str, timeout: float = 5.0) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code in (200, 207):
            return r.json()
    except Exception:
        pass
    return None


def build_metrics_panel(data: Optional[dict]) -> Panel:
    if not data:
        return Panel("[red]⚠ Could not reach API[/red]", title="Metrics", border_style="red")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="cyan", width=28)
    table.add_column("Value", style="bold white", justify="right", width=18)

    table.add_row("Unique Visitors Today", str(data.get("unique_visitors", "—")))
    conv = data.get("conversion_rate", 0)
    table.add_row("Conversion Rate", f"{conv*100:.1f}%")
    basket = data.get("avg_basket_value_inr", 0)
    table.add_row("Avg Basket (INR)", f"₹{basket:,.0f}")
    queue = data.get("queue_depth_now", 0)
    q_color = "green" if queue < 3 else ("yellow" if queue < 6 else "red")
    table.add_row("Queue Depth Now", f"[{q_color}]{queue}[/{q_color}]")
    abandon = data.get("abandonment_rate", 0)
    table.add_row("Abandonment Rate", f"{abandon*100:.1f}%")
    table.add_row("Transactions Today", str(data.get("total_transactions", "—")))
    table.add_row("As Of", data.get("as_of", "—")[-8:])

    return Panel(table, title="[bold magenta]Live Metrics[/bold magenta]", border_style="magenta")


def build_funnel_panel(data: Optional[dict]) -> Panel:
    if not data or "stages" not in data:
        return Panel("[dim]No funnel data[/dim]", title="Funnel", border_style="dim")

    lines = []
    max_count = max((s["count"] for s in data["stages"]), default=1) or 1
    bar_width = 30

    for stage in data["stages"]:
        count = stage["count"]
        label = stage["stage"].ljust(16)
        fill = int((count / max_count) * bar_width)
        bar = "█" * fill + "░" * (bar_width - fill)
        drop = f" (-{stage['drop_off_pct']:.0f}%)" if stage["drop_off_pct"] > 0 else ""
        lines.append(f"[cyan]{label}[/cyan] [green]{bar}[/green] {count:>4}{drop}")

    return Panel("\n".join(lines), title="[bold blue]Conversion Funnel[/bold blue]", border_style="blue")


def build_anomalies_panel(data: Optional[dict]) -> Panel:
    if not data or "active_anomalies" not in data:
        return Panel("[dim]Checking...[/dim]", title="Anomalies", border_style="dim")

    anomalies = data["active_anomalies"]
    if not anomalies:
        return Panel("[green]✓ No active anomalies[/green]", title="Anomalies", border_style="green")

    lines = []
    colors = {"INFO": "blue", "WARN": "yellow", "CRITICAL": "red"}
    for a in anomalies[:5]:
        sev = a.get("severity", "INFO")
        c = colors.get(sev, "white")
        atype = a.get("anomaly_type", "").replace("_", " ")
        desc = a.get("description", "")[:60]
        lines.append(f"[{c}]● {sev}[/{c}] {atype}")
        lines.append(f"  [dim]{desc}[/dim]")

    return Panel("\n".join(lines), title="[bold red]Active Anomalies[/bold red]", border_style="red")


def build_heatmap_panel(data: Optional[dict]) -> Panel:
    if not data or "cells" not in data:
        return Panel("[dim]No heatmap data[/dim]", title="Heatmap", border_style="dim")

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("Zone", style="cyan", width=18)
    table.add_column("Visits", justify="right", width=7)
    table.add_column("Avg Dwell", justify="right", width=10)
    table.add_column("Score", width=22)

    for cell in data["cells"][:10]:
        score = cell["normalized_score"]
        fill = int(score / 5)
        bar = "▓" * fill + "░" * (20 - fill)
        color = "green" if score > 66 else ("yellow" if score > 33 else "red")
        table.add_row(
            cell["zone_name"][:18],
            str(cell["visit_frequency"]),
            f"{cell['avg_dwell_sec']:.0f}s",
            f"[{color}]{bar}[/{color}]",
        )

    conf = data.get("data_confidence", True)
    conf_note = "" if conf else "\n[yellow]⚠ Low data confidence (<20 sessions)[/yellow]"
    return Panel(str(table) + conf_note, title="[bold]Zone Heatmap[/bold]", border_style="cyan")


def build_layout(
    store_id: str,
    metrics: Optional[dict],
    funnel: Optional[dict],
    anomalies: Optional[dict],
    heatmap: Optional[dict],
    tick: int,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="metrics", ratio=2),
        Layout(name="funnel", ratio=3),
    )
    layout["right"].split_column(
        Layout(name="heatmap", ratio=3),
        Layout(name="anomalies", ratio=2),
    )

    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[tick % 10]
    layout["header"].update(Panel(
        f"[bold magenta]🏪 Store Intelligence — {store_id}[/bold magenta]  "
        f"[dim]{spinner} Live · {datetime.now().strftime('%H:%M:%S')}[/dim]",
        border_style="magenta"
    ))
    layout["metrics"].update(build_metrics_panel(metrics))
    layout["funnel"].update(build_funnel_panel(funnel))
    layout["heatmap"].update(build_heatmap_panel(heatmap))
    layout["anomalies"].update(build_anomalies_panel(anomalies))
    layout["footer"].update(Panel(
        "[dim]Press Ctrl+C to exit  ·  Refreshes every 5 seconds[/dim]",
        border_style="dim"
    ))
    return layout


def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Terminal Dashboard")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--store", default="STORE_BLR_002", help="Store ID")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval (seconds)")
    args = parser.parse_args()

    api = args.api.rstrip("/")
    store = args.store

    console.print(f"[green]Connecting to {api}...[/green]")

    metrics = funnel = anomalies = heatmap = None
    tick = 0
    last_slow_refresh = 0

    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            now = time.time()

            # Fast refresh: metrics + anomalies every 5s
            metrics = fetch(f"{api}/stores/{store}/metrics")
            anomalies = fetch(f"{api}/stores/{store}/anomalies")

            # Slow refresh: funnel + heatmap every 30s
            if now - last_slow_refresh > 30:
                funnel = fetch(f"{api}/stores/{store}/funnel")
                heatmap = fetch(f"{api}/stores/{store}/heatmap")
                last_slow_refresh = now

            layout = build_layout(store, metrics, funnel, anomalies, heatmap, tick)
            live.update(layout)
            tick += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[green]Dashboard closed.[/green]")
