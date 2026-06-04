"""
Simulated real-time event replay.

Reads events.jsonl and feeds them into the API at the original
event rate (or compressed time). Used for the Part E live dashboard demo.

Usage:
    python pipeline/replay.py [--events events.jsonl] [--api http://localhost:8000]
                              [--speed 10] [--batch 20]
"""

import argparse
import json
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("replay")


def load_events(path: str) -> List[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def sort_by_timestamp(events: List[dict]) -> List[dict]:
    def ts_key(e):
        try:
            return datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0
    return sorted(events, key=ts_key)


def ingest_batch(api: str, events: List[dict]) -> dict:
    resp = requests.post(
        f"{api}/events/ingest",
        json={"events": events},
        timeout=30,
    )
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Simulated real-time event replay")
    parser.add_argument("--events", default="events.jsonl", help="Path to events.jsonl")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--speed", type=float, default=60.0,
                        help="Time compression factor (60 = 1 hour of events plays in 1 minute)")
    parser.add_argument("--batch", type=int, default=20,
                        help="Events per batch sent to API")
    args = parser.parse_args()

    if not Path(args.events).exists():
        logger.error(f"Events file not found: {args.events}")
        sys.exit(1)

    events = sort_by_timestamp(load_events(args.events))
    if not events:
        logger.warning("No events to replay")
        return

    logger.info(f"Loaded {len(events)} events. Speed: {args.speed}x. API: {args.api}")

    # Check API health
    try:
        r = requests.get(f"{args.api}/health", timeout=5)
        logger.info(f"API health: {r.status_code}")
    except Exception as e:
        logger.error(f"Cannot reach API at {args.api}: {e}")
        sys.exit(1)

    # Replay events grouped into time buckets
    first_ts = datetime.fromisoformat(events[0]["timestamp"].replace("Z", "+00:00")).timestamp()
    replay_start = time.time()
    total_accepted = 0

    batch_buffer = []
    last_emit_time = replay_start

    for event in events:
        event_ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp()
        elapsed_event_time = event_ts - first_ts  # how far into the recording this event is
        target_real_time = replay_start + elapsed_event_time / args.speed

        batch_buffer.append(event)

        if len(batch_buffer) >= args.batch or time.time() >= target_real_time:
            now = time.time()
            if now < target_real_time:
                time.sleep(target_real_time - now)

            try:
                result = ingest_batch(args.api, batch_buffer)
                total_accepted += result.get("accepted", 0)
                logger.info(
                    f"Sent batch of {len(batch_buffer)}: "
                    f"accepted={result.get('accepted')}, "
                    f"dup={result.get('duplicate')} | "
                    f"Total accepted: {total_accepted}"
                )
            except Exception as e:
                logger.error(f"Batch error: {e}")

            batch_buffer = []

    # Flush remaining
    if batch_buffer:
        try:
            result = ingest_batch(args.api, batch_buffer)
            total_accepted += result.get("accepted", 0)
        except Exception:
            pass

    logger.info(f"Replay complete. Total accepted: {total_accepted}/{len(events)}")


if __name__ == "__main__":
    main()
