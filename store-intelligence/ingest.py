import json
import requests

# Load all events
with open("events_real.jsonl") as f:
    events = [json.loads(line) for line in f]

print(f"Loaded {len(events)} events")

# Send in batches of 500 (API limit)
batch_size = 500

for i in range(0, len(events), batch_size):
    batch = events[i:i + batch_size]

    response = requests.post(
        "http://127.0.0.1:8000/events/ingest",
        json={"events": batch}
    )

    print(
        f"Batch {i//batch_size + 1}: "
        f"Status={response.status_code} "
        f"Response={response.text}"
    )