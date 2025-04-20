import re
import json
import matplotlib.pyplot as plt
from datetime import datetime

# File paths
input_file = "justitia.txt"  # Your MPC-formatted observation file
output_json = "justitia_observations_with_location.json"  # Output JSON

# Regex to extract date, RA, Dec, and location code
pattern = re.compile(
    r"(\d{4} \d{2} \d{2}\.\d+)\s+(\d{2} \d{2} \d{2}(?:\.\d+)?)\s+([+-]\d{2} \d{2} \d{2}(?:\.\d+)?).*?([A-Z0-9]{3,})\s*$"
)

observations = []

# Read and parse the file
with open(input_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        date_str, ra_str, dec_str, location = match.groups()

        # Parse date
        try:
            date = datetime.strptime(date_str, "%Y %m %d.%f")
        except:
            continue

        # Convert RA to decimal degrees
        ra_h, ra_m, ra_s = [float(x) for x in ra_str.split()]
        ra_deg = 15 * (ra_h + ra_m / 60 + ra_s / 3600)

        # Convert Dec to decimal degrees
        dec_sign = -1 if dec_str.strip().startswith("-") else 1
        dec_d, dec_m, dec_s = [abs(float(x)) for x in dec_str.split()]
        dec_deg = dec_sign * (dec_d + dec_m / 60 + dec_s / 3600)

        # Save entry
        observations.append(
            {
                "datetime_utc": date.isoformat(),
                "RA_deg": round(ra_deg, 6),
                "Dec_deg": round(dec_deg, 6),
                "location": location,
            }
        )

# Save to JSON
with open(output_json, "w") as f:
    json.dump(observations, f, indent=2)

print(f"Saved {len(observations)} observations to {output_json}")

# --- Scatter plot ---
dates = [datetime.fromisoformat(obs["datetime_utc"]) for obs in observations]
ra_vals = [obs["RA_deg"] for obs in observations]
dec_vals = [obs["Dec_deg"] for obs in observations]

plt.figure(figsize=(12, 6))
plt.scatter(dates, ra_vals, label="RA (deg)", s=10, alpha=0.7)
plt.scatter(dates, dec_vals, label="Dec (deg)", s=10, alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Degrees")
plt.title("RA and Dec of (269) Justitia Over Time (Scatter Plot)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
