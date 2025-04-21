import re
import json
from datetime import datetime
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import matplotlib.pyplot as plt

# --- CONFIG ---
obs_txt_file = "justitia.txt"
obs_json_file = "justitia_observations_cleaned.json"
residuals_json_file = "justitia_residuals.json"

# --- Parse Observation File (MPC style) ---
pattern = re.compile(
    r"(\d{4} \d{2} \d{2}\.\d+)\s+(\d{2} \d{2} \d{2}(?:\.\d+)?)\s+([+-]\d{2} \d{2} \d{2}(?:\.\d+)?).*?([A-Z0-9]{3,})\s*$"
)

observations = []
epochs_list = []

with open(obs_txt_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        date_str, ra_str, dec_str, location = match.groups()

        try:
            date = datetime.strptime(date_str, "%Y %m %d.%f")
            date_iso = date.isoformat()
            jd = Time(date_iso).jd
        except Exception as e:
            print(f"Error parsing date: {e}")
            continue

        ra_h, ra_m, ra_s = [float(x) for x in ra_str.split()]
        ra_deg = 15 * (ra_h + ra_m / 60 + ra_s / 3600)

        dec_sign = -1 if dec_str.strip().startswith("-") else 1
        dec_d, dec_m, dec_s = [abs(float(x)) for x in dec_str.split()]
        dec_deg = dec_sign * (dec_d + dec_m / 60 + dec_s / 3600)

        observations.append(
            {
                "datetime_utc": date_iso,
                "RA_deg": round(ra_deg, 6),
                "Dec_deg": round(dec_deg, 6),
                "location": location,
                "jd": jd,
            }
        )

# Save cleaned observations
with open(obs_json_file, "w") as f:
    json.dump(observations, f, indent=2)

print(f"Parsed and saved {len(observations)} observations.")

# --- Query JPL Horizons at exact epochs ---
print("Querying JPL Horizons at observation epochs...")

from tqdm import tqdm  # Optional for progress bar


# Split epochs into chunks to avoid long URI issues
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


all_epochs = [obs["jd"] for obs in observations]
residuals = []

print("Querying JPL Horizons in chunks...")

for chunk in tqdm(list(chunk_list(all_epochs, 50))):
    obj = Horizons(
        id="20000",  # Justitia
        location="500",  # Geocentric observer
        epochs=chunk,
        id_type="smallbody",
    )
    eph_chunk = obj.ephemerides()

    for i, model in enumerate(eph_chunk):
        obs = observations.pop(0)  # Safe because we query in same order

        ra_obs = obs["RA_deg"]
        dec_obs = obs["Dec_deg"]
        ra_mod = float(model["RA"])
        dec_mod = float(model["DEC"])

        dra = ra_obs - ra_mod
        if dra > 180:
            dra -= 360
        elif dra < -180:
            dra += 360

        ra_res_arcsec = dra * 3600
        dec_res_arcsec = (dec_obs - dec_mod) * 3600

        residuals.append(
            {
                "datetime_utc": obs["datetime_utc"],
                "RA_residual_arcsec": ra_res_arcsec,
                "Dec_residual_arcsec": dec_res_arcsec,
            }
        )


with open(residuals_json_file, "w") as f:
    json.dump(residuals, f, indent=2)

print(f"Saved {len(residuals)} residuals to {residuals_json_file}.")

# --- Plot Residuals ---
dates = [datetime.fromisoformat(r["datetime_utc"]) for r in residuals]
ra_res = [r["RA_residual_arcsec"] / 3600 for r in residuals]
dec_res = [r["Dec_residual_arcsec"] / 3600 for r in residuals]

plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.scatter(dates, ra_res, s=10, color="blue")
plt.axhline(0, linestyle="--", color="gray")
plt.ylabel("RA Residual (deg)")
plt.title("RA Residuals vs. Horizons")

plt.subplot(2, 1, 2)
plt.scatter(dates, dec_res, s=10, color="green")
plt.axhline(0, linestyle="--", color="gray")
plt.ylabel("Dec Residual (deg)")
plt.xlabel("Date")
plt.title("Dec Residuals vs. Horizons")

plt.tight_layout()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
