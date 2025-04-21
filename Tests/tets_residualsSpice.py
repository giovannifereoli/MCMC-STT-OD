import re
import json
import spiceypy as spice
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import matplotlib.pyplot as plt
from datetime import datetime

# === CONFIG ===
obs_file = "justitia.txt"
output_json = "justitia_observations_cleaned.json"
residual_json = "justitia_residuals.json"
asteroid_name = "269"  # SPICE ID for Justitia
observer = "EARTH"
frame = "J2000"
abcorr = "LT+S"

# === LOAD SPICE KERNELS ===
# spice.furnsh("naif0012.tls")  # leap seconds
# spice.furnsh("de440s.bsp")    # planetary ephemerides
# spice.furnsh("justitia.bsp")  # asteroid SPK

# === PARSE RAW FILE ===
pattern = re.compile(
    r"(\d{4} \d{2} \d{2}\.\d+)\s+(\d{2} \d{2} \d{2}(?:\.\d+)?)\s+([+-]\d{2} \d{2} \d{2}(?:\.\d+)?).*?([A-Z0-9]{3,})\s*$"
)

observations = []
residuals = []

with open(obs_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        date_str, ra_str, dec_str, location = match.groups()

        try:
            date_obj = datetime.strptime(date_str, "%Y %m %d.%f")
            date_iso = date_obj.isoformat()
            time = Time(date_iso, scale="utc")
        except:
            continue

        # RA conversion
        ra_h, ra_m, ra_s = [float(x) for x in ra_str.split()]
        ra_deg = 15.0 * (ra_h + ra_m / 60 + ra_s / 3600)

        # Dec conversion
        dec_sign = -1 if dec_str.strip().startswith("-") else 1
        dec_d, dec_m, dec_s = [abs(float(x)) for x in dec_str.split()]
        dec_deg = dec_sign * (dec_d + dec_m / 60 + dec_s / 3600)

        observations.append(
            {
                "datetime_utc": date_iso,
                "RA_deg": round(ra_deg, 6),
                "Dec_deg": round(dec_deg, 6),
                "location": location,
            }
        )

        # === Compute Residuals with SPICE ===
        try:
            et = spice.utc2et(date_iso)
            state, _ = spice.spkezr(asteroid_name, et, frame, abcorr, observer)
            x, y, z = state[:3]
            coord = SkyCoord(
                x=x, y=y, z=z, representation_type="cartesian", unit="km", frame="icrs"
            )
            modeled_ra = coord.ra.deg
            modeled_dec = coord.dec.deg

            res_ra = (ra_deg - modeled_ra) * 3600.0  # arcsec
            res_dec = (dec_deg - modeled_dec) * 3600.0  # arcsec

            residuals.append(
                {
                    "datetime_utc": date_iso,
                    "RA_residual_arcsec": res_ra,
                    "Dec_residual_arcsec": res_dec,
                }
            )
        except Exception as e:
            print(f"SPICE error at {date_iso}: {e}")

# === SAVE CLEANED JSON ===
with open(output_json, "w") as f:
    json.dump(observations, f, indent=2)

with open(residual_json, "w") as f:
    json.dump(residuals, f, indent=2)

print(
    f"✅ Saved {len(observations)} cleaned observations and {len(residuals)} residuals."
)

# === PLOT ===
times = [Time(r["datetime_utc"]).datetime for r in residuals]
ra_res = [r["RA_residual_arcsec"] for r in residuals]
dec_res = [r["Dec_residual_arcsec"] for r in residuals]

plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.scatter(times, ra_res, s=10, color="blue")
plt.axhline(0, linestyle="--", color="gray")
plt.ylabel("RA Residual (arcsec)")
plt.title("Residuals in Right Ascension")

plt.subplot(2, 1, 2)
plt.scatter(times, dec_res, s=10, color="green")
plt.axhline(0, linestyle="--", color="gray")
plt.ylabel("Dec Residual (arcsec)")
plt.xlabel("Time")
plt.title("Residuals in Declination")

plt.tight_layout()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
