## SIMULATED OBSERVATIONS OF 2009 DD45
"""
from astroquery.jplhorizons import Horizons
import pandas as pd

# Query for state vectors
obj_vectors = Horizons(
    id="2009 DD45",
    id_type="designation",
    location="500@399",  # Earth-centered
    epochs={"start": "2009-02-27", "stop": "2009-03-06", "step": "1h"},
)

vectors = obj_vectors.vectors()
df_vectors = vectors.to_pandas()

print("State Vectors:")
print(df_vectors[["datetime_str", "x", "y", "z", "vx", "vy", "vz"]])

# Query for observational ephemerides
obj_obs = Horizons(
    id="2009 DD45",
    id_type="designation",
    location="500",  # Geocentric astrometric obs
    epochs={"start": "2009-02-27", "stop": "2009-03-06", "step": "1h"},
)

ephem = obj_obs.ephemerides()
df_obs = ephem.to_pandas()

print("Observational Data:")
print(df_obs[["datetime_str", "RA", "DEC", "delta", "r", "V", "EL"]])
"""

## TRUE OBSERVATIONS OF 2009 DD45
import requests
import pandas as pd

# Define the API endpoint
api_url = "https://ssd-api.jpl.nasa.gov/sbdb.api?des=2009%20DD45"

# Send a GET request to the API
response = requests.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()

    # Extract observation data if available
    if "orbit" in data and "obs" in data["orbit"]:
        observations = data["orbit"]["obs"]

        # Convert the observation data into a pandas DataFrame
        df_observations = pd.DataFrame(observations)

        # Display the first few rows
        print(df_observations.head())

        # Optional: Save to CSV
        df_observations.to_csv("2009DD45_observations.csv", index=False)
    else:
        print("Observation data not found in the API response.")
else:
    print(f"Failed to retrieve data. HTTP status code: {response.status_code}")
