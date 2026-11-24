import numpy as np 

import matplotlib.pyplot as plt 

import requests

import json

simulation_results = []

url = "https://api.open-meteo.com/v1/forecast"

params = {
"latitude": 51.51,
    "longitude": -0.13,
    "hourly": "wind_speed_10m,temperature_2m,relative_humidity_2m,precipitation,cloud_cover",
    "forecast_days": 14,
    "timezone": "Europe/London"

}


data = requests.get(url, params=params, timeout=10)
data.raise_for_status()
json_data = data.json() 

wind = np.array(json_data["hourly"]["wind_speed_10m"])
temp = np.array(json_data["hourly"]["temperature_2m"])
humidity = np.array(json_data["hourly"]["relative_humidity_2m"])
precipitation = np.array(json_data["hourly"]["precipitation"])
cloud_cover = np.array(json_data["hourly"]["cloud_cover"])

states = [

"Clear",

"Cloudy",

"Windy",

"Misty",

"Rainy",

]

#each row describes probalities of the next state 
P = np.array([

[0.1, 0.4, 0.3, 0.2, 0.0],
    [0.2, 0.2, 0.4, 0.1, 0.1],
    [0.3, 0.1, 0.1, 0.3, 0.2],
    [0.2, 0.1, 0.2, 0.2, 0.3],
    [0.1, 0.1, 0.1, 0.5, 0.2]

])

current = 0 

history = [current]


# Evolve the system
for t in range(len(wind)):

    wind_speed = wind[t]
    temp_data = temp[t]
    humidity_data = humidity[t]
    precipitation_data = precipitation[t]
    cloud_cover_data = cloud_cover[t]
   

    influence = np.clip(wind_speed / 50, 0, 1)
    temp_influence = np.clip(temp_data / 50, 0, 1)
    humidity_influence = np.clip(humidity_data / 50, 0, 1)
    precipitation_influence = np.clip(precipitation_data / 50, 0, 1)
    cloud_cover_influence = np.clip(cloud_cover_data / 50, 0, 1)

    #copy probs for the current state 

    probabilities = P[current].copy()

    probabilities[4] += (influence * 0.4 + temp_influence * 0.2 + humidity_influence * 0.1 + cloud_cover_influence * 0.04 + precipitation_influence * 0.06
                         
                         )

    probabilities /= probabilities.sum()

    # Select the next state
    current = np.random.choice(
        len(states),
        p=probabilities
    )

    history.append(current)

    simulation_results.append({
            "time": json_data["hourly"]["time"][t],
            "wind": float(wind[t]),
            "temperature": float(temp[t]),
            "humidity": float(humidity[t]),
            "precipitation": float(precipitation[t]),
            "cloud_cover": float(cloud_cover[t]),
            "state": states[current]
        })

with open("simulation.json", "w") as file:
    json.dump(simulation_results, file, indent=4)

print(f"Exported {len(simulation_results)} hours")


# Visualise the sequence of states
plt.figure(figsize=(12, 5))
plt.plot(history, alpha=0.7)
plt.yticks(range(len(states)), states)
plt.xlabel("Time")
plt.ylabel("System state")
plt.title("Probabilistic State Machine")
plt.grid(alpha=0.2)
plt.show()
