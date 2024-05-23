from matplotlib import pyplot as plt
import json
import pandas as pd

# Data from danish weather stations as json objects by station
weather_data_src = "C:/Users/au662726/OneDrive - Aarhus universitet/AU - 2024/Data/2023-09-30.txt"
readings = []
with open(weather_data_src, "r") as file:
    for line in file:
        readings.append(json.loads(line))
# exmple data
# {"geometry":{"coordinates":[10.3304,55.4748],"type":"Point"},"properties":{"calculatedAt":"2023-10-02T15:03:17.087000+00:00","created":"2023-12-22T09:30:24.607055+00:00","from":"2023-09-30T23:00:00+00:00","parameterId":"max_wind_speed_10min","qcStatus":"manual","stationId":"06120","timeResolution":"hour","to":"2023-10-01T00:00:00+00:00","validity":true,"value":2.6},"type":"Feature","id":"000683b0-a7df-4858-210c-ee365bcec3db"}
# {"geometry":{"coordinates":[15.0953,55.0557],"type":"Point"},"properties":{"calculatedAt":"2023-10-02T15:03:17.474000+00:00","created":"2023-12-22T09:30:25.885052+00:00","from":"2023-09-30T23:00:00+00:00","parameterId":"max_wind_speed_3sec","qcStatus":"manual","stationId":"06197","timeResolution":"hour","to":"2023-10-01T00:00:00+00:00","validity":true,"value":10.7},"type":"Feature","id":"00098368-2df6-5a18-5f89-3852acf7c9c1"}
# {"geometry":{"coordinates":[12.4121,55.8764],"type":"Point"},"properties":{"calculatedAt":"2023-10-01T00:05:03.136198+00:00","created":"2023-12-22T09:30:39.533250+00:00","from":"2023-09-30T23:00:00+00:00","parameterId":"temp_grass","qcStatus":"none","stationId":"06188","timeResolution":"hour","to":"2023-10-01T00:00:00+00:00","validity":true,"value":11.5},"type":"Feature","id":"0039d1d4-b68c-3e0e-b25a-8a40f7ee4051"}
# {"geometry":{"coordinates":[-50.7172,67.0133],"type":"Point"},"properties":{"calculatedAt":"2023-10-01T00:51:12.548090+00:00","created":"2023-12-22T09:30:34.229477+00:00","from":"2023-09-30T23:00:00+00:00","parameterId":"mean_wind_dir","qcStatus":"manual","stationId":"04231","timeResolution":"hour","to":"2023-10-01T00:00:00+00:00","validity":true,"value":55},"type":"Feature","id":"006bc77e-46bb-773f-14c7-472f35036ba5"}
# {"geometry":{"coordinates":[-18.6681,76.7694],"type":"Point"},"properties":{"calculatedAt":"2023-10-01T00:51:12.548090+00:00","created":"2023-12-22T09:30:36.697102+00:00","from":"2023-09-30T23:00:00+00:00","parameterId":"mean_radiation","qcStatus":"manual","stationId":"04320","timeResolution":"hour","to":"2023-10-01T00:00:00+00:00","validity":true,"value":0},"type":"Feature","id":"007b6c5d-4fca-07d2-3dc3-398a463b9739"}
# {"geometry":{"coordinates":[10.5135,56.0955],"type":"Point"},"properties":{"calculatedAt":"2023-10-02T15:03:15.013000+00:00","created":"2023-12-22T09:30:20.452260+00:00","from":"2023-09-30T23:00:00+00:00","parameterId":"min_temp","qcStatus":"manual","stationId":"06073","timeResolution":"hour","to":"2023-10-01T00:00:00+00:00","validity":true,"value":13.3},"type":"Feature","id":"00b779b4-27ef-550a-06b1-03b911ea09b3"}
# {"geometry":{"coordinates":[10.4398,55.3088],"type":"Point"},"properties":{"calculatedAt":"2023-10-04T08:16:51.913000+00:00","created":"2023-12-22T09:30:12.701533+00:00","from":"2023-09-30T23:00:00+00:00","parameterId":"acc_precip","qcStatus":"manual","stationId":"06126","timeResolution":"hour","to":"2023-10-01T00:00:00+00:00","validity":true,"value":0},"type":"Feature","id":"00b9e46c-4b25-d73f-b88d-9b505a57ea9f"}

# Create a pandas dataframe from the json objects, extracting the properties, and including coordinates, type and id
# this should be in long format, meaning a column for parameterID and a column for value
df = pd.DataFrame(readings)
df["parameterId"] = df["properties"].apply(lambda x: x["parameterId"])
df["value"] = df["properties"].apply(lambda x: x["value"])
df["stationId"] = df["properties"].apply(lambda x: x["stationId"])
df["timeResolution"] = df["properties"].apply(lambda x: x["timeResolution"])
df["from"] = df["properties"].apply(lambda x: x["from"])
df["to"] = df["properties"].apply(lambda x: x["to"])
df["validity"] = df["properties"].apply(lambda x: x["validity"])
df["qcStatus"] = df["properties"].apply(lambda x: x["qcStatus"])
df["calculatedAt"] = df["properties"].apply(lambda x: x["calculatedAt"])
df["created"] = df["properties"].apply(lambda x: x["created"])
df["geometry"] = df["geometry"].apply(lambda x: x["coordinates"])
df["geometry_type"] = df["geometry"].apply(lambda x: x[0])
df.head()


df["from"] = pd.to_datetime(df["from"], format="mixed", utc=True)
df["to"] = pd.to_datetime(df["to"], format="mixed", utc=True)
df["calculatedAt"] = pd.to_datetime(df["calculatedAt"], format="mixed", utc=True)
df["created"] = pd.to_datetime(df["created"], format="mixed", utc=True)
df["geometry"] = df["geometry"].apply(lambda x: (x[0], x[1]))
df["value"] = pd.to_numeric(df["value"])
df["stationId"] = pd.to_numeric(df["stationId"])
df["geometry_type"] = pd.to_numeric(df["geometry_type"])
df["validity"] = df["validity"].astype(bool)
df["qcStatus"] = df["qcStatus"].astype("category")
df["parameterId"] = df["parameterId"].astype("category")
df["timeResolution"] = df["timeResolution"].astype("category")
df["geometry"] = df["geometry"].astype("category")
df["geometry_type"] = df["geometry_type"].astype("category")

params = df["parameterId"].unique().tolist()
params
# plot the max_wind_speed_3sec parameterID values (y) by calculatedAt (x) and color by station
df_max_wind_speed_3sec = df[df["parameterId"] == "max_wind_speed_3sec"]
df_max_wind_speed_3sec = df_max_wind_speed_3sec.sort_values("calculatedAt")
df_max_wind_speed_3sec = df_max_wind_speed_3sec.reset_index(drop=True)
plt.scatter(df_max_wind_speed_3sec["calculatedAt"], df_max_wind_speed_3sec["value"], c=df_max_wind_speed_3sec["stationId"])
plt.show()

