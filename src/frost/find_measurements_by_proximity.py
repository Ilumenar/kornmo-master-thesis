import os
import pandas as pd
from tqdm import tqdm
from utils import distance


weather_data_path = "../../../kornmo-data-files/raw-data/weather-data/"


def assign_to_farmer_and_fill_by_proximity(start_date, end_date, weather_feature):
	cleaned_weather_data = pd.read_csv(os.path.join(weather_data_path, f"cleaned/{weather_feature}/{weather_feature}_cleaned_{start_date}_to_{end_date}.csv"))
	stations_df = pd.read_csv(os.path.join(weather_data_path, f"frost_weather_sources.csv"))
	stations_with_weather_df = pd.merge(stations_df, cleaned_weather_data, left_on='id', right_on='station_id')

	columns_to_keep = ['id', 'lng', 'lat'] + list(filter(lambda x: x.startswith('day_'), stations_with_weather_df.columns.tolist()))
	stations_df = stations_with_weather_df[columns_to_keep]

	farmers = pd.read_csv(f"../../../kornmo-data-files/raw-data/farm-information/all-farmers-with-location.csv")[['orgnr', 'longitude', 'latitude', 'elevation']]
	farmers_with_weather = []

	number_of_farmers = farmers.shape[0]
	print("---- Assign weather to farmers by proximity ----")
	print(f"Number of farmers: {number_of_farmers}")

	p_bar = tqdm(farmers.iterrows(), total=number_of_farmers)
	for index, farmer in p_bar:
		p_bar.set_description(f"Calculating for index {index}")
		farmer_coordinates = (farmer.latitude, farmer.longitude)
		stations_with_distance_df = stations_df.copy()
		stations_with_distance_df['ws_distance'] = stations_with_distance_df.apply(
			lambda ws: distance(farmer_coordinates, (ws.lat, ws.lng)), axis=1)
		# Find the closest station, order by distance and ffill, keeping the top-most row.
		farmer_weather_df = stations_with_distance_df.sort_values(by=['ws_distance']).head(1)
		farmer_weather_df['orgnr'] = farmer.orgnr

		if not farmer_weather_df.dropna().empty:
			farmer_weather_df = farmer_weather_df.drop(['lng', 'lat'], axis=1)
			closest_station_id = farmer_weather_df.iloc[0].id
			missing_measurements_on_closest_station = \
			stations_df.loc[stations_df['id'] == closest_station_id].isnull().sum(axis=1).tolist()[0]
			farmer_weather_df['missing_measurements_from_closest_ws'] = missing_measurements_on_closest_station
			farmer_weather_df = farmer_weather_df.rename(columns={'id': 'ws_id', 'lng': 'ws_lng', 'lat': 'ws_lat'})
			farmers_with_weather.append(farmer_weather_df)
		else:
			print(f"Farmer location or weather missing - {farmer.orgnr}: {(farmer.lat, farmer.lng)}")

	farmers_with_weather_df = pd.concat(farmers_with_weather, ignore_index=True)
	farmers_with_weather_df.to_csv(os.path.join(weather_data_path, f"by_proximity/{weather_feature}/{weather_feature}_by_proximity_{start_date}_to_{start_date}.csv"), index=False)
	print("\nDone.")
