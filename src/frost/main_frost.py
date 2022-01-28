# from fill_weather_by_proximity import assign_to_farmer_and_fill_by_proximity
from preprocess_weather import preprocess_weather
from source_weather_to_file import raw_frost_readings_to_file
from utils import WEATHER_TYPES
from weather_sources_to_file import download_frost_sources

if __name__ == '__main__':

	growth_season = 2019
	start_date = f'{growth_season}-03-01'
	end_date = f'{growth_season}-10-01'
	client_id = 'c114a6ef-9081-4c42-b41b-0b3344a08ac4'
	secret = '8756c739-6d4e-47ad-b893-28d80b218df3'

	# use WEATHER_TYPES .TEMPERATURE or .PRECIPITATION
	# precipitation = WEATHER_TYPES.PRECIPITATION

	# Get all the sources from FROST
	# print(download_frost_sources(client_id))

	# Download the weather readings for each of the sources of a given type
	# raw_frost_readings_to_file(start_date, end_date, growth_season, WEATHER_TYPES.PRECIPITATION, client_id)
	# raw_frost_readings_to_file(start_date, end_date, growth_season, WEATHER_TYPES.TEMPERATURE, client_id)
	# raw_frost_readings_to_file(start_date, end_date, growth_season, WEATHER_TYPES.DAYDEGREE0, client_id)
	# raw_frost_readings_to_file(start_date, end_date, growth_season, WEATHER_TYPES.DAYDEGREE5, client_id)

	# Todo: Check out: Solar/stråling, Soil temperature, State of ground, Duration of sunshine,

	# Process the readings
	# These files can be used for interpolation / filling in the blanks by
	# looking at the 2nd closest, etc.
	# preprocess_weather(start_date, end_date, WEATHER_TYPES.PRECIPITATION)
	# preprocess_weather(start_date, end_date, WEATHER_TYPES.TEMPERATURE)
	# preprocess_weather(start_date, end_date, WEATHER_TYPES.DAYDEGREE0)
	# preprocess_weather(start_date, end_date, WEATHER_TYPES.DAYDEGREE5)

	# assign_to_farmer_and_fill_by_proximity(start_date, end_date, precipitation)
