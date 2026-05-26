# Air Quality Data

## Data Source
- **Source**: Central Pollution Control Board (CPCB), India
- **Dataset**: city_day.csv

## Expected Columns
- `Date`: Date of measurement (YYYY-MM-DD)
- `City`: Location/City name
- `PM2.5`: PM2.5 concentration (µg/m³)
- `PM10`: PM10 concentration (µg/m³)
- `NO2`: NO2 concentration (µg/m³)
- `CO`: CO concentration (mg/m³)
- `SO2`: SO2 concentration (µg/m³)
- `O3`: O3 concentration (µg/m³)
- `NH3`: NH3 concentration (µg/m³)

## Data Format
- CSV file with comma separation
- First row: column headers
- Date format: YYYY-MM-DD
- Missing values: Empty cells or "NaN"

## Usage
```python
from air_quality_forecasting.preprocessing.data_loader import DataLoader

loader = DataLoader('data/city_day.csv')
data = loader.load()
data = loader.parse_dates()
```
