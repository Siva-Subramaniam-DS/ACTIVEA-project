import os
import csv
import json
import math
import time
import numpy as np
from datetime import datetime, timezone
import concurrent.futures

INPUT_FILE = r'endomondoHR.json\endomondoHR.json'
OUTPUT_FILE = r'endomondoHR.csv'
CHUNK_SIZE = 2500
MAX_WORKERS = min(8, os.cpu_count() or 4)

COLUMNS = [
    'workout_id',
    'user_id',
    'sport',
    'gender',
    'start_time',
    'start_timestamp',
    'end_timestamp',
    'duration_sec',
    'num_points',
    'heart_rate_avg',
    'heart_rate_min',
    'heart_rate_max',
    'heart_rate_std',
    'speed_avg_kmh',
    'speed_min_kmh',
    'speed_max_kmh',
    'altitude_avg_m',
    'altitude_min_m',
    'altitude_max_m',
    'elevation_gain_m',
    'distance_km',
    'start_latitude',
    'start_longitude',
    'end_latitude',
    'end_longitude',
    'url'
]

def summarize_workout(d):
    wid = d.get('id', '')
    user_id = d.get('userId', '')
    sport = d.get('sport', '')
    gender = d.get('gender', '')
    url = d.get('url', '')

    timestamps = d.get('timestamp')
    hrs = d.get('heart_rate')
    speeds = d.get('speed')
    altitudes = d.get('altitude')
    lats = d.get('latitude')
    lons = d.get('longitude')

    num_points = len(timestamps) if timestamps else 0

    if num_points > 0:
        start_ts = timestamps[0]
        end_ts = timestamps[-1]
        duration_sec = max(0, end_ts - start_ts)
        try:
            start_datetime = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            start_datetime = ''
    else:
        start_ts = end_ts = duration_sec = ''
        start_datetime = ''

    if hrs:
        arr_hr = np.asarray(hrs, dtype=np.float32)
        hr_avg = round(float(np.mean(arr_hr)), 2)
        hr_min = int(np.min(arr_hr))
        hr_max = int(np.max(arr_hr))
        hr_std = round(float(np.std(arr_hr)), 2)
    else:
        hr_avg = hr_min = hr_max = hr_std = ''

    if speeds:
        arr_spd = np.asarray(speeds, dtype=np.float32)
        speed_avg = round(float(np.mean(arr_spd)), 2)
        speed_min = round(float(np.min(arr_spd)), 2)
        speed_max = round(float(np.max(arr_spd)), 2)
    else:
        speed_avg = speed_min = speed_max = ''

    if altitudes:
        arr_alt = np.asarray(altitudes, dtype=np.float32)
        alt_avg = round(float(np.mean(arr_alt)), 2)
        alt_min = round(float(np.min(arr_alt)), 2)
        alt_max = round(float(np.max(arr_alt)), 2)
        diff_alt = np.diff(arr_alt)
        elev_gain = round(float(np.sum(diff_alt[diff_alt > 0])), 2)
    else:
        alt_avg = alt_min = alt_max = elev_gain = ''

    if lats and lons and len(lats) > 1 and len(lons) > 1:
        start_lat = round(lats[0], 6)
        start_lon = round(lons[0], 6)
        end_lat = round(lats[-1], 6)
        end_lon = round(lons[-1], 6)

        arr_lat = np.radians(np.asarray(lats, dtype=np.float64))
        arr_lon = np.radians(np.asarray(lons, dtype=np.float64))
        dlat = np.diff(arr_lat)
        dlon = np.diff(arr_lon)
        a = np.sin(dlat / 2.0)**2 + np.cos(arr_lat[:-1]) * np.cos(arr_lat[1:]) * np.sin(dlon / 2.0)**2
        c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
        dist_km = round(float(np.sum(6371.0 * c)), 2)
    else:
        start_lat = start_lon = end_lat = end_lon = dist_km = ''

    return [
        wid, user_id, sport, gender, start_datetime, start_ts, end_ts,
        duration_sec, num_points, hr_avg, hr_min, hr_max, hr_std,
        speed_avg, speed_min, speed_max, alt_avg, alt_min, alt_max,
        elev_gain, dist_km, start_lat, start_lon, end_lat, end_lon, url
    ]

def process_line_chunk(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line.replace("'", '"'))
        except Exception:
            import ast
            try:
                d = ast.literal_eval(line)
            except Exception:
                continue
        rows.append(summarize_workout(d))
    return rows

def generate_chunks(filepath, chunk_size):
    current_chunk = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            current_chunk.append(line)
            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []
    if current_chunk:
        yield current_chunk

def main():
    print(f'Starting conversion: {INPUT_FILE} -> {OUTPUT_FILE}')
    print(f'Using {MAX_WORKERS} worker processes with chunk size {CHUNK_SIZE}...')
    t0 = time.time()
    total_records = 0

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(COLUMNS)

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Maintain a small sliding window of futures to avoid loading too many chunks in memory
            active_futures = []
            chunk_iterator = generate_chunks(INPUT_FILE, CHUNK_SIZE)

            # Preload a few chunks
            for _ in range(MAX_WORKERS * 2):
                try:
                    chunk = next(chunk_iterator)
                    active_futures.append(executor.submit(process_line_chunk, chunk))
                except StopIteration:
                    break

            last_report = 0
            while active_futures:
                # Get first completed or pop left
                future = active_futures.pop(0)
                rows = future.result()
                writer.writerows(rows)
                total_records += len(rows)

                # Feed next chunk
                try:
                    next_chunk = next(chunk_iterator)
                    active_futures.append(executor.submit(process_line_chunk, next_chunk))
                except StopIteration:
                    pass

                # Progress log every ~25,000 records
                if total_records - last_report >= 25000:
                    elapsed = time.time() - t0
                    speed = total_records / elapsed if elapsed > 0 else 0
                    print(f'Processed {total_records:,} records ({total_records/253020*100:.1f}%) in {elapsed:.1f}s - {speed:.0f} rec/s')
                    last_report = total_records

    total_time = time.time() - t0
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print('========================================================')
    print(f'CONVERSION COMPLETED SUCCESSFULLY!')
    print(f'Total records written: {total_records:,}')
    print(f'Total time taken:      {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
    print(f'Output CSV file:       {OUTPUT_FILE} ({file_size_mb:.2f} MB)')
    print('========================================================')

if __name__ == '__main__':
    main()

