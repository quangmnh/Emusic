import time
import random
import collections
import json


start_record_time = prev_time = time.time()

duration = 5

frame_rate = 10

results = []

while (int(time.time() - start_record_time) < duration):
    time_elapsed = time.time() - prev_time
    if time_elapsed > 1./frame_rate:
        prev_time = time.time()
        result = random.choice(
            ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'])
        results.append(result)


results_count = collections.Counter(results)
results_json = json.dumps(results_count)
print(results_json)
