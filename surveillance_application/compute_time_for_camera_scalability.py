import time

camera_data = {
    "camera1": 20,
    # "camera2": 16,
    # "camera3": 25,
    # "camera4": 18,
    # "camera5": 22,
    # "camera6": 19,
    # "camera7": 21,
    # "camera8": 23,
    # "camera9": 17,
    # "camera10": 24,
    # "camera11": 26,
    # "camera12": 20,
    # "camera13": 15,
    # "camera14": 27,
    # "camera15": 18,
    # "camera16": 22,
    # "camera17": 19,
    # "camera18": 23,
    # "camera19": 16,
    # "camera20": 21,
    # "camera21": 25,
    # "camera22": 17,
    # "camera23": 20,
    # "camera24": 18,
    # "camera25": 24,
    # "camera26": 19,
    # "camera27": 26,
    # "camera28": 15,
    # "camera29": 22,
    # "camera30": 27
}

# Start timing
start_time = time.time()

# Find the key with the maximum value
max_key = max(camera_data, key=camera_data.get)
max_value = camera_data[max_key]

# End timing
end_time = time.time()

# Calculate the processing time
processing_time = end_time - start_time

print(f"The camera with the maximum value is {max_key} with a value of {max_value}.")
print(f"Processing time: {processing_time} seconds")
