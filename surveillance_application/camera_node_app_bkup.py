# Imports and Initial Setup
import cv2
import os
import sys
import pickle
import threading
import queue
import traceback
import time
import numpy as np
import socket
from PIL import Image, ImageDraw, ImageFont
from typing import List, Union, Optional
from datetime import datetime, timedelta

import zmq
import torch
from imagezmq import ImageSender
import paho.mqtt.client as mqtt

sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from model.unity import Unity
from evaluation.evaluations import calculate_similarity
from utils.video_stream import VideoFileStream
from utils.camera_stream import WebcamVideoStream
#from nano.nano_webcam import WebcamVideoStream

# +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++[Utility Functions]++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++

def on_connect(client, userdata, flags, rc):
    """Handles connection to MQTT broker."""
    print(f"Connected with result code {rc}")

def setup_mqtt_client(configs):
    """Sets up the MQTT client."""
    client = mqtt.Client()
    client.on_connect = on_connect
    client.connect(configs.command_station_address, 1883, 60)
    return client

def setup_subscriber(address: str) -> tuple:
    """Sets up a ZMQ subscriber."""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(address)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages
    return context, socket

def thread_target_wrapper(func):
    """Wrapper for thread target functions to handle exceptions."""
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in the thread: {e}")
            traceback.print_exc()
    return wrapped

def receive_message(socket, data_queue: queue.Queue):
    """Receives messages from ZMQ socket and puts them into the queue."""
    while True:
        try:
            if socket.poll(500):  # Wait for up to 500ms for a message
                message_parts = socket.recv_multipart()

                if len(message_parts) == 2:
                    command, parameter = message_parts
                    command = command.decode('utf-8')
                    parameter = pickle.loads(parameter)

                    if isinstance(parameter, list):
                        parameter = [p.decode('utf-8') for p in parameter]

                    data_queue.put((command, parameter))

        except zmq.Again:
            time.sleep(0.01)
            continue  # No message received
        except (pickle.UnpicklingError, zmq.ZMQError) as e:
            print(f"An error occurred in receive_message(): {e}")
        except Exception as e:
            print(f"An unexpected error occurred in receive_message(): {e}")
            traceback.print_exc()
            break

# +++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++[Confidence Score Reporting]++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++

def report_confidence_score(client, camera_name, highest_score):
    """Publishes the highest confidence score to the MQTT broker."""
    client.publish(f"camera/confidence", f"{camera_name}_{highest_score}")

def confidence_reporting_thread(client, camera_name, confidence_queue):
    """Reports confidence scores at regular intervals."""
    while True:
        try:
            highest_score = 0
            while not confidence_queue.empty():
                score = confidence_queue.get()
                if score > highest_score:
                    highest_score = score
            if highest_score > 0:
                report_confidence_score(client, camera_name, highest_score)
            time.sleep(1)  # Report every second
        except Exception as e:
            print(f"Error in confidence_reporting_thread() thread: {e}")

# +++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++[Video Processing]+++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++

def process_bounding_boxes(bbox, similarity, threshold):
    """Filters bounding boxes based on similarity scores."""
    filtered_boxes = [bbox[i].tolist() for i, sim in enumerate(similarity) if sim >= threshold]
    filtered_scores = [sim for sim in similarity if sim >= threshold]
    return filtered_boxes, filtered_scores

def draw_bounding_boxes(image: Image.Image, bounding_boxes: List[List[int]], confidence_scores: List[float], 
                        camera_name: str, top_1: bool = False, fonts: List[ImageFont.ImageFont] = None, 
                        color: Optional[Union[str, tuple]] = None) -> Image.Image:
    """Draws bounding boxes and confidence scores on the image."""
    if top_1:
        bounding_boxes = bounding_boxes[:1]
        confidence_scores = confidence_scores[:1]
    
    draw = ImageDraw.Draw(image)
    confidence_font, camera_name_font, other_stats_font = fonts
    camera_text = f"Person found in {camera_name}"
    draw.rectangle([0, 5, 512, 55], fill=color)
    left, top, right, bottom = camera_name_font.getbbox(camera_text)
    text_width = right - left
    text_height = bottom - top
    text_x = (512 - text_width) // 2
    text_y = (55 - text_height) // 2
    draw.text((text_x, text_y), camera_text, font=camera_name_font, fill=(255, 255, 255)) 

    # timestamp when the processing was completed
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    timestamp = f"Processed: {timestamp}"
    left, top, right, bottom = other_stats_font.getbbox(timestamp)
    text_width = right - left
    text_height = bottom - top
    x = 512 - text_width - 298 # some small padding
    y = 512 - text_height - 18 # push it to the top
    draw.text((x, y), timestamp, font=other_stats_font, fill=(255, 255, 255))  # White text

    for bbox, score in zip(bounding_boxes, confidence_scores):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"Conf: {float(score):.2f}"  # Ensure score is a float
        text_bbox = draw.textbbox((x1, y1), text, font=confidence_font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        draw.rectangle([x1, y1 - text_height, x1 + text_width + 10, y1], fill=color)
        draw.text((x1 + 5, y1 - text_height), text, fill="white", font=confidence_font)

    return image

def process_video(data_queue, camera_stream, confidence_queue, model, configs, camera_name, top_1):
    """Main video processing function."""
    previous_token_ids = torch.zeros((1, 100)).to(configs.device)  # Dummy previous tokens
    textual_embedding = None
    confidence_font = ImageFont.truetype("arial.ttf", 15) 
    camera_name_font = ImageFont.truetype("arial.ttf", 30)
    other_stats_font = ImageFont.truetype("arial.ttf", 19) 
    fonts = [confidence_font, camera_name_font, other_stats_font]
    color = tuple(np.random.randint(0, 255, size=3))
    operation_command = None
    stream_retrieved_video = False
    reID_confidence_threshold = configs.reID_confidence_threshold
    global status_code

    while True:
        try:
            # Check for new commands
            while not data_queue.empty():
                command, parameter = data_queue.get_nowait()
                
                print("The command is: ", command)
                if command == "set_sensitivity":
                    reID_confidence_threshold = float(parameter)
            
                elif command == 'search_person':
                    operation_command = 'search_person'
                    operation_parameter = parameter
                    stream_retrieved_video = False
                    current_token_ids = operation_parameter[:, :100].to(configs.device)
                    current_orig_token_length = operation_parameter[:, 100:].squeeze(-1).to(configs.device)

                    if not torch.equal(previous_token_ids, current_token_ids):
                        with torch.no_grad():
                            textual_embedding = model.target_persons_text_embeddings(current_token_ids, current_orig_token_length)
                        previous_token_ids = current_token_ids.clone()
                
                elif command == 'stop_person_search':
                    operation_command = 'stop_person_search'
                    reID_confidence_threshold = configs.reID_confidence_threshold
                    stream_retrieved_video = False
                
                elif command == 'stop_video_stream':
                    operation_command = 'stop_video_stream'
                    stream_retrieved_video = False

                elif command == 'stream_retrieved_video':
                    operation_parameter = parameter
                    if camera_name in operation_parameter:
                        stream_retrieved_video = True
                
                elif command == 'stream_live_video':
                    operation_command = 'stream_live_video'
                    operation_parameter = parameter
                    if camera_name in operation_parameter:
                        stream_retrieved_video = True

                else:
                    ...  # TODO: clean up later
                
                print(f"Received command - {command} | Current operation command is {operation_command}")

            if operation_command is None:
                if status_code != "CODE01": status_code = "CODE01"
                continue

            elif (operation_command == 'stop_person_search') or (operation_command == 'stop_video_stream'):
                operation_parameter = parameter
                if camera_name in operation_parameter:
                    operation_command = None
                    status_code = "CODE01"
            
            elif operation_command == 'stream_live_video':
                ret, frame, _, _ = camera_stream.read()
                if not camera_stream.grabbed:
                    status_code = "CODE404"
                    print("Camera stream broken")
                    continue

                if stream_retrieved_video:
                    status_code = "CODE03"
                    _, original, original_img_as_tensor, preprocessed = camera_stream.read()
                    # timestamp when sending
                    draw = ImageDraw.Draw(original)
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    timestamp = f"Transmitted: {timestamp}"
                    left, top, right, bottom = other_stats_font.getbbox(timestamp)
                    text_width = right - left
                    text_height = bottom - top
                    x = 512 - text_width - 1 # some small padding
                    y = 512 - text_height - 46 # push it to the top
                    draw.text((x, y), timestamp, font=other_stats_font, fill=(255, 255, 255))  # White text

                    
             
                    camera_output = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
                    sender.send_image(camera_name, camera_output)

                    # for debugging purposes TODO: clean it
                    # cv2.imshow('Person Search', camera_output)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

            elif operation_command == 'search_person':
                if not camera_stream.grabbed:
                    status_code = "CODE404"
                    print("Camera stream broken")
                    continue
                
                _, original, original_img_as_tensor, preprocessed = camera_stream.read()
                draw_on_original = ImageDraw.Draw(original)

                # timestamp the image for when captured
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                timestamp = f"Captured:   {timestamp}"
                left, top, right, bottom = other_stats_font.getbbox(timestamp)
                text_width = right - left
                text_height = bottom - top
                x = 512 - text_width - 300 # some small padding
                y = 512 - text_height - 40 # push it to the top
                draw_on_original.text((x, y), timestamp, font=other_stats_font, fill=(255, 255, 255))  # White text


                camera_output = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
                
                if textual_embedding is not None:
                    with torch.no_grad():
                        persons_embeddings_out = model.persons_embeddings(original_img_as_tensor, preprocessed.to(configs.device))

                    if persons_embeddings_out is not None:
                        image_embedding, bbox = persons_embeddings_out
                        similarity = calculate_similarity(image_embedding, textual_embedding).detach().cpu().numpy()

                        filtered_boxes, filtered_scores = process_bounding_boxes(bbox, similarity, reID_confidence_threshold)

                        if filtered_boxes:
                            sorted_indices = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)
                            sorted_boxes = [filtered_boxes[i] for i in sorted_indices]
                            sorted_scores = [filtered_scores[i] for i in sorted_indices]
                            confidence_queue.put((sorted_scores[0].tolist()[0]))

                            if stream_retrieved_video:
                                original_img = draw_bounding_boxes(original, sorted_boxes, sorted_scores, camera_name, top_1=top_1, fonts=fonts, color=color)

                                # timestamp when sending
                                draw = ImageDraw.Draw(original_img)
                                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                timestamp = f"Transmitted: {timestamp}"
                                left, top, right, bottom = other_stats_font.getbbox(timestamp)
                                text_width = right - left
                                text_height = bottom - top
                                x = 512 - text_width - 1 # some small padding
                                y = 512 - text_height - 46 # push it to the top
                                draw.text((x, y), timestamp, font=other_stats_font, fill=(255, 255, 255))  # White text

                                # send to command station
                                camera_output = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
                                sender.send_image(camera_name, camera_output)

                if status_code != "CODE02": status_code = "CODE02"

                # for debugging purposes TODO: clean it
                # cv2.imshow('Person Search', camera_output)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        except Exception as e:
            print(f"Error in process_video() thread: {e}")
            traceback.print_exc()
            break




# +++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++[Main Program]++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    camera_name = socket.gethostname()
    camera_ip_address = socket.gethostbyname(camera_name)
    status_code = "CODE01"
    top_1 = True
    
    configs = sys_configuration()
    model = Unity(configs).to(configs.device)
    model.eval()

    data_queue = queue.Queue()
    confidence_queue = queue.Queue()
    context, subscriber_socket = setup_subscriber(f"tcp://{configs.command_station_address}:{configs.message_parsing_app_port}")
    #camera_stream = VideoFileStream(configs=configs, video_path=f"../data/sample_videos/{camera_name}.mp4", loop=True).start()
    camera_stream = WebcamVideoStream(configs=configs) # read the usb camera
    sender = ImageSender(connect_to=f"tcp://{configs.command_station_address}:{configs.video_parsing_app_port}")

    mqtt_client = setup_mqtt_client(configs)

    recv_thread = threading.Thread(target=thread_target_wrapper(receive_message), args=(subscriber_socket, data_queue))
    recv_thread.daemon = True
    recv_thread.start()

    video_thread = threading.Thread(target=process_video, args=(data_queue, camera_stream, confidence_queue, model, configs, camera_name, top_1))
    video_thread.daemon = True
    video_thread.start()

    reporting_thread = threading.Thread(target=confidence_reporting_thread, args=(mqtt_client, camera_name, confidence_queue))
    reporting_thread.daemon = True
    reporting_thread.start()

    try:
        while True:
            mqtt_client.publish(f"camera/status", f"{camera_name}_{camera_ip_address}_{status_code}")
            time.sleep(1)  # Keep the main thread alive

    except KeyboardInterrupt:
        print("Main program interrupted")
    finally:
        subscriber_socket.close()
        context.term()
        camera_stream.stop()
        mqtt_client.disconnect()
        cv2.destroyAllWindows()
