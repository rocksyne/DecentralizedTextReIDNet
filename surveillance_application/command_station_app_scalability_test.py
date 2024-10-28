# Master Node Improvements

import os
import sys
import time
import pickle
import io
from datetime import datetime

import zmq
import nltk
import torch
from textblob import TextBlob
from flask import Flask
import paho.mqtt.client as mqtt
from loguru import logger

sys.path.insert(0, os.path.abspath('../'))
from datasets.custom_dataloader import process_text_into_tokens
from config import sys_configuration
from utils.iotools import read_json, write_to_text_file

nltk.download('punkt')
app = Flask(__name__)
configs = sys_configuration()
confidence_scores = {}
selected_camera = None
camera_search_logs = {}
no_of_camera_nodes = 5


context = zmq.Context()
socket = context.socket(zmq.PUB)
address = f"tcp://{configs.command_station_address}:{configs.message_parsing_app_port}"
socket.bind(address)
time.sleep(1)



logger.remove()
logger.add(os.path.join("..", "data", "logs", "command_station.log"), format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}", level="INFO")
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}", level="INFO")
logger.add(sys.stderr, level="DEBUG", filter=lambda record: record["level"].name == "DEBUG")


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("camera/status", qos=2)
    client.subscribe("camera/confidence", qos=2)


def on_message(client, userdata, msg):
    global waiting_for_response
    topic = msg.topic
    payload = msg.payload.decode()
    if topic == "camera/confidence":
        handle_camera_confidence(payload)


def handle_camera_confidence(payload):
    global waiting_for_response
    global selected_camera
    camera_name, score = payload.split("#")
    confidence_scores[camera_name] = float(score)

    if len(confidence_scores) == no_of_camera_nodes:
        selected_camera = max(confidence_scores, key= confidence_scores.get)
        waiting_for_response = False
        logger.info("All scores received and global value has been selected")
    # print(len(confidence_scores), "this is confidence")


def correct_sentence(text):
    blob = TextBlob(text)
    corrected_text = ''

    for sentence in blob.sentences:
        corrected_sentence = sentence.string.capitalize()
        if not corrected_sentence.endswith('.'):
            corrected_sentence += '.'
        corrected_text += corrected_sentence + ' '

    return corrected_text.strip()


def serialize_tensor(tensor: torch.Tensor, method: str = "pickle") -> bytes:
    if method == "pickle":
        return pickle.dumps(tensor)
    elif method == "io_bytes":
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()
    else:
        raise NotImplementedError(f"{method} method is not implemented.")


def send_command(command, parameter=None):
    message = [command.encode()]
    if parameter is not None:
        if isinstance(parameter, torch.Tensor):
            parameter = serialize_tensor(parameter)
        elif isinstance(parameter, str):
            parameter = parameter.encode()
        elif isinstance(parameter, list):
            encoded_list = [item.encode() if isinstance(item, str) else serialize_tensor(item) for item in parameter]
            parameter = pickle.dumps(encoded_list)
        message.append(parameter)
    zmq_sender(message, socket)
    logger.info("Search command sent successfully")


def zmq_sender(message,socket):
    # context = zmq.Context()
    # socket = context.socket(zmq.PUB)
    # address = f"tcp://{configs.command_station_address}:{configs.message_parsing_app_port}"
    # socket.bind(address)
    # time.sleep(1)
    socket.send_multipart(message)
    # socket.close()
    # context.term()


if __name__ == '__main__':
    mqtt_client = mqtt.Client()
    mqtt_client.connect(configs.command_station_address, 1883, 60)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.loop_start()

    textual_description = "The woman is wearing a black t-shirt with pink writing. She is wearing khaki shorts and white shoes."
    gt_image_name = "1420_c6s3_085567_00.jpg"

    user_text = correct_sentence(textual_description)
    token_ids, orig_token_length = process_text_into_tokens(user_text)
    token_ids = token_ids.unsqueeze(0).long()
    orig_token_length = torch.tensor([orig_token_length]).unsqueeze(-1)
    tensor = torch.cat((token_ids, orig_token_length), dim=1)
    
    operation_start_time = time.perf_counter_ns()
    send_command('search_person', tensor)

    waiting_for_response = True
    confidence_scores = {}
    while waiting_for_response:
        time.sleep(0.01)
    logger.info("Person search completed")
    operation_end_time = time.perf_counter_ns()

    # .strftime('%H:%M:%S:%f')[:-3]
    operation_time_elapsed = operation_end_time- operation_start_time
    operation_time_elapsed = operation_time_elapsed/1_000_000_000
    operation_time_elapsed = f"{operation_time_elapsed:.9f}"

    # operation_start_time = operation_start_time.strftime('%H:%M:%S:%f')[:-3]
    # operation_end_time = operation_end_time.strftime('%H:%M:%S:%f')[:-3]

    data_2_write = f"{no_of_camera_nodes},{operation_time_elapsed}"
    write_to_text_file(os.path.join("..","data","logs","camera_scalability_test.txt"),data_2_write,"append")

    logger.info(f"Result returned {confidence_scores[selected_camera]} from {selected_camera} with confidence score {confidence_scores[selected_camera]}")
    logger.info(f"Total time elapsed: {operation_time_elapsed}")
    logger.info("*"*70)
    logger.info("\n")
    socket.close()
    context.term()

