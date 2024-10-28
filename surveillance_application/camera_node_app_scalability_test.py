# Camera Node Improvements

import os
import sys
import pickle
import traceback
import time
from tqdm import tqdm

import zmq
import torch
import paho.mqtt.client as mqtt
from torchvision import transforms
from loguru import logger

sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from model.unity import Unity
from evaluation.evaluations import calculate_similarity
from utils.iotools import read_image
from utils.mhpv2_dataset_utils import CustomMHPv2ImageResizer
import argparse


def list_images_in_directory(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    files = os.listdir(directory)
    return [os.path.join(directory, file) for file in files if file.lower().endswith(image_extensions)]


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")


def setup_mqtt_client(configs):
    client = mqtt.Client()
    client.on_connect = on_connect
    try:
        client.connect(configs.command_station_address, 1883, 60)
        client.loop_start()
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        return None
    return client


def setup_subscriber(address: str):
    try:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(address)
        socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages
        return context, socket
    except Exception as e:
        print(f"Problem in setup_subscriber: {e}")
        return None, None


def report_confidence_score(client, camera_name, image_name, highest_score):
    client.publish(f"camera/confidence", f"{camera_name}#{image_name}#{highest_score}", qos=2)
    print("Sent confidence score")


def read_and_process_image(path):
    img = read_image(img_path=path, mode='RGB')
    processed_img = rgb_img_resizer(img)
    original_img = processed_img.copy()
    processed_img = MHPv2_transform(processed_img)
    processed_img = processed_img.unsqueeze(0)
    original_img_as_tensor = transforms.ToTensor()(original_img)
    return original_img, original_img_as_tensor, processed_img


def search_person(model, configs, operation_parameter, camera_name, image_embedding_list, original_image_list):
    logger.info("Searching person started")
    image_list = list_images_in_directory(f"../data/gallery/{camera_name}")
    current_token_ids = operation_parameter[:, :100].to(configs.device)
    current_orig_token_length = operation_parameter[:, 100:].squeeze(-1).to(configs.device)

    with torch.no_grad():
        logger.info("Text feature extraction started")
        timer_started = time.time()
        textual_embedding = model.target_persons_text_embeddings(current_token_ids, current_orig_token_length)
        logger.info(f"Text feature extraction finished in {time.time()-timer_started} seconds")

    logger.info("Similarity calculation started")
    timer_started = time.time()
    ranking_result = []
    for img_embedding, image_path in zip(image_embedding_list, original_image_list):
        similarity = calculate_similarity(img_embedding, textual_embedding).numpy()[0][0]
        ranking_result.append([image_path, similarity])
    sorted_paths_and_scores = sorted(ranking_result, key=lambda x: x[1], reverse=True)
    logger.info(f"Similarity calculation finished in {time.time()-timer_started} seconds")
    return sorted_paths_and_scores[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera name argument')
    parser.add_argument('--camera_name', type=str, required=True, help='Name of the camera')
    args = parser.parse_args()

    camera_name = args.camera_name
    logger.remove()
    logger.add(os.path.join("..", "data", "logs", f"{camera_name}.log"), format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}", level="INFO")
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}", level="INFO")
    logger.add(sys.stderr, level="DEBUG", filter=lambda record: record["level"].name == "DEBUG")

    print("Camera name is: %s" % camera_name)
    configs = sys_configuration()
    model = Unity(configs).to(configs.device)
    model.eval()

    MHPv2_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=configs.MHPV2_means, std=configs.MHPV2_stds)])
    rgb_img_resizer = CustomMHPv2ImageResizer(dimensions=configs.MHPV2_image_size, image_type='RGB')

    context, subscriber_socket = setup_subscriber(f"tcp://{configs.command_station_address}:{configs.message_parsing_app_port}")
    if context is None or subscriber_socket is None:
        print("Failed to setup subscriber. Exiting.")
        sys.exit(1)

    mqtt_client = setup_mqtt_client(configs)
    if mqtt_client is None:
        print("Failed to setup MQTT client. Exiting.")
        sys.exit(1)

    image_embedding_list = []
    original_image_list = []

    while True:
        try:
            if subscriber_socket.poll(500):
                logger.info("Message received from master node")
                message_parts = subscriber_socket.recv_multipart()
                if len(message_parts) == 2:
                    command, parameter = message_parts
                    command = command.decode('utf-8')
                    parameter = pickle.loads(parameter)

                    if isinstance(parameter, list):
                        parameter = [p.decode('utf-8') for p in parameter]

                    logger.info("Searching person started...loading images and extracting embeddings")
                    image_list = list_images_in_directory(f"../data/gallery/{camera_name}")
                    for image_path in tqdm(image_list):
                        original_img, original_img_as_tensor, processed_img = read_and_process_image(image_path)

                        with torch.no_grad():
                            persons_embeddings_out = model.persons_embeddings(original_img_as_tensor.to(configs.device), processed_img.to(configs.device))

                        if persons_embeddings_out is not None:
                            image_embedding, bbox = persons_embeddings_out
                            image_embedding_list.append(image_embedding)
                            original_image_list.append(image_path)
                    logger.info("Loading images and extracting embeddings completed successfully")

                    search_result = search_person(model, configs, parameter, camera_name, image_embedding_list, original_image_list)

                    print(search_result)
                    image_name = search_result[0]
                    image_name = image_name.split(os.sep)[-1]
                    report_confidence_score(mqtt_client, camera_name, image_name, search_result[1])
                    logger.info("Confidence score sent back to master node")
                    print("*" * 50)
                    print("\n\n")

        except zmq.Again:
            time.sleep(0.01)
            continue
        except (pickle.UnpicklingError, zmq.ZMQError) as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            break
