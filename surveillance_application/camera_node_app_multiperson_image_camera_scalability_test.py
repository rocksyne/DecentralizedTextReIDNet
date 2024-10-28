# Scalability test for number of galleryies

import os
import sys
import pickle
import traceback
import time
from tqdm import tqdm
import datetime

import zmq
import torch
import paho.mqtt.client as mqtt
from torchvision import transforms
from loguru import logger

sys.path.insert(0, os.path.abspath('../'))
from config import sys_configuration
from utils.miscellaneous_utils import set_seed
from model.unity import Unity
from evaluation.evaluations import calculate_similarity
from utils.iotools import read_image, write_to_text_file
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


def report_confidence_score(client, camera_name, highest_score):
    client.publish(f"camera/confidence", f"{camera_name}#{highest_score}", qos=2)
    print("Sent confidence score")


def read_and_process_image(path):
    img = read_image(img_path=path, mode='RGB')
    processed_img = rgb_img_resizer(img)
    original_img = processed_img.copy()
    processed_img = MHPv2_transform(processed_img)
    processed_img = processed_img.unsqueeze(0)
    original_img_as_tensor = transforms.ToTensor()(original_img)
    return original_img, original_img_as_tensor, processed_img


def search_person(model, configs, operation_parameter, camera_name, image_embedding, original_image_list):
    
    logger.info("Searching person started")
    current_token_ids = operation_parameter[:, :100].to(configs.device)
    current_orig_token_length = operation_parameter[:, 100:].squeeze(-1).to(configs.device)

    text_processing_timer_started = time.time()
    with torch.no_grad():
        logger.info("Text feature extraction started")
        textual_embedding = model.target_persons_text_embeddings(current_token_ids, current_orig_token_length)
        text_processing_elapsed_time = time.time() - text_processing_timer_started
        logger.info(f"Text feature extraction finished in {text_processing_elapsed_time} seconds")

    logger.info("Similarity calculation started")
    timer_started = time.time()
    
    similarity = calculate_similarity(image_embedding, textual_embedding).numpy()[0][0]


    print(similarity)
    # indices_sorted = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)
    # ranked_scores = [similarity[i].tolist()[0] for i in indices_sorted]
    similarity_time_elapsed = time.time()-timer_started
    logger.info(f"Similarity calculation finished in {similarity_time_elapsed } seconds")
    return [1], text_processing_elapsed_time, similarity_time_elapsed


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
    set_seed(configs.seed)
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
    number_of_persons = 1 # number of images

    test_count = 0

    while True:
        try:
            if subscriber_socket.poll(500):
                logger.info("Message received from master node")
                message_recieved_time = time.perf_counter_ns()
                image_processing_times = []
                img_read_elapsed_times = []
                message_parts = subscriber_socket.recv_multipart()
                if len(message_parts) == 2:
                    command, parameter = message_parts
                    command = command.decode('utf-8')
                    parameter = pickle.loads(parameter)

                    if isinstance(parameter, list):
                        parameter = [p.decode('utf-8') for p in parameter]

                    logger.info("Searching person started...loading images and extracting embeddings")

                    image_path = f"../data/gallery/{camera_name}/merged_images_10.jpg"
                    
                    img_processing_start = datetime.datetime.now()

                    img_read_time = datetime.datetime.now()
                    original_img, original_img_as_tensor, processed_img = read_and_process_image(image_path)
                    img_read_elapsed_time = datetime.datetime.now() - img_read_time
                    img_read_elapsed_time = img_read_elapsed_time.total_seconds()
                    img_read_elapsed_times.append(img_read_elapsed_time)

                    logger.info("Starting person search")
                    current_token_ids = parameter[:, :100].to(configs.device)
                    current_orig_token_length = parameter[:, 100:].squeeze(-1).to(configs.device)

                    text_processing_timer_started = time.time()
                    logger.info("Text feature extraction started")
                    with torch.no_grad():
                        textual_embedding = model.target_persons_text_embeddings(current_token_ids, current_orig_token_length)
                    text_processing_elapsed_time = time.time() - text_processing_timer_started
                    logger.info(f"Text feature extraction finished in {text_processing_elapsed_time} seconds")

                    logger.info("Multi-person image feature extraction started")
                    img_processing_start = datetime.datetime.now()
                    with torch.no_grad():
                        persons_embeddings_out = model.persons_embeddings(original_img_as_tensor.to(configs.device), processed_img.to(configs.device))
                    logger.info("Multi-person image feature extraction finished")
                    img_processing_end = datetime.datetime.now()
                    img_process_time_elapsed = img_processing_end - img_processing_start
                    img_process_time_elapsed = img_process_time_elapsed.total_seconds()
                    logger.info(f"Multi-person image feature extraction finished in {img_process_time_elapsed} seconds")

                    if persons_embeddings_out != None:
                        image_embedding, bbox = persons_embeddings_out

                        logger.info("Similarity calculation started")
                        similarity_timer_started = time.time()
                        similarity = calculate_similarity(image_embedding,textual_embedding).numpy()
                        logger.info("Similarity calculation finished")
                        similarity_timer_end = time.time()
                        similarity_timer_elapsed = similarity_timer_end - similarity_timer_started


                        # Sort the indeces of similarity score, from the highest score to the lowest
                        indices_sorted = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)
                        ranked_bbs = [bbox[i].tolist() for i in indices_sorted]
                        ranked_scores = [similarity[i].tolist()[0] for i in indices_sorted]

                    # total time
                    search_person_completed = time.perf_counter_ns()
                    total_time = search_person_completed - message_recieved_time
                    total_time = total_time/1_000_000_000
                    # data2write = f"{number_of_persons},{message_recieved_time},{search_person_completed},{total_time},{similarity_timer_elapsed}\n"
                    # write_to_text_file(os.path.join("..","data","logs","gallery_multipersonimage_scalability_test.txt"),data2write,"append")
                    logger.info(f"Total time elapsed: {total_time:.9f}")
                    logger.info(f"The scores are: {ranked_scores}")
                    report_confidence_score(mqtt_client, camera_name,ranked_scores[0]) # take the heighest ranked score
                    logger.info("Confidence score sent back to master node")
                    print("*" * 50)
                    print("\n")
        

        except zmq.Again:
            time.sleep(0.01)
            continue
        except (pickle.UnpicklingError, zmq.ZMQError) as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            break
