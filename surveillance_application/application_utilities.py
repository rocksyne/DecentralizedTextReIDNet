# Sys. modules
import time
import pickle
import io
import traceback
from datetime import datetime
from threading import Thread

# 3rd-party modules
import zmq
import imagezmq
import torch
from textblob import TextBlob
import pymysql
import pymysql.cursors
from dateutil.relativedelta import relativedelta

# +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++[Utility Functions]++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++
def thread_target_wrapper(func):
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in the thread: {e}")
            traceback.print_exc()
    return wrapped


def setup_image_hub():
    address = f"tcp://{configs.command_station_address}:{configs.video_parsing_app_port}"
    image_hub = imagezmq.ImageHub(open_port=address, REQ_REP=True)
    while not shutdown_event.is_set():  # Check for shutdown signal
        try:
            node_name, image = image_hub.recv_image()
            thread_queue.put([node_name, image])
            image_hub.send_reply(b'OK')
        except Exception as e:
            print(f"Error in ImageHub thread: {e}")
            break
    image_hub.close()
    print("ImageHub connection closed.")
        

def start_image_hub_thread():
    thread = Thread(target=thread_target_wrapper(setup_image_hub), name='video_feed_thread', daemon=True)
    thread.start()
    return thread


def correct_sentence(text):
    blob = TextBlob(text)
    corrected_text = ''

    for sentence in blob.sentences:
        corrected_sentence = sentence.string.capitalize()
        if not corrected_sentence.endswith('.'):
            corrected_sentence += '.'
        corrected_text += corrected_sentence + ' '

    return corrected_text.strip()



def serialize_tensor(tensor:torch.Tensor,method:str="pickle")->bytes:
    """
    Doc.:   Method for serializing pytorch tensor (converting to bytes).
    Args.:  -tensor: the torch tensor to be converted to bytes
            -method: the method to be used for the serialization
    """
    if method == "pickle":
        tensor_as_bytes = pickle.dumps(tensor)
    elif method == "io_bytes":
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        tensor_as_bytes = buffer.getvalue() 
    else:
        raise NotImplementedError("{} method is not implemented.".format(method))
    return tensor_as_bytes



def send_command(command, parameter=None):
    # Prepare the message
    message = [command.encode()]
    if parameter is not None:
        # If parameter is a tensor, serialize it
        if isinstance(parameter, torch.Tensor):
            parameter = serialize_tensor(parameter)
        elif isinstance(parameter, str):
            parameter = parameter.encode()
        
        message.append(parameter)

    # Prepare the ZMQ context and socket
    try:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        address = f"tcp://{configs.command_station_address}:{configs.message_parsing_app_port}"
        socket.bind(address)
        time.sleep(1)  # Wait a bit for the socket to initialize
        socket.send_multipart(message)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        socket.close()
        context.term()









def send_tensor(serialized_tensor):
    # Prepare the ZMQ context and socket
    try:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        address = f"tcp://{configs.command_station_address}:{configs.message_parsing_app_port}"
        socket.bind(address)
        time.sleep(1) # Wait a bit for the socket to initialize
        socket.send(serialized_tensor)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        socket.close()
        context.term()


def get_db_connection(config):
    try:
        connection = pymysql.connect(host=config.server_address,
                                 user=config.username,
                                 password=config.password,
                                 database=config.database_name,
                                 cursorclass=pymysql.cursors.DictCursor)
        return connection
    except Exception as e:
        print(f"An error occurred: {e}")



def time_ago(past_datetime):
    # Get the current datetime
    now = datetime.now()
    
    # Calculate the difference between the current time and the past time
    delta = relativedelta(now, past_datetime)
    
    # Define the human-readable strings for each time unit
    attributes = ["years", "months", "days", "hours", "minutes", "seconds"]
    names = ["year", "month", "day", "hour", "minute", "second"]
    
    # Construct a string based on the largest non-zero time difference
    for attr, name in zip(attributes, names):
        value = getattr(delta, attr)
        if value:
            if value == 1:
                return f"{value} {name} ago"
            else:
                return f"{value} {name}s ago"

    return "just now"      