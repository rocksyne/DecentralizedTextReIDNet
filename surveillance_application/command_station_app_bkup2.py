"""
Disclaimer: This is not an enterprise-grade "ready to deploy" application. It is only a prototype
            (proof of concept) and should therefore be used within this scope.
            Additionally, features related to (network)security are outside the scope of this work.
            Therefore, authors bare no liability for the use of this work in private or commercial settings.
            Please see license for more information.

Doc.:   Application for decentralized person re-identification
Date:   2nd April, 2024
For:    Institute of Networked and Embedded Systems, University of Klagenfurt.
"""

# System modules
import os
import sys
import time
import queue
import pickle
import io
import traceback
from os.path import abspath
from datetime import datetime
from threading import Thread, Event
from collections import deque

# Third-party modules
import cv2
import zmq
import imagezmq
import nltk
import torch
from textblob import TextBlob
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, flash, session
import pymysql
import pymysql.cursors
import paho.mqtt.client as mqtt
from dateutil.relativedelta import relativedelta
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Application modules and configurations
sys.path.insert(0, os.path.abspath('../'))
from datasets.custom_dataloader import process_text_into_tokens
from config import sys_configuration

# Global variables
nltk.download('punkt')
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necessary for flash messages
mqtt_client = mqtt.Client()
configs = sys_configuration()
video_queue = queue.Queue()
send_queue = queue.Queue()
shutdown_event = Event()  # Event to signal thread shutdown
confidence_scores = {}
selected_camera = None
camera_search_logs = {}


# +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++[Utility Functions]++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++
def clear_queue_thread_safe(q):
    with q.mutex:
        q.queue.clear()

def on_connect(client, userdata, flags, rc):
    """Handles connection to MQTT broker."""
    print("Connected with result code " + str(rc))
    client.subscribe("camera/status")
    client.subscribe("camera/confidence")


def on_message(client, userdata, msg):
    """Handles incoming MQTT messages."""
    global selected_camera
    topic = msg.topic
    payload = msg.payload.decode()

    if topic == "camera/status":
        handle_camera_status(payload)
    elif topic == "camera/confidence":
        handle_camera_confidence(payload)


def handle_camera_status(payload):
    """Processes camera status messages."""
    camera_name, camera_ip_address, status_code = payload.split("_")
    try:
        connection = get_db_connection(configs)
        with connection.cursor() as cursor:
            sql = (
                f"INSERT INTO `camera_status`(`camera_name`, `ip_address`, `status`, `last_seen`) "
                f"VALUES ('{camera_name}', '{camera_ip_address}', '{status_code}', NOW()) "
                f"ON DUPLICATE KEY UPDATE `ip_address` = VALUES(`ip_address`), `status` = VALUES(`status`), `last_seen` = NOW();"
            )
            cursor.execute(sql)
            connection.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        connection.close()


def handle_camera_confidence(payload):
    """Processes camera confidence messages."""
    global selected_camera
    global camera_search_logs
    camera_name, score = payload.split("_")
    confidence_scores[camera_name] = float(score)

    if confidence_scores:
        selected_camera = max(confidence_scores, key=confidence_scores.get)
        send_command('stream_retrieved_video', [selected_camera])
        camera_search_logs[camera_name] = [score, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        print("Selected camera", selected_camera)


def run_mqtt_loop():
    """Runs the MQTT loop."""
    mqtt_client.loop_forever()

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(configs.command_station_address, 1883, 60)  # Update the broker address and port


def thread_target_wrapper(func):
    """Wrapper for thread target functions to handle exceptions."""
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in the thread thread_target_wrapper(): {e}")
            traceback.print_exc()
    return wrapped


def setup_image_hub():
    """Sets up the ImageHub for receiving images."""
    address = f"tcp://{configs.command_station_address}:{configs.video_parsing_app_port}"
    image_hub = imagezmq.ImageHub(open_port=address, REQ_REP=True)
    try:
        while not shutdown_event.is_set():  # Check for shutdown signal
            try:
                node_name, image = image_hub.recv_image()
                if node_name is None or image is None:
                    print("No data received, clearing video queue.")
                    clear_queue_thread_safe(video_queue)
                else:
                    video_queue.put([node_name, image])
                    image_hub.send_reply(b'OK')
            except zmq.ZMQError as zmq_error:
                print(f"ZMQ error in ImageHub thread: {zmq_error}")
                clear_queue_thread_safe(video_queue)
                break
            except Exception as e:
                print(f"Error in ImageHub thread: {e}")
                clear_queue_thread_safe(video_queue)
                break
    finally:
        image_hub.close()
        print("ImageHub connection closed.")


def correct_sentence(text):
    """Corrects and formats sentences."""
    blob = TextBlob(text)
    corrected_text = ''

    for sentence in blob.sentences:
        corrected_sentence = sentence.string.capitalize()
        if not corrected_sentence.endswith('.'):
            corrected_sentence += '.'
        corrected_text += corrected_sentence + ' '

    return corrected_text.strip()


def serialize_tensor(tensor: torch.Tensor, method: str = "pickle") -> bytes:
    """Serializes a PyTorch tensor to bytes."""
    if method == "pickle":
        tensor_as_bytes = pickle.dumps(tensor)
    elif method == "io_bytes":
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        tensor_as_bytes = buffer.getvalue() 
    else:
        raise NotImplementedError(f"{method} method is not implemented.")
    return tensor_as_bytes


def send_command(command, parameter=None):
    """Sends a command via ZMQ."""
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
    
    send_queue.put(message)
    print("Message put in the queue")


def get_db_connection(config):
    """Gets a database connection."""
    try:
        connection = pymysql.connect(
            host=config.server_address,
            user=config.username,
            password=config.password,
            database=config.database_name,
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"An error occurred in get_db_connection(): {e}")


def time_ago(past_datetime):
    """Calculates the time difference in a human-readable format."""
    now = datetime.now()
    delta = relativedelta(now, past_datetime)
    attributes = ["years", "months", "days", "hours", "minutes", "seconds"]
    names = ["year", "month", "day", "hour", "minute", "second"]
    
    for attr, name in zip(attributes, names):
        value = getattr(delta, attr)
        if value:
            return [value, name]
    return [0, 'second']


def zmq_sender():
    """Sends messages via ZMQ."""
    while not shutdown_event.is_set():
        try:
            if not send_queue.empty():
                message = send_queue.get(timeout=1)
                context = zmq.Context()
                socket = context.socket(zmq.PUB)
                address = f"tcp://{configs.command_station_address}:{configs.message_parsing_app_port}"
                socket.bind(address)
                time.sleep(1)
                socket.send_multipart(message)
                print("Message was sent successfully...")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"An error occurred in zmq_sender(): {e}")


# +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++[Flask Routes]+++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++
@app.route('/')
def home():
    """Home route."""
    if session.get('logged_in'):
        session.pop('logged_in', None)

    current_year = datetime.now().year
    return render_template('login.html', app_version=configs.app_version, current_year=current_year, default_sensitivity=configs.reID_confidence_threshold)


@app.route('/login', methods=['POST'])
def login():
    password = request.form.get('password')
    
    if password == configs.PASSWORD:
        session['logged_in'] = True
        return redirect(url_for('search_from_cameras'))
    else:
        flash('Incorrect password. Please see PASSWORD value in config.py file.')
        return redirect(url_for('home'))
    

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have successfully logged out.')
    return redirect(url_for('home'))


@app.route('/search-from-cameras')
def search_from_cameras():
    """Home route."""
    if not session.get('logged_in'):
        flash('Page is protected. Please log in first to continue.')
        return redirect(url_for('home'))
    
    current_year = datetime.now().year
    clear_queue_thread_safe(video_queue) # clear the queue
    send_command('set_sensitivity', pickle.dumps(configs.reID_confidence_threshold))
    send_command('stop_person_search', ['cameraone', 'cameratwo', 'camerathree', 'camerafour', 'camerafive', 'deeplearning'])
    return render_template('search_from_caameras.html', app_version=configs.app_version, current_year=current_year, default_sensitivity=configs.reID_confidence_threshold)


@app.route('/get-camera-live-view')
def get_camera_live_view():
    """Route to get camera live view."""
    if not session.get('logged_in'):
        flash('Page is protected. Please log in first to continue.')
        return redirect(url_for('home'))
    
    current_year = datetime.now().year
    clear_queue_thread_safe(video_queue) # clear the queue
    send_command('set_sensitivity', pickle.dumps(configs.reID_confidence_threshold))
    send_command('stop_person_search', ['cameraone', 'cameratwo', 'camerathree', 'camerafour', 'camerafive', 'deeplearning'])
    return render_template('get_camera_live_view.html', app_version=configs.app_version, current_year=current_year, default_sensitivity=configs.reID_confidence_threshold)


@app.route('/stream-live-camera-feed', methods=['POST'])
def stream_live_camera_feed():
    """Route to stream live camera feed."""
    camera_name = request.get_json().get('camera_name')
    clear_queue_thread_safe(video_queue) # clear the queue
    send_command('set_sensitivity', pickle.dumps(configs.reID_confidence_threshold))
    send_command('stop_person_search', [camera_name])
    send_command('stream_live_video', [camera_name])
    return jsonify({"message": "successful"})


@app.route('/video-feed')
def video_feed():
    """Route to serve the video stream with initial dummy image and timeout handling."""
    
    other_stats_font = ImageFont.truetype("arial.ttf", 20)
    
    def read_staic_image_as_bytes(image_path: str = "../data/sample_images/no_video.jpg"):
        """Reads an image, overlays a timestamp, and converts it to JPEG bytes."""
        # Load the image and ensure it is in RGB mode
        pil_image = Image.open(image_path).convert("RGB")
        pil_image_draw = ImageDraw.Draw(pil_image)
        
        # Time text matters
        timestamp = datetime.now().strftime('%A %d %B, %Y | %H:%M:%S')
        left, top, right, bottom = pil_image_draw.textbbox((0, 0), timestamp, font=other_stats_font)
        text_width = right - left
        text_height = bottom - top
        text_x = (pil_image.width - text_width) // 2
        text_y = 70 + ((90 - 25) - text_height) // 2

        # Overlay the timestamp on the image
        pil_image_draw.text((text_x, text_y), timestamp, font=other_stats_font, fill=(255, 255, 255)) 

        # Convert the image to JPEG format and save to a BytesIO object
        with io.BytesIO() as output:
            pil_image.save(output, format="PNG")
            jpg_bytes = output.getvalue()

        return jpg_bytes
    
    def generate():
        
        serve_dummy = True
        last_time = time.time()
        while True:
            
            try:
                if serve_dummy or time.time() - last_time > 1:
                    frame = read_staic_image_as_bytes()
                    serve_dummy = False
                    last_time = time.time()
                else:
                    node_name, image = video_queue.get(block=False, timeout=None)

                    # TODO: Messy code. Clean up
                    # time stamp the image before showing in browser
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    timestamp = f"Frame received at monitoring station: {timestamp}"
                    left, top, right, bottom = other_stats_font.getbbox(timestamp)
                    text_width = right - left
                    text_height = bottom - top
                    x = 512 - text_width - 1 # some small padding
                    y = 512 - text_height - 3 # push it to the top

                    # write on the image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    draw_on_original = ImageDraw.Draw(image)
                    draw_on_original.text((x, y), timestamp, font=other_stats_font, fill=(255, 255, 255))  # White text
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # convert to bytes
                    _, buffer = cv2.imencode('.jpg', image)
                    frame = buffer.tobytes()
                    last_time = time.time()
            except queue.Empty:
                frame = read_staic_image_as_bytes()
                last_time = time.time()
            
            except Exception as e:
                print(f"Error in generate(): {e}")

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get-search-operation-status')
def get_search_operation_status():
    """Route to get summarized search operation logs."""
    summary_log = [{'camera': camera, 'score': logs[0], 'timestamp': logs[1]} for camera, logs in camera_search_logs.items()]
    summary_log.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'), reverse=True) # Sort the list by timestamp
    return jsonify(summary_log)



@app.route('/get-camera-status')
def get_camera_status():
    """Route to get camera status."""
    connection = get_db_connection(configs)
    
    status_mapping = {
        "CODE01": ("Online and Idle", "success"),
        "CODE02": ("Online and Searching", "warning"),
        "CODE03": ("Online and Streaming", "warning"),
        "CODE404": ("Camera Error - Restart needed", "danger")
    }
    
    try:
        with connection.cursor() as cursor:
            sql = ( "SELECT `camera_name`, `ip_address`, `status`, `last_seen` "
                    "FROM `camera_status` " 
                    "ORDER BY `ip_address` ASC"
                    )

            cursor.execute(sql)
            camera_statuses = cursor.fetchall()

            if not camera_statuses:
                return jsonify({"message": "No camera data available", "data": []})

            for camera in camera_statuses:
                time_value, time_unit = time_ago(camera['last_seen'])
                camera_online = (time_value < 3 and time_unit == 'second')
                
                if camera_online:
                    if camera['status'] in status_mapping:
                        camera['status'], camera['badge'] = status_mapping[camera['status']]
                    else:
                        raise ValueError(f"Invalid status code: {camera['status']}")
                else:
                    camera['status'] = "Offline"
                    camera['badge'] = 'danger'

                if time_value == 0 and time_unit == 'second':
                    camera['last_seen'] = "just now"
                else:
                    unit = time_unit if time_value == 1 else f"{time_unit}s"
                    camera['last_seen'] = f"{time_value} {unit} ago"

            return jsonify(camera_statuses)
    
    except ValueError as ve:
        print(f"ValueError occurred: {ve}")
        return jsonify({"message": f"ValueError occurred: {ve}"})
    except Exception as e:
        print(f"An error occurred in get_camera_status(): {e}")
        return jsonify({"message": f"An error occurred: {e}"})
    finally:
        connection.close()


@app.route('/search-person', methods=['POST'])
def search_person():
    global camera_search_logs
    camera_search_logs = {}
    """Route to search for a person."""
    data = request.get_json()
    search_text = data.get('user_text')
    user_text = correct_sentence(search_text)
    token_ids, orig_token_length = process_text_into_tokens(user_text)
    token_ids = token_ids.unsqueeze(0).long()
    orig_token_length = torch.tensor([orig_token_length]).unsqueeze(-1)
    tensor = torch.cat((token_ids, orig_token_length), dim=1)
    if search_text:
        send_command('set_sensitivity', pickle.dumps(configs.reID_confidence_threshold))
        send_command('search_person', tensor)
        return jsonify({'status': 'success', 'message': 'Text received', 'text': user_text})
    return jsonify({'status': 'error', 'message': 'No search text provided'}), 400


@app.route('/stop-person-search', methods=['POST'])
def stop_person_search():
    """Route to stop person search."""
    clear_queue_thread_safe(video_queue) # clear the queue
    send_command('stop_person_search', ['cameraone', 'cameratwo', 'camerathree', 'camerafour', 'camerafive', 'deeplearning'])
    return jsonify({'status': 'success', 'message': 'Command sent'})


@app.route('/set-sensitivity', methods=['POST'])
def set_sensitivity():
    """Route to set sensitivity."""
    data = request.get_json()
    sensitivity = data.get('sensitivity')
    if sensitivity is not None:
        send_command('set_sensitivity', pickle.dumps(sensitivity))
        return jsonify({'status': 'success', 'sensitivity': sensitivity})
    return jsonify({'status': 'error', 'message': 'No sensitivity value provided'}), 400



if __name__ == '__main__':    
    # video feed thread
    video_thread = Thread(target=thread_target_wrapper(setup_image_hub), name='video_feed_thread', daemon=True)
    video_thread.start()

    mqtt_thread = Thread(target=run_mqtt_loop, name='mqtt_thread', daemon=True)
    mqtt_thread.start()

    # Start the ZMQ sender thread
    sender_thread = Thread(target=zmq_sender, name='zmq_sender_thread', daemon=True)
    sender_thread.start()

    try:
        app.run(host="143.205.116.144", port=5001)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        shutdown_event.set()  # Signal the thread to shutdown
        clear_queue_thread_safe(video_queue) # clear the queu
        print("Server and threads have been shut down.")
    except Exception as e:
        print(f"An error occurred in main(): {e}")
    finally:
        shutdown_event.set()  # Signal the thread to shutdown
