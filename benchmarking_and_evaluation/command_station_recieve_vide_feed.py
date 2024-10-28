import cv2
import imagezmq
import sys, os, socket
import numpy as np
from datetime import datetime
import random
import threading
import queue
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import time

sys.path.insert(0, os.path.abspath('../'))

from config import sys_configuration
from evaluation.evaluations import calculate_similarity
from datasets.custom_dataloader import process_text_into_tokens
from model.unity import Unity
from utils.miscellaneous_utils import set_seed
from evaluation.evaluations import calculate_similarity
from utils.mhpv2_dataset_utils import CustomMHPv2ImageResizer

configs = sys_configuration()
camera_name = socket.gethostname()
model = Unity(configs).to(configs.device)
model.eval()
MHPv2_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=configs.MHPV2_means, std=configs.MHPV2_stds)])
rgb_img_resizer = CustomMHPv2ImageResizer(dimensions=configs.MHPV2_image_size, image_type='RGB')
previous_good_frame = black_canvas = np.ones((512, 512, 3), dtype=np.uint8) * 100
color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

frame_queue = queue.Queue()
processed_frame = np.ones((512, 512, 3), dtype=np.uint8) * 255  # Start with a white canvas

def process_frames(frame, token_ids, orig_token_length):
    global previous_good_frame
    # print("Processing frame...")

    def draw_bounding_boxes(image, bounding_boxes, confidence_scores, top_1=False):
        if not isinstance(bounding_boxes, list):
            bounding_boxes = [bounding_boxes]

        if not isinstance(confidence_scores, list):
            confidence_scores = [confidence_scores]

        if top_1:
            bounding_boxes = [bounding_boxes[0]]
            confidence_scores = [confidence_scores[0]]

        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        for bbox, score in zip(bounding_boxes, confidence_scores):
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"Conf: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, y1 - text_height, x1 + text_width + 10, y1], fill=color)
            draw.text((x1 + 5, y1 - text_height), text, fill="white", font=font)

        return image

    def read_and_process_image(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        processed_img = rgb_img_resizer(frame)
        original_img = processed_img.copy()
        processed_img = MHPv2_transform(processed_img)
        processed_img = processed_img.unsqueeze(0)

        original_img_as_tensor = transforms.ToTensor()(original_img)
        return original_img, original_img_as_tensor, processed_img

    start_time = time.time()
    original_img, original_img_as_tensor, processed_img = read_and_process_image(frame)

    with torch.no_grad():
        persons_embeddings_out = model.persons_embeddings(original_img_as_tensor, processed_img.to(configs.device))
        textual_embedding = model.target_persons_text_embeddings(token_ids, orig_token_length)

    if persons_embeddings_out is not None:
        image_embedding, bbox = persons_embeddings_out
        similarity = calculate_similarity(image_embedding, textual_embedding).numpy()
        indices_sorted = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)
        ranked_bbs = [bbox[i].tolist() for i in indices_sorted]
        ranked_scores = [similarity[i].tolist()[0] for i in indices_sorted]

        filtered_boxes_with_scores = [(bbox, score) for bbox, score in zip(ranked_bbs, ranked_scores) if score >= 0.64]

        if len(filtered_boxes_with_scores) > 0:
            filtered_boxes, filtered_scores = zip(*filtered_boxes_with_scores)
            filtered_boxes, filtered_scores = list(filtered_boxes), list(filtered_scores)
            camera_output = draw_bounding_boxes(original_img, filtered_boxes, filtered_scores, top_1=True)
            camera_output = np.array(camera_output)

            cv2.putText(camera_output, "Target last spotted in this location", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.87, (100, 80, 255), 2, cv2.LINE_AA)
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            timestamp = f"Processed: {timestamp}"

            timestamp_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            timestamp_x = camera_output.shape[1] - timestamp_size[0] - 10
            cv2.putText(camera_output, timestamp, (timestamp_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            camera_output = cv2.cvtColor(camera_output, cv2.COLOR_RGB2BGR)
            previous_good_frame = camera_output.copy()
        else:
            camera_output = previous_good_frame
    else:
        camera_output = previous_good_frame

    end_time = time.time()
    print(f"Processed frame in {end_time - start_time:.2f} seconds")

    return camera_output

def processing_worker(token_ids, orig_token_length):
    global processed_frame

    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        processed_frame = process_frames(frame, token_ids, orig_token_length)
        frame_queue.task_done()

def receive_video(bind_ip, bind_port):
    global processed_frame
    receiver = imagezmq.ImageHub(open_port=f'tcp://{bind_ip}:{bind_port}')
    textual_description = "The man is wearing a white shirt and brown pants. He is wearing a black backpack."
    token_ids, orig_token_length = process_text_into_tokens(textual_description)
    token_ids = token_ids.unsqueeze(0).to(configs.device).long()
    orig_token_length = torch.tensor([orig_token_length]).to(configs.device)

    frames = {}  # Dictionary to store frames from each camera

    processing_thread = threading.Thread(target=processing_worker, args=(token_ids, orig_token_length))
    processing_thread.daemon = True
    processing_thread.start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('combined_video.mp4', fourcc, 20.0, (1024, 1576))  # Adjust the size as per your combined frame size

    try:
        frame_counter = 1
        while True:
            camera_id, frame = receiver.recv_image()

            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            timestamp = f"Received: {timestamp}"

            timestamp_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            timestamp_x = frame.shape[1] - timestamp_size[0] - 10
            cv2.putText(frame, timestamp, (timestamp_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            frame_queue.put(frame.copy())

            frames[camera_id] = frame

            if frames:
                frame_list = list(frames.values())

                while len(frame_list) < 5:
                    frame_list.append(np.zeros_like(frame_list[0]))

                frame_list.append(processed_frame)

                spacer = np.zeros((frame_list[0].shape[0], 20, 3), dtype=np.uint8)

                top_row = np.hstack([frame_list[0], spacer, frame_list[1], spacer, frame_list[2]])
                bottom_row = np.hstack([frame_list[3], spacer, frame_list[4], spacer, frame_list[5]])
                combined_frame = np.vstack([top_row, bottom_row])

                cv2.imshow('Combined Video Feed', combined_frame)

                #print("Shape is: ", combined_frame.shape)

                # Write the combined frame to the video file
                cv2.imwrite(f"../data/centralized_frames/frame_{frame_counter}.jpg",combined_frame)
                #out.write(combined_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            receiver.send_reply(b'OK')

    except KeyboardInterrupt:
        frame_queue.put(None)
        processing_thread.join()
        out.release()
        cv2.destroyAllWindows()

    finally:
        frame_queue.put(None)
        processing_thread.join()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    receive_video(configs.command_station_address, configs.video_parsing_app_port)
