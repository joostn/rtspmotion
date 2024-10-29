import cv2
import numpy as np
import paho.mqtt.client as mqtt
import argparse
import time

# Configuration
#args.video_url = "rtsp://camserver.lan.newhouse.nl:8554/tuin_motion"  # Replace with your RTSP stream URL
#THRESHOLD = 5000  # Define an appropriate threshold for motion detection

last_trigger_time = 0

def publish_mqtt_message(args):
    """Publishes a message to the MQTT server."""
    client = mqtt.Client()
    
    if args.mqtt_username and args.mqtt_password:
        client.username_pw_set(args.mqtt_username, args.mqtt_password)
    
    client.connect(args.mqtt_server, args.mqtt_port, 60)
    client.publish(args.mqtt_topic, args.mqtt_value)
    client.disconnect()

def triggerMotionDebounced(args):
    global last_trigger_time
    current_time = time.time()
    if current_time - last_trigger_time < args.backoff_time:
        print("Motion detected, but debounced.")
        return
    print("Motion detected!")
    publish_mqtt_message(args)

def main():
    parser = argparse.ArgumentParser(description="Motion detection using RTSP video stream with MQTT alerts.")
    parser.add_argument("--video_url", required=True, help="RTSP URL of the video stream.")
    parser.add_argument("--threshold", type=int, choices=range(1, 256), default=10, help="Pixel difference threshold (1 - 255).")
    parser.add_argument("--threshold_count", type=float, default=0.0004, help="Fraction of different pixels to trigger an alert.")
    parser.add_argument("--mqtt_server", required=True, help="Hostname of the MQTT server.")
    parser.add_argument("--mqtt_port", type=int, default=1883, help="Port of the MQTT server. Default is 1883.")
    parser.add_argument("--mqtt_username", help="Username for authenticating with the MQTT server.")
    parser.add_argument("--mqtt_password", help="Password for authenticating with the MQTT server.")
    parser.add_argument("--mqtt_topic", required=True, help="Topic to publish to on the MQTT server.")
    parser.add_argument("--mqtt_value", required=True, help="Value to publish on the specified topic.")
    parser.add_argument("--backoff_time", type=int, default=30, help="Number of seconds to back off after triggering an alert. Default is 30 seconds.")

    args = parser.parse_args()

    # Open the RTSP stream
    cap = cv2.VideoCapture(args.video_url)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream:  {args.video_url}")
        return 1

    print(f"Opened RTSP stream: {args.video_url}")
    prev_gray = None
    while True:
        # Read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from stream.")
            return 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if not prev_gray is None:
            diff = cv2.absdiff(prev_gray, gray)
            max_value = np.max(diff)
            # print(f"Max value: {max_value}")
            _, thresh = cv2.threshold(diff, args.threshold, 255, cv2.THRESH_BINARY)
            non_zero_count = np.count_nonzero(thresh)
            non_zero_fraction = non_zero_count / (height * width)
            #print (f"non_zero_fraction: {non_zero_fraction}")
            if non_zero_count > args.threshold:
                triggerMotionDebounced(args)
        prev_gray = gray

if __name__ == "__main__":
    exitcode = main()
    exit(exitcode)
