import socket
import json
from display import display_weather
from gpiozero import LED

TEMP_THRESHOLD = 30.0

def receive_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 5555))
    sock.listen(5)
    print("Server is listening on port 5555...")

    led = LED(26)

    while True:
        conn, addr = sock.accept()
        print(f"Connection from {addr}")
        data = conn.recv(1024)
        if not data:
            conn.close()
            continue

        try:
            weather_data = json.loads(data.decode('utf-8'))
            print("Received data:\n", weather_data)

            temp = weather_data.get("temperature")
            pressure = weather_data.get("pressure")
            condition = weather_data.get("condition")

            display_weather(temp, pressure, condition)

            if temp > TEMP_THRESHOLD:
                led.on()
                print("High temp! LED ON")
            else:
                led.off()
                print("Temp normal. LED OFF")

        except Exception as e:
            print(f"Error in server.py: {e}")

        conn.close()
        print("Connection closed.")

if __name__ == "__main__":
    try:
        receive_data()
    except Exception as e:
        print(f"Server error: {e}")
