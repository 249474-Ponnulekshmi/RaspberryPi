import requests
import socket
import json

API_KEY = "fc05766c72ad695e4df378d2aecef2a0"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(city_name):
    try:
        params = {
            "q": city_name,
            "appid": API_KEY,
            "units": "metric"
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if response.status_code != 200:
            return f"Error: {data.get('message', 'Unknown error')}"

        temp = data["main"]["temp"]
        pressure = data["main"]["pressure"]
        condition = data["weather"][0]["description"]

        return {
            "temperature": temp,
            "pressure": pressure,
            "condition": condition.capitalize()
        }

    except Exception as e:
        return f"Exception: {e}"

def send_to_pi(weather_data):
    try:
        json_data = json.dumps(weather_data)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("192.168.250.169", 5555))
            s.sendall(json_data.encode('utf-8'))
            print("Data sent to Raspberry Pi.")
    except Exception as e:
        print(f" Error sending data: {e}")

if __name__ == "__main__":
    while True:
        try:
            city_name = input("Enter city name (or type '!' to quit): ").strip()
            if city_name.lower() == "!":
                print("Exiting client.")
                break

            weather_info = get_weather(city_name)

            if isinstance(weather_info, str):
                print("Info", weather_info)
            else:
                print(f"Temp: {weather_info['temperature']}°C")
                print(f"Pressure: {weather_info['pressure']} hPa")
                print(f"Condition: {weather_info['condition']}")
                send_to_pi(weather_info)

        except Exception as e:
            print(f"Error Occurred: {e}")
