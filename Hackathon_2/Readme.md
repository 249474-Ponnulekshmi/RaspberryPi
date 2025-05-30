# Weather Display & Alert System

This project fetches real-time weather data from the **OpenWeatherMap API** on a laptop and sends it to a **Raspberry Pi 5** over Wi-Fi. The Pi displays the data on an **OLED screen (Pioneer600 HAT)** and controls an **LED on GPIO26** based on a temperature threshold.

## Components Used

- Raspberry Pi 5 with Pioneer600 HAT
- SSD1306 OLED Display (128x64)
- LED connected to GPIO26
- Laptop or PC (as client)
- Python 3.7+
- Internet connection (for API calls)

## How It Works

1. **Client Side (Laptop):**
   - Asks for a city name.
   - Fetches weather data (temperature, pressure, condition).
   - Sends JSON data to Raspberry Pi over TCP.

2. **Server Side (Raspberry Pi):**
   - Listens on port `5555` for incoming weather data.
   - Displays data on the OLED screen using Pioneer600.
   - Turns **ON the LED** on GPIO26 if temperature > 30°C.

---

## Run the server
  - python server.py
## Run the client
  - python client.py
## Things to note
  - Give the correct IP of the raspberry Pi in the server code.
  - You can use any port number, but it should be same in both server and client

## Example output at Client Side
  -> Enter city name (or type '!' to quit): Trivandrum
  -> Temp: 24.5°C
  -> Pressure: 1013 hPa
  -> Condition: Heavy intensity rain
  -> Data sent to Raspberry Pi.
## Example output at Server side
  -> Server is listening on port 5555...
  -> Connection from ('192.168.1.2', 52674)
  -> Received data:
  -> {'temperature': 24.5, 'pressure': 1013, 'condition': 'Heavy intensity rain'}
  -> Temp normal. LED OFF




