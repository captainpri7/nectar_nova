# nectar_nova
nectar nova
(AI was used to help me write this code) 
15% AI % 85% Human

Artificial Pollinator Project
Autonomous robotic pollination system

Description: 
NectarNova is an autonomous AI-powered drone built to address the global decline in bee populations and its impact on pollination. The project combines a custom-built quadcopter — powered by a Pixhawk 2.4.8 Pro flight controller running ArduCopter firmware — with a Raspberry Pi 5 acting as an onboard AI computer. A custom YOLOv8 object detection model, trained on a flower dataset, runs in real time on the Pi to identify flowers through an onboard camera. Once a flower is detected, the system calculates its position and distance, then sends live velocity commands to the flight controller over a MAVLink connection, autonomously guiding the drone toward the flower and stopping at a set distance. The project involved end-to-end engineering work including firmware flashing, sensor calibration, motor and ESC configuration, PID flight tuning, and integrating computer vision with real-time flight control — demonstrating how robotics and artificial intelligence can be applied to support pollinator-assistance technology and address a real environmental challenge.

