from dronekit import connect, VehicleMode
import time

# Connect to the Pixhawk
vehicle = connect('/dev/ttyUSB0', baud=57600, wait_ready=True)

# Arm the vehicle
print("Arming motors...")
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True

# Wait for arm to complete
while not vehicle.armed:
    print("Waiting for arming...")
    time.sleep(1)

print("Motors armed!")

# Disarm to stop motors
print("Disarming motors...")
vehicle.armed = False

# Close connection
vehicle.close()
print("Motor stopped")