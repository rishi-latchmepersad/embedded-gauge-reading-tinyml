import serial
import sys

try:
    # Open the serial port
    ser = serial.Serial('COM3', 115200, timeout=1)
    print("Monitoring COM3... Press Ctrl+C to stop.")
    print("-" * 40)
    
    while True:
        # Read a line from the serial port
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)
                
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
except KeyboardInterrupt:
    print("\nMonitoring stopped by user.")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")