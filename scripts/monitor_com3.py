#!/usr/bin/env python3
"""
Monitor COM3 for STM32 N657 board output
"""

import serial
import sys
import time

def monitor_com3():
    try:
        # Configure the serial port
        ser = serial.Serial(
            port='COM3',
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
        )
        
        print(f"Monitoring COM3 at 115200 baud...")
        print("Press Ctrl+C to stop.")
        print("-" * 40)
        
        # Read and display data
        while True:
            if ser.in_waiting > 0:
                # Read all available data
                data = ser.read(ser.in_waiting)
                # Decode and print
                try:
                    text = data.decode('utf-8', errors='ignore')
                    print(text, end='', flush=True)
                except UnicodeDecodeError:
                    # If we can't decode, print raw bytes
                    print(data.hex(), end='', flush=True)
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
            
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        return 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("\nSerial port closed.")

if __name__ == "__main__":
    sys.exit(monitor_com3())