import os
import subprocess
import sys
import re
import math

def amplitude_to_db(value):
    if value <= 0:
        return float("-inf")
    return 20.0 * math.log10(value)

def check_audio_levels(directory):
    header = "File Path|Peak Level (dB)|RMS Value (dB)|Noise Floor (dB)|ACX Peak|ACX RMS|ACX Noise Floor"
    print(header)
    print("-" * len(header))

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                stats_command = ["sox", file_path, "-n", "stats"]
                try:
                    stats_output = subprocess.check_output(stats_command, stderr=subprocess.STDOUT)
                    stats_lines = stats_output.decode().split("\n")
                    
                    peak_level_line = [line for line in stats_lines if line.startswith("Pk lev dB")]
                    rms_level_line = [line for line in stats_lines if line.startswith("RMS lev dB")]
                    noise_floor_line = [line for line in stats_lines if line.startswith("RMS Tr dB")]
                    
                    if peak_level_line and rms_level_line and noise_floor_line:
                        peak_level_db = float(re.findall(r"[-+]?\d*\.\d+|\d+", peak_level_line[0])[0])
                        rms_level_db = float(re.findall(r"[-+]?\d*\.\d+|\d+", rms_level_line[0])[0])
                        noise_floor_db = float(re.findall(r"[-+]?\d*\.\d+|\d+", noise_floor_line[0])[0])

                        acx_peak = "Compliant" if -23.0 <= peak_level_db <= -3.0 else "Non-Compliant"
                        acx_rms = "Compliant" if -23.0 <= rms_level_db <= -18.0 else "Non-Compliant"
                        acx_noise_floor = "Compliant" if noise_floor_db < -60.0 else "Non-Compliant"
                        
                        output_line = f"{file_path}|{peak_level_db:.2f}|{rms_level_db:.2f}|{noise_floor_db:.2f}|{acx_peak}|{acx_rms}|{acx_noise_floor}"
                        print(output_line)
                    else:
                        print(f"Error: Unable to extract audio levels for file: {file_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing file: {file_path}")
                    print(f"Error message: {e.output.decode()}")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")

# Usage example
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_acx_check.py <source_directory>")
        sys.exit(1)

    directory_path = sys.argv[1]
    check_audio_levels(directory_path)
