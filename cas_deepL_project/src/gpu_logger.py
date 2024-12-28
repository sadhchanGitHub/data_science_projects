import os
import subprocess
import time

def log_gpu_usage(log_dir="logs", interval=10):
    print("GPU logging process started.")  # Debugging message
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gpu_usage_{int(time.time())}.log")

    with open(log_file, "a") as f:
        try:
            while True:
                gpu_stats = subprocess.run(
                    ["nvidia-smi"], stdout=subprocess.PIPE, text=True, timeout=5
                )
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(gpu_stats.stdout)
                f.write("\n" + "=" * 80 + "\n")
                time.sleep(interval)
        except subprocess.TimeoutExpired:
            f.write("nvidia-smi command timed out.\n")
        except KeyboardInterrupt:
            print(f"Logging stopped. GPU usage log saved to: {log_file}")
        except Exception as e:
            f.write(f"Error while logging GPU usage: {e}\n")
