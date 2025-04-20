import os
import subprocess

def convert_video(input_path, temp_output_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        temp_output_path
    ]
    print(f"Converting: {input_path} -> {temp_output_path}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("ffmpeg stdout:", result.stdout)
        if result.stderr:
            print("ffmpeg stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}:\n", e.stderr)
        return
    if os.path.exists(temp_output_path):
        print("Conversion successful.  Replacing original file.")
        os.remove(input_path)
        os.rename(temp_output_path, input_path)
    else:
        print("Converted file not found.")

def traverse_and_convert(root_dir="examples"):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".mp4"):
                input_path = os.path.join(dirpath, file)
                base, ext = os.path.splitext(input_path)
                temp_output = f"{base}_converted{ext}"
                convert_video(input_path, temp_output)

if __name__ == "__main__":
    traverse_and_convert()
