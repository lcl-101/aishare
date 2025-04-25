import re

def seconds_to_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def parse_line(line):
    # 匹配 0.0s-3.0s: 内容
    match = re.match(r"([\d.]+)s-([\d.]+)s:\s*(.*)", line)
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        text = match.group(3).strip()
        return start, end, text
    return None

def txt_to_srt(txt_path, srt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entries = []
    for line in lines:
        parsed = parse_line(line)
        if parsed:
            entries.append(parsed)
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(entries, 1):
            f.write(f"{idx}\n")
            f.write(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n")
            f.write(f"{text}\n\n")

if __name__ == "__main__":
    txt_to_srt("input.txt", "output.srt")