import os
import io
import json
import time
import base64
import subprocess
import urllib.error
import urllib.request
import numpy as np
import pymupdf
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from PIL import Image, ImageEnhance, ImageFilter

repo_id = "mradermacher/LightOnOCR-3-4B-i1-GGUF"
model_filename = "LightOnOCR-3-4B.i1-Q4_K_M.gguf"
mmproj_repo_id = "mradermacher/LightOnOCR-3-4B-GGUF"
mmproj_filename = "LightOnOCR-3-4B.mmproj-f16.gguf"
server_command = ["llama/llama-server"]
server_host = "127.0.0.1"
server_port = 8089
server_log_path = "llama_server.log"
server_start_timeout = 300
request_timeout = 3600
n_threads = 0
context_size = 12288
batch_size = 2048
max_new_tokens = 4096
temperature = 0.2
top_p = 0.9
default_input_pdf = "input.pdf"
max_image_edge = 2048
separator = "\n\n"
ocr_instruction = ""
use_preprocessing = True
stop_file = "stop"
server_url = f"http://{server_host}:{server_port}"
server_process = None
doc = None

def get_model_file(repo, filename):
    try:
        return hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"Downloading {filename}")
        return hf_hub_download(repo_id=repo, filename=filename)

def server_is_ready():
    try:
        with urllib.request.urlopen(server_url + "/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False

def start_server():
    global server_process
    model_path = get_model_file(repo_id, model_filename)
    mmproj_path = get_model_file(mmproj_repo_id, mmproj_filename)
    command = server_command + ["-m", model_path, "--mmproj", mmproj_path, "--host", server_host, "--port", str(server_port), "-c", str(context_size), "-b", str(batch_size), "-ub", str(batch_size), "-np", "1", "--jinja", "--no-webui"]
    if n_threads > 0:
        command += ["-t", str(n_threads)]
    print("Starting llama-server")
    server_process = subprocess.Popen(command, stdout=open(server_log_path, "w"), stderr=subprocess.STDOUT)
    deadline = time.time() + server_start_timeout
    while time.time() < deadline:
        if server_process.poll() is not None:
            print(f"llama-server exited with code {server_process.returncode}, see {server_log_path}")
            return False
        if server_is_ready():
            print("Server ready")
            return True
        time.sleep(1)
    print(f"Timed out waiting for llama-server, see {server_log_path}")
    return False

def stop_server():
    if server_process:
        server_process.terminate()
        server_process.wait()

def encode_image(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def clean_text(text):
    return "\n".join(line for line in text.splitlines() if line.strip()).strip()

def request_ocr(image):
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + encode_image(image)}}]
    if ocr_instruction:
        content.append({"type": "text", "text": ocr_instruction})
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    request = urllib.request.Request(server_url + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Server returned {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from e
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        print(f"Warning: output reached the {max_new_tokens} token limit")
    return choice["message"]["content"] or ""

def get_page_image(page):
    zoom = max_image_edge / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def preprocess_for_ocr(image):
    arr = np.asarray(image).astype(np.float32)
    h, w = arr.shape[:2]
    strip = arr[int(h * 0.015):int(h * 0.045) + 1, int(w * 0.45):int(w * 0.55)]
    background = np.median(strip.reshape(-1, 3), axis=0)
    ink = np.linalg.norm(arr - background, axis=2) > 60.0
    cleaned = np.where(ink[..., None], arr, 255.0).astype(np.uint8)
    gray = Image.fromarray(cleaned).convert("L").filter(ImageFilter.MedianFilter(size=3))
    gray = np.asarray(ImageEnhance.Contrast(gray).enhance(1.3))
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    w0 = np.cumsum(hist)
    w1 = gray.size - w0
    s0 = np.cumsum(hist * np.arange(256))
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    threshold = np.argmax(w0 * w1 * (m0 - m1) ** 2)
    return Image.fromarray(((gray > threshold) * 255).astype(np.uint8))

def ocr_page(page_num):
    try:
        image = get_page_image(doc.load_page(page_num - 1))
        if use_preprocessing:
            image = preprocess_for_ocr(image)
        return clean_text(request_ocr(image))
    except Exception as e:
        error_msg = f"Error processing page {page_num}: {e}"
        print(error_msg)
        return f"Page {page_num}\n{error_msg}"

def format_duration(seconds):
    hours, minutes = divmod(max(1, round(seconds / 60)), 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"

def resolve_pdf_path(user_input):
    for path in (user_input, user_input + ".pdf"):
        if os.path.isfile(path):
            return path
    return None

def ask_for_pdf_path():
    user_input = input("PDF file name: ").strip() or default_input_pdf
    file_path = resolve_pdf_path(user_input)
    if not file_path:
        print(f"Error: File '{user_input}' not found.")
    return file_path

def ask_for_start_page():
    total_pages = len(doc)
    while True:
        answer = input(f"The PDF has {total_pages} pages. Starting page (1): ").strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= total_pages:
            return int(answer)
        print(f"Enter a number between 1 and {total_pages}.")

def run_ocr_loop(start_page, output_file):
    count = len(doc) - start_page + 1
    started = time.time()
    with open(output_file, "w", encoding="utf-8") as outfile:
        for idx in range(count):
            if os.path.exists(stop_file):
                print("Stop file detected")
                os.remove(stop_file)
                return idx
            print(f"Progress {idx + 1}/{count}")
            page_start = time.time()
            text = ocr_page(start_page + idx)
            if idx:
                outfile.write(separator)
            outfile.write(text)
            outfile.flush()
            average = (time.time() - started) / (idx + 1)
            remaining = count - idx - 1
            print(f"Page {start_page + idx}: {len(text)} characters in {time.time() - page_start:.0f} s, average {average:.0f} s")
            if remaining:
                print(f"Estimated time for remaining {remaining} pages: {format_duration(average * remaining)}")
    return count

def process_pdf():
    global doc
    file_path = ask_for_pdf_path()
    if not file_path:
        return
    doc = pymupdf.open(file_path)
    start_page = ask_for_start_page()
    try:
        if not start_server():
            return
        output_file = os.path.splitext(file_path)[0] + ".txt"
        processed = run_ocr_loop(start_page, output_file)
        print(f"Processed {processed} pages from page {start_page}, saved to {output_file}")
    finally:
        stop_server()
        doc.close()

if __name__ == "__main__":
    process_pdf()
