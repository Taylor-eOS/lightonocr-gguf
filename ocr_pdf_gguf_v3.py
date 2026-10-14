import os
import io
import json
import time
import base64
import subprocess
import urllib.error
import urllib.request
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from PIL import Image, ImageEnhance, ImageFilter

try:
    import pymupdf as fitz
except ImportError:
    import fitz

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
pdf_dpi = 400
max_image_edge = 2048
separator = "\n\n"
ocr_instruction = ""
use_preprocessing = True
stop_file = "stop"
server_url = f"http://{server_host}:{server_port}"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
server_process = None
server_log = None
doc = None

def resolve_model_file(repo, filename):
    try:
        return hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"{filename} is not cached, downloading.")
        return hf_hub_download(repo_id=repo, filename=filename)

def server_is_ready():
    try:
        with opener.open(server_url + "/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False

def print_log_tail(line_count=15):
    try:
        with open(server_log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return
    print(f"Last lines of {server_log_path}:")
    for line in lines[-line_count:]:
        print(line)

def start_server():
    global server_process, server_log
    if server_is_ready():
        print(f"Something is already listening on port {server_port}. Stop it or change server_port.")
        return False
    try:
        model_path = resolve_model_file(repo_id, model_filename)
        mmproj_path = resolve_model_file(mmproj_repo_id, mmproj_filename)
    except Exception as e:
        print(f"Could not fetch model files: {e}")
        return False
    command = server_command + ["-m", model_path, "--mmproj", mmproj_path, "--host", server_host, "--port", str(server_port), "-c", str(context_size), "-b", str(batch_size), "-ub", str(batch_size), "-np", "1", "--jinja", "--no-webui"]
    if n_threads > 0:
        command += ["-t", str(n_threads)]
    print("Starting llama-server")
    server_log = open(server_log_path, "w", encoding="utf-8")
    try:
        server_process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        print(f"Could not run '{server_command[0]}'. Install llama.cpp or set server_command to the full path of llama-server.")
        return False
    deadline = time.time() + server_start_timeout
    while time.time() < deadline:
        if server_process.poll() is not None:
            print(f"llama-server exited with code {server_process.returncode}")
            print_log_tail()
            return False
        if server_is_ready():
            print("Server ready")
            return True
        time.sleep(1)
    print("Timed out waiting for llama-server")
    print_log_tail()
    return False

def stop_server():
    global server_process, server_log
    if server_process is not None:
        server_process.terminate()
        try:
            server_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        server_process = None
    if server_log is not None:
        server_log.close()
        server_log = None

def encode_image(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def clean_text(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()

def request_ocr(image_b64):
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}}]
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
        with opener.open(request, timeout=request_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Server returned {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from e
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        print(f"Warning: output reached the {max_new_tokens} token limit")
    return choice["message"]["content"] or ""

def ocr_image(image):
    return clean_text(request_ocr(encode_image(image)))

def resolve_pdf_path(user_input):
    ending = '.pdf'
    if os.path.isfile(user_input):
        return user_input
    if not user_input.lower().endswith(ending):
        pdf_path = user_input + ending
        if os.path.isfile(pdf_path):
            return pdf_path
    return None

def get_page_image(page, dpi):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_data = pix.tobytes("ppm")
    image = Image.frombytes("RGB", [pix.width, pix.height], img_data)
    current_max = max(image.width, image.height)
    if current_max > max_image_edge:
        scale = max_image_edge / current_max
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def format_duration(seconds):
    seconds = max(seconds, 0.0)
    total_minutes = seconds / 60.0
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    if total_minutes < 1:
        return "1 minute"
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0 or hours == 0:
        time_parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    return ' '.join(time_parts)

def ask_for_pdf_path():
    user_input = input('PDF file name: ').strip() or default_input_pdf
    file_path = resolve_pdf_path(user_input)
    if not file_path:
        print(f"Error: File '{user_input}' not found.")
        return None
    print(f"Using PDF file: {file_path}")
    return file_path

def open_pdf_document(file_path):
    global doc
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return False
    if len(doc) == 0:
        print("No pages detected in file.")
        doc.close()
        return False
    print(f"The PDF has {len(doc)} page{'s' if len(doc) > 1 else ''}.")
    return True

def ask_for_start_page():
    total_pages = len(doc)
    while True:
        start_input = input(f"Enter starting page number to {total_pages} (1): ").strip()
        if not start_input:
            return 1
        try:
            proposed = int(start_input)
            if 1 <= proposed <= total_pages:
                return proposed
            print(f"Page number must be between 1 and {total_pages}. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def preprocess_for_ocr(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    x0 = int(w * 0.45)
    x1 = int(w * 0.55)
    y_positions = [int(h * p) for p in (0.015, 0.03, 0.045)]
    samples = []
    step = max(1, (x1 - x0) // 10)
    for y in y_positions:
        yy = min(y, h - 1)
        for x in range(x0, x1, step):
            samples.append(image.getpixel((x, yy)))
    if not samples:
        bg_r, bg_g, bg_b = 255, 255, 255
    else:
        rs = sorted(p[0] for p in samples)
        gs = sorted(p[1] for p in samples)
        bs = sorted(p[2] for p in samples)
        mid = len(rs) // 2
        bg_r, bg_g, bg_b = rs[mid], gs[mid], bs[mid]
    arr = np.asarray(image).astype(np.float32)
    dr = arr[..., 0] - float(bg_r)
    dg = arr[..., 1] - float(bg_g)
    db = arr[..., 2] - float(bg_b)
    dist = np.sqrt(dr * dr + dg * dg + db * db)
    dist_threshold = 60.0
    mask = dist > dist_threshold
    mask3 = mask[..., None].astype(np.float32)
    cleaned = arr * mask3 + 255.0 * (1.0 - mask3)
    cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)
    gray = Image.fromarray(cleaned).convert("L")
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    garr = np.asarray(gray)
    hist = np.bincount(garr.flatten(), minlength=256).astype(np.float32)
    total = garr.size
    sum_total = (np.arange(256) * hist).sum()
    sumB = 0.0
    wB = 0.0
    max_var = 0.0
    thresh = 128
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) * (mB - mF)
        if var_between > max_var:
            max_var = var_between
            thresh = t
    bw = (garr > thresh).astype(np.uint8) * 255
    final = Image.fromarray(bw).convert("L")
    return final

def process_single_page(page_num, page, outfile, is_first):
    page_start = time.time()
    try:
        image = get_page_image(page, pdf_dpi)
        if use_preprocessing:
            image = preprocess_for_ocr(image)
        text = ocr_image(image)
        if not is_first:
            outfile.write(separator)
        outfile.write(text)
        outfile.flush()
        print(f"Completed {len(text)} characters")
    except Exception as e:
        error_msg = f"Error processing page {page_num}: {str(e)}"
        print(error_msg)
        if not is_first:
            outfile.write(separator)
        outfile.write(f"Page {page_num}\n{error_msg}\n\n")
        outfile.flush()
    return time.time() - page_start

def run_ocr_loop(start_page, output_file):
    total_pages = len(doc)
    num_to_process = total_pages - start_page + 1
    print(f"Processing pages {start_page} to {total_pages} at {pdf_dpi} DPI")
    running_sum = 0.0
    processed = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for idx in range(num_to_process):
            page_num = start_page + idx
            if os.path.exists(stop_file):
                print("Stop file detected")
                os.remove(stop_file)
                break
            if server_process.poll() is not None:
                print(f"llama-server stopped unexpectedly, see {server_log_path}")
                break
            print(f"Progress {idx + 1}/{num_to_process}")
            duration = process_single_page(page_num, doc.load_page(page_num - 1), outfile, idx == 0)
            running_sum += duration
            processed += 1
            avg = running_sum / processed
            print(f"Page {page_num} took {duration:.0f} seconds (average {avg:.0f} per page)")
            remaining = num_to_process - processed
            if remaining > 0:
                print(f"Estimated time for remaining {remaining} pages: {format_duration(avg * remaining)}")
    return processed, total_pages

def process_pdf():
    file_path = ask_for_pdf_path()
    if not file_path:
        return
    if not open_pdf_document(file_path):
        return
    try:
        start_page = ask_for_start_page()
        if not start_server():
            return
        output_file = os.path.splitext(file_path)[0] + '.txt'
        processed_count, total_pages = run_ocr_loop(start_page, output_file)
        print(f"All results saved to {output_file}")
        print(f"Processed {processed_count} page{'s' if processed_count != 1 else ''} from {start_page} to {total_pages}")
    finally:
        stop_server()
        doc.close()

if __name__ == "__main__":
    process_pdf()
