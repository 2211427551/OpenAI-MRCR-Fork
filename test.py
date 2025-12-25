from huggingface_hub import hf_hub_download
import pandas as pd
from openai import OpenAI
import json
from difflib import SequenceMatcher
import tiktoken
import os
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

# ================= Configuration =================
API_BASE_URL = "https://api.deepseek.com/v1"  
API_KEY = ""
MODEL_NAME = "deepseek-chat" 
MAX_CONTEXT_WINDOW = 128000 
LOG_FILE = "evaluation_log_deepseek_v32_chat_below_16000.jsonl" 
LOG_FULL_CONTEXT = False 
MAX_CONCURRENCY = 10 
NEEDLE_COUNT = 2
# ===============================================

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Tokenizer init
try:
    enc = tiktoken.get_encoding("o200k_base")
except:
    enc = tiktoken.get_encoding("cl100k_base")

log_lock = threading.Lock()


def get_processed_indices(log_file_path):
    """
    Read the log file and return a set of all processed dataset_indices (including successes and errors).
    """
    processed = set()
    if not os.path.exists(log_file_path):
        return processed

    print(f"🔄 Log file detected. Scanning checkpoints: {log_file_path} ...")
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)

                # [Safety Check] If log files are mixed (e.g., filename wasn't changed), check if needle count matches.
                # If needle_count in log differs from current config, ignore it to avoid confusion.
                if "needle_count" in data and data["needle_count"] != NEEDLE_COUNT:
                    continue  # Do not include indices from other needle counts

                if "dataset_index" in data:
                    processed.add(data["dataset_index"])
            except json.JSONDecodeError:
                continue
    return processed


def write_log(data: dict):
    with log_lock:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception:
            pass


def grade(response, answer, random_string_to_prepend) -> float:
    if not response:
        return .0

    if not response.startswith(random_string_to_prepend):
        return 0

    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def n_tokens(messages: list[dict]) -> int:
    return sum([len(enc.encode(m["content"])) for m in messages])


def process_single_row(index, row):
    try:
        messages = json.loads(row["prompt"])
        token_count = n_tokens(messages)

        # Length filtering
        if token_count > MAX_CONTEXT_WINDOW:
            return None

        start_time = datetime.datetime.now()
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=65536
        )
        end_time = datetime.datetime.now()
        latency = (end_time - start_time).total_seconds()

        response = completion.choices[0].message.content
        score = grade(response, row["answer"], row["random_string_to_prepend"])

        if LOG_FULL_CONTEXT:
            log_input = messages
        else:
            log_input = messages[-1] if messages else "No content"

        log_entry = {
            "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_index": index,
            "needle_count": NEEDLE_COUNT,  # [Added] Record needle count
            "token_count": token_count,
            "latency_seconds": latency,
            "model": MODEL_NAME,
            "input": log_input,
            "model_response": response,
            "ground_truth": row["answer"],
            "score": score
        }
        write_log(log_entry)

        console_msg = (
            f"[Done] Index: {index:<4} | "
            f"Len: {token_count:<7} | "
            f"Time: {latency:<5.1f}s | "
            f"Score: {score:.4f}"
        )
        return {"status": "success", "console_msg": console_msg}

    except Exception as e:
        # Log error
        write_log({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_index": index,
            "needle_count": NEEDLE_COUNT,  # [Added] Record needle count on error
            "error": str(e)
        })

        err_str = str(e)[:30] + "..."
        console_msg = f"[Error] Index: {index:<4} | Exception: {err_str}"
        return {"status": "error", "console_msg": console_msg}


if __name__ == "__main__":
    print(f"📥 Loading dataset (Needle Count: {NEEDLE_COUNT})...")

    # [Modified] Dynamically generate HuggingFace file paths based on NEEDLE_COUNT
    # 2needle -> 2needle/2needle_0.parquet
    # 4needle -> 4needle/4needle_0.parquet
    # 8needle -> 8needle/8needle_0.parquet
    folder_prefix = f"{NEEDLE_COUNT}needle"
    filename_0 = f"{folder_prefix}/{folder_prefix}_0.parquet"
    filename_1 = f"{folder_prefix}/{folder_prefix}_1.parquet"

    try:
        file_0 = hf_hub_download(repo_id="openai/mrcr", filename=filename_0, repo_type="dataset")
        file_1 = hf_hub_download(repo_id="openai/mrcr", filename=filename_1, repo_type="dataset")

        dataset = pd.concat([pd.read_parquet(file_0), pd.read_parquet(file_1)], ignore_index=True)
    except Exception as e:
        print(f"❌ Data load failed (Please check if NEEDLE_COUNT is correct): {e}")
        exit(1)

    # 1. Get processed indices
    processed_indices = get_processed_indices(LOG_FILE)
    total_count = len(dataset)

    # 2. Filter dataset
    if processed_indices:
        # Use ~dataset.index.isin(...) for inverse selection
        dataset = dataset[~dataset.index.isin(processed_indices)]
        print(f"⏭️  Resuming: Skipped {len(processed_indices)} records. Remaining: {len(dataset)} to process.")
    else:
        print(f"🆕 Fresh start: Total {total_count} records.")

    if dataset.empty:
        print("🎉 All data processed!")
        exit(0)

    # Get terminal width
    try:
        term_width = shutil.get_terminal_size().columns
    except:
        term_width = 100
    safe_width = max(term_width - 2, 80)

    print(f"🚀 Starting evaluation: {MODEL_NAME} | Concurrency: {MAX_CONCURRENCY}")
    print("-" * safe_width)

    # Thread pool
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        futures = {executor.submit(process_single_row, idx, row): idx for idx, row in dataset.iterrows()}

        pbar = tqdm(
            total=len(futures),
            desc="Progress",
            ncols=safe_width,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

        for future in as_completed(futures):
            result = future.result()
            pbar.update(1)

            if result is None:
                continue

            if result.get("console_msg"):
                pbar.write(result["console_msg"])

        pbar.close()

    print("\n✅ All tasks completed.")
