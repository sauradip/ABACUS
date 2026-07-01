import json
import os
import re
from pathlib import Path
from collections import Counter

# Paths - Update these to your local environment
SFT_FILE = "outputs/adaptive_hf_multi_image_count_sft_fsc147_train/train_messages.jsonl"
GRPO_FILE = "outputs/unilip_grpo_adaptive/hf_native_multi_image_grpo_train_dense.jsonl"

def check_sft_format(file_path):
    print(f"\n--- 🔍 Auditing SFT File: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"❌ File not found.")
        return

    errors = []
    grid_patterns = Counter()
    total = 0
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            total += 1
            data = json.loads(line)
            
            # 1. Basic Keys
            if "messages" not in data or "id" not in data:
                errors.append(f"Row {i}: Missing 'messages' or 'id'")
                continue
                
            msgs = data["messages"]
            if len(msgs) != 2:
                errors.append(f"Row {i}: Expected 2 messages (user/assistant), found {len(msgs)}")
                
            # 2. User Message Multi-Image Check
            user_content = msgs[0]["content"]
            images = [c for c in user_content if c["type"] == "image"]
            texts = [c for c in user_content if c["type"] == "text"]
            
            if len(images) != 2:
                errors.append(f"Row {i}: Found {len(images)} images in prompt (Expected: 2)")
            
            # 3. Path Validity
            for img in images:
                if not os.path.exists(img["url"]):
                    errors.append(f"Row {i}: Image path does not exist: {img['url']}")
            
            # 4. Adaptive Grid Check
            if texts:
                grid_match = re.search(r'shape of (\d+ \s*\* \s*\d+)', texts[0]["text"])
                if grid_match:
                    grid_patterns[grid_match.group(1)] += 1
                else:
                    errors.append(f"Row {i}: No grid shape instruction found in prompt.")
            else:
                errors.append(f"Row {i}: No text prompt found.")

            # 5. Assistant JSON Strictness
            assistant_text = msgs[1]["content"][0]["text"]
            if not re.match(r'^\{\s*"total_count"\s*:\s*\d+\s*\}$', assistant_text.strip()):
                errors.append(f"Row {i}: Assistant output is not strict JSON: {assistant_text}")

    print(f"✅ Processed {total} rows.")
    print(f"📊 Grid Distributions found: {dict(grid_patterns)}")
    if errors:
        print(f"❌ Found {len(errors)} errors. First 5: {errors[:5]}")
    else:
        print("💎 SFT Schema Check Passed: 100% Valid.")

def check_grpo_format(file_path):
    print(f"\n--- 🔍 Auditing GRPO File: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"❌ File not found.")
        return

    errors = []
    counts = []
    total = 0
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            total += 1
            data = json.loads(line)
            
            # 1. Structure Check
            if not all(k in data for k in ["id", "prompt", "gt_count"]):
                errors.append(f"Row {i}: Missing required GRPO keys (id, prompt, gt_count)")
                continue
                
            # 2. Prompt Check (Must be User-Only)
            if len(data["prompt"]) != 1 or data["prompt"][0]["role"] != "user":
                errors.append(f"Row {i}: GRPO prompt must contain exactly ONE user message.")
            
            # 3. Leakage Check
            prompt_str = json.dumps(data["prompt"])
            if "total_count" in prompt_str:
                errors.append(f"Row {i}: LEAKAGE! Ground truth found in prompt.")

            # 4. Curriculum Filter Check (Assuming Dense Curriculum > 50)
            counts.append(data["gt_count"])
            if data["gt_count"] < 50:
                 errors.append(f"Row {i}: Curriculum failure. Count {data['gt_count']} found in dense file.")

    print(f"✅ Processed {total} rows.")
    if counts:
        print(f"📊 Count Range: {min(counts)} to {max(counts)} (Mean: {sum(counts)/len(counts):.2f})")
    
    if errors:
        print(f"❌ Found {len(errors)} errors. First 5: {errors[:5]}")
    else:
        print("💎 GRPO Schema Check Passed: 100% Valid.")

if __name__ == "__main__":
    check_sft_format(SFT_FILE)
    check_grpo_format(GRPO_FILE)
