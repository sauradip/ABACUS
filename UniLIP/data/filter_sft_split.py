import json
import os

def filter_split(sft_json, split_json, split_name, output_json):
    with open(sft_json, 'r') as f:
        sft_data = json.load(f)
    
    with open(split_json, 'r') as f:
        splits = json.load(f)
        
    test_images = set(splits[split_name])
    
    filtered_data = [item for item in sft_data if os.path.basename(item["image"]) in test_images]
    
    with open(output_json, 'w') as f:
        json.dump(filtered_data, f, indent=4)
        
    print(f"Filtered {len(filtered_data)} entries for split '{split_name}' and saved to {output_json}")

if __name__ == "__main__":
    filter_split(
        "/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_understanding_sft.json",
        "/projects/u6bl/myprojects/Datasets/FSC-147/Train_Test_Val_FSC_147.json",
        "test",
        "/projects/u6bl/myprojects/Datasets/FSC-147/fsc147_understanding_test.json"
    )
