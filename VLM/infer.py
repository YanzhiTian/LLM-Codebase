from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TrainingArguments, set_seed
from qwen_vl_utils import process_vision_info
from trl import SFTTrainer, SFTConfig
import json
import torch
# from datasets import Dataset
from peft import LoraConfig, get_peft_model
import wandb
import os
from torch.utils.data import Dataset


class TrainArgs:
    model_path = "/home/yztian/Qwen2.5-VL-7B-Instruct"
    output_dir = "/home/yztian/VLM/model"

    gradient_accumulation_steps=16
    per_device_train_batch_size=4

    wandb_project = "sft-vlm"
    run_name = "iimt30k-qwen2.5vl-7b"
"""
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": },
            {"type": "text", "text": },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": },
        ],
    }
]
"""

class dataset_preprocess(Dataset):
    def __init__(self):
        captions = []
        with open("/home/yztian/Arial/test_flickr/de/subtitle.txt", "r") as f:
            for l in f:
                captions.append(l.strip())
        img_paths = []
        imgs = sorted(os.listdir("/home/yztian/Arial/test_flickr/de/image"), key=lambda x: int(x.split(".")[0]))
        for img in imgs:
            img_paths.append(os.path.join("/home/yztian/Arial/test_flickr/de/image", img))
        self.data = []
        for caption, img_path in zip(captions, img_paths):
            self.data.append(
                [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": caption},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "What is the text in the image?"},
                    ],
                }]
            )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

@torch.no_grad()
def main():
    # model_path = "/home/yztian/Qwen2.5-VL-7B-Instruct"
    model_path = "/home/yztian/VLM/model/qwen2.5-vl-7b-lora"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    # adapter_path = "/home/yztian/VLM/model/checkpoint-393"
    # model.load_adapter(adapter_path)
    model.cuda()

    dataset = dataset_preprocess()
    for data in dataset:
        text = processor.apply_chat_template(
            data, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(data)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

if __name__ == "__main__":
    main()

