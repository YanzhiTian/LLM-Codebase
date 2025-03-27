from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TrainingArguments, set_seed
from qwen_vl_utils import process_vision_info
from trl import SFTTrainer, SFTConfig
import json
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
        with open("/home/yztian/Arial/train/de/subtitle.txt", "r") as f:
            for l in f:
                captions.append(l.strip())
        img_paths = []
        imgs = sorted(os.listdir("/home/yztian/Arial/train/de/image"), key=lambda x: int(x.split(".")[0]))
        for img in imgs:
            img_paths.append(os.path.join("/home/yztian/Arial/train/de/image", img))
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

def main(args: TrainArgs):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # peft_model = get_peft_model(model, peft_config)
    # peft_model.print_trainable_parameters()

    # dataset_id = "HuggingFaceM4/ChartQA"
    # train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train[:10%]", "val[:10%]", "test[:10%]"])
    train_dataset = dataset_preprocess()
    training_args = SFTConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        fp16=True,

        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=1,

        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",
        save_strategy="epoch",
        save_steps=1,

        optim="adamw_torch",
        lr_scheduler_type="linear",
        learning_rate=5e-5,

        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        remove_unused_columns=False,
    )
    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        # breakpoint()
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing
        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        # if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        #     image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        # else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        return batch  # Return the prepared batch


    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        args=training_args,
        peft_config=peft_config,
    )

    if trainer.accelerator.is_local_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name)

    trainer.train()


if __name__ == "__main__":
    set_seed(42)
    main(TrainArgs)

