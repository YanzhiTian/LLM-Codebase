from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from trl import apply_chat_template, SFTTrainer
import json
from datasets import load_dataset, Dataset
from peft import LoraConfig
import wandb
import os


class TrainArgs:
    json_path = "/data1/yztian/SFT/grade_school_math/data/train.jsonl"

    model_path = "/data1/yztian/Qwen2.5-3B-Instruct"
    output_dir = "/data1/yztian/SFT/model"

    gradient_accumulation_steps=16
    per_device_train_batch_size=1

    wandb_project = "sft-llm"
    run_name = "gsm8k-3b"


def construct_dataset(file_path, tokenizer):
    with open(file_path, "r") as f:
        data = {
            "prompt": [],
            "completion": []
        }
        prompt_list = []
        completion_list = []
        for l in f:
            q = json.loads(l)["question"]
            a = json.loads(l)["answer"]
            item = {"prompt": [{"role": "user", "content": q}], "completion": [{"role": "assistant", "content": a}]}
            template_dict = apply_chat_template(item, tokenizer)
            prompt_list.append(template_dict["prompt"])
            completion_list.append(template_dict["completion"])
        data["prompt"] = prompt_list
        data["completion"] = completion_list
        dataset = Dataset.from_dict(data)
        return dataset


def main(args: TrainArgs):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = construct_dataset(args.json_path, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    training_args = TrainingArguments(
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
        learning_rate=5e-5
    )

    # if "wandb" in training_args.report_to:
        # wandb.init(project="sft-llm", name="gsm8k-3b-zero3")
        # if args.wandb_project is not None:
            # os.environ["WANDB_PROJECT"] = args.wandb_project

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        train_dataset=dataset
    )

    if trainer.accelerator.is_local_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name)

    trainer.train()


if __name__ == "__main__":
    set_seed(42)
    main(TrainArgs)

