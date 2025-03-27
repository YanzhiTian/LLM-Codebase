
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


BASE_MODEL = "/home/yztian/Qwen2.5-VL-7B-Instruct"
LORA_WEIGHTS = "/home/yztian/VLM/model/checkpoint-393"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL)
processor = AutoProcessor.from_pretrained(BASE_MODEL)

model = PeftModel.from_pretrained(model, LORA_WEIGHTS)

merged_model = model.merge_and_unload()

merged_model.save_pretrained("/home/yztian/VLM/model/qwen2.5-vl-7b-lora")
processor.save_pretrained("/home/yztian/VLM/model/qwen2.5-vl-7b-lora")
