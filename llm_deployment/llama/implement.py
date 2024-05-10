import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import json
import uvicorn
from fastapi import FastAPI

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


model_id = "/data/gongbu/LLMCraft/models/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto',
                                         torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "/data/gongbu/internlm-sft/ouput")
app = FastAPI()


def extract_info(response):
    # 提取key和value的值
    key_start_index = response.find("{'key': '") + len("{'key': '")
    key_end_index = response.find("', 'value'")
    key = response[key_start_index:key_end_index]

    value_start_index = response.find("'value': '") + len("'value': '")
    value_end_index = response.find("'}]", key_end_index)
    value = response[value_start_index:value_end_index]
    prediction = {
        'key': key,
        'value': value
    }
    return prediction


@app.post("/predict")
def pred_chat(user_msg: str):
    with torch.no_grad():
        input_text = user_msg + "\n\n"
        model_input = tokenizer(input_text, return_tensors="pt").to("cuda")
        model.eval()
        response = tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True)
        prediction = extract_info(response)
        torch_gc()
        return prediction


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8081, workers=1)
