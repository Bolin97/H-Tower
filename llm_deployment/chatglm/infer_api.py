import re
from typing import List

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# E:\\ICES backup\\rpapool\\ChatGLM-Tuning-1\\output
app = FastAPI()


model = AutoModel.from_pretrained("E:\\checkpoint\\chatglm", trust_remote_code=True, load_in_8bit=True,
                                  device_map='auto', revision="")
tokenizer = AutoTokenizer.from_pretrained("E:\\checkpoint\\chatglm", trust_remote_code=True, revision="")

model = PeftModel.from_pretrained(model, "E:\\ICES backup\\rpapool\\ChatGLM-Tuning-1\\output")


@app.post("/predict")
def pred_chat(user_msg: str,
              history: List[List[str]],
              ):
    input_text = user_msg
    ids = tokenizer.encode(input_text)
    input_ids = torch.LongTensor([ids])
    input_ids = input_ids.to(CUDA_DEVICE)
    out = model.generate(
        input_ids=input_ids,
        max_length=1920,
        do_sample=False,
        temperature=0
    )
    out_text = tokenizer.decode(out[0])
    pred = "[" + re.search(r'\s\[(.*)', out_text).group(1).strip()
    torch_gc()
    return {"response": pred, "history": history}


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8081, workers=1)
