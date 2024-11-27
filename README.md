# Insights from Rights and Wrongs: A Large Language Model for Solving Assertion Failures in RTL Design
(DAC 2025 submission 2157)



This study introduces **AssertSolver**, the first open-source LLM specifically designed to address assertion failures in RTL design. Leveraging an innovative data augmentation approach to enrich the representation of diverse assertion failure scenarios in the training dataset, and incorporating insights gained from error responses to challenging cases, AssertSolver achieves up to an 11.97% improvement in solving assertion failures compared to OpenAI’s o1-preview.


## Directory Structure

- `testbench`:   Testbench we open-sourced
- `fix.py`:  Script for running repairs based on our released model

## Model Link

The model we open-sourced can is now available for download from [AssertSolver](https://huggingface.co/1412312anonymous/AssertSolver).

## Pre-requisities

Linux operating system

Nvidia A800-SXM4-80GB(or other advanced Nvidia GPUs) and the corresponding [driver](https://www.nvidia.com/en-us/drivers/), CUDA, cudnn

python 3.10~3.12


## Usage

To utilize the model we've released, please follow these instructions:

1. Download this project & `cd AssertSolver`
2. Install the dependencies: `pip3 install -r requirements.txt`
3. Run fix script: `python fix.py`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "1412312anonymous/AssertSolver"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
prompt = "Tell me how to fix the bugs inside: `always(*) // Pretend that this * should be rst`"

messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt ").to(model.device)
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    do_sample=False,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True))
```

# Training Strategy

We trained the model using the following steps:
1. Pretraining: `llamafactory-cli train train_pt.yaml`
2. Supervised Fine-Tuning (SFT): `llamafactory-cli train train_sft.yaml`
3. Learning from Error Responses to Challenging Cases: `llamafactory-cli train train_dpo.yaml`

# Acknowledgement

This project benefits from [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory).Thanks for their wonderful works.