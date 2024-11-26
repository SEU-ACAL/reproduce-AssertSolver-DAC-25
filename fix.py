from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "1412312anonymous/AssertSolver"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
# Here is a demo input
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