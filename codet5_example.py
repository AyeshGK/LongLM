# Description: Example usage of CodeT5Model and CustomCodeT5Model
import warnings
warnings.filterwarnings("ignore")

from modify_utils import modify_method_of_instance
from functools import partial
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Model, T5Config
from codet5_self_extend_patch import CustomCodeT5Model

# original_llama_forward = LlamaAttention.forward

original_t5_forward = T5Model.forward
self_extend_forward = partial(CustomCodeT5Model.self_extend_forwar, group_size_1=8, group_size_2=1024)


# model_path = 'meta-llama/Llama-2-7b-chat-hf'
model_checkpoint = 'codet5-base'
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


config = T5Config.from_pretrained(model_checkpoint)
model = CustomCodeT5Model(config)

model.eval()

prompt = "def foo(x):\n    return x + 1\n\nprint(foo(2))"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print( "-----------------------------------" )

modify_method_of_instance(model, "T5Model", "forward", original_t5_forward)
tokens = model.generate(input_ids, max_new_tokens=6)
answer= "Original:  [" + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
answer = answer.replace("\n", "\\n")
print( answer )


# for line in open("passkey_examples_5k.jsonl", "r"):
#     example = json.loads(line)
#     prompt_postfix = "What is the pass key? The pass key is "
#     prompt = example["input"] + prompt_postfix
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     print( "-----------------------------------" )
#     print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
#     print( "Passkey target:", example["target"] )


#     modify_method_of_instance(model, "LlamaAttention", "forward", original_llama_forward)
#     tokens = model.generate(input_ids, max_new_tokens=6)
#     answer= "Llama2:     [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
#     answer = answer.replace("\n", "\\n")
#     print( answer )


#     modify_method_of_instance(model, "LlamaAttention", "forward", self_extend_forward)
#     tokens = model.generate(input_ids, max_new_tokens=6)
#     answer= "SelfExtend: [" + prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)  + "]"
#     answer = answer.replace("\n", "\\n")
#     print( answer )
#     print( "-----------------------------------\n" )













# # Example usage:
# config = CodeT5Config.from_pretrained("microsoft/codebert-base")
# model = CustomCodeT5Model(config)
# input_ids = torch.tensor([[1, 2, 3, 4, 5]])
# outputs = model(input_ids)