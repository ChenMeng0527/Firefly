from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
使用该脚本，将lora的权重合并大base model中
"""

def merge_lora_to_base_model():
    '''

    '''
    # baichuan-7B模型
    model_name_or_path = 'baichuan-inc/baichuan-7B'
    # baichuan-7b经过qlora-sft后 地址
    adapter_name_or_path = 'YeungNLP/firefly-baichuan-7b-qlora-sft'
    # 合并后地址
    save_path = 'checkpoint/firefly-baichuan-7b-qlora-sft-merge'


    # baichuan-7B模型的token
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载baichuan-7B模型
    model = AutoModelForCausalLM.from_pretrained(
                                                model_name_or_path,
                                                trust_remote_code=True,
                                                low_cpu_mem_usage=True,
                                                torch_dtype=torch.float16,
                                                device_map='auto'
                                            )
    # peft将 "baichuan-7B模型参数" 与 "经过qlora-sft后参数" 进行合并
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    # 将两个模型权重进行合并
    model = model.merge_and_unload()

    # token / model进行保存
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
