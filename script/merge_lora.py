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
    # model_name_or_path = 'baichuan-inc/baichuan-7B'
    model_name_or_path = "/root/.cache/huggingface/hub/models--baichuan-inc--Baichuan-13B-Chat/snapshots/b3ca596c403e84a72476349de5cb2a03a522c368"

    # baichuan-7b经过qlora-sft后 地址
    # adapter_name_or_path = 'YeungNLP/firefly-baichuan-7b-qlora-sft'
    adapter_name_or_path = "/home/hadoop/Firefly/output/firefly-baichuan-13b/final"

    # 合并后地址
    save_path = '/home/hadoop/Firefly/out_meger_model/lora_model/youshu-baichuan-13b-qa_qlora-sft-merge'


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
    # model = PeftModel.from_pretrained(model, adapter_name_or_path)  # 原代码
    model = PeftModel.from_pretrained(model, adapter_name_or_path, offload_folder='offload' )
    # 将两个模型权重进行合并
    model = model.merge_and_unload()

    # token / model进行保存
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
