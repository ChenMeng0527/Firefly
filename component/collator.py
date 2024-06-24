from typing import Any, Dict, List
import torch


class SFTDataCollator(object):
    '''
    将batch中的input_ids/attention_mask/target_mask进行处理

    --输入batch
    输入: [{'input_ids':[],'attention_mask':[],'target_mask':[]},
              {},
              {}]
    --处理
        1：padding
            input_ids：用pad_token_id进行填充id
            attention_mask：pad补充0
            target_mask：pad补充0
        2：截断
        3：转tensor

    --返回
        inputs = {
            'input_ids': input_ids_batch,  # input_ids_batch:[[],[],[],[]]
            'attention_mask': attention_mask_batch,  # attention_mask_batch:[[],[],[],[]]
            'target_mask': target_mask_batch  # target_mask_batch:[[],[],[],[]]
        }
    '''

    def __init__(self, tokenizer, max_seq_length):
        # tokenizer
        self.tokenizer = tokenizer
        # 句子最大尺寸
        self.max_seq_length = max_seq_length
        # pad的token_id
        self.pad_token_id = tokenizer.pad_token_id


    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''
        输入: [{'input_ids':[],'attention_mask':[],'target_mask':[]},
              {},
              {}]
        重点：将每条样本中input_ids,attention_mask,target_mask进行处理
        1：padding
            input_ids：用pad_token_id进行填充id
            attention_mask：pad补充0
            target_mask：pad补充0,制作label
        2：截断
        3：转tensor
        '''
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch]

        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length


        # 输入input_ids，attention_mask，target_mask
        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []

        # truncate and padding
        # batch:
        #   [
        #       {'input_ids':[],'attention_mask':[],'target_mask':[]},
        #       {'input_ids':[],'attention_mask':[],'target_mask':[]},
        #       {'input_ids':[],'attention_mask':[],'target_mask':[]}
        #   ]
        for x in batch:

            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']

            # ------------padding--------
            # 需要pad的长度
            padding_len = batch_max_len - len(input_ids)
            # input_ids：用pad_token_id进行填充id
            input_ids = input_ids + [self.pad_token_id] * padding_len
            # attention_mask：pad补充0
            attention_mask = attention_mask + [0] * padding_len
            # target_mask：pad补充0
            target_mask = target_mask + [0] * padding_len

            # ------------根据最长长度进行截断--------
            # truncate
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)


        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        # 模型输入 元素是个list
        # input_ids_batch:[[],[],[],[]]
        # attention_mask_batch:[[],[],[],[]]
        # target_mask_batch:[[],[],[],[]]
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'target_mask': target_mask_batch
        }
        return inputs

