import torch
import torch.nn as nn


class Loss(object):
    """
    所有loss的类父类
    """
    def __call__(self, model, inputs, training_args, return_outputs=False):
        """
        todo label smoothing
        用于计算loss。
        看源码发现，return_outputs=True为train时调用，return_outputs=False为eval和predict调用
        :param model: 模型
        :param inputs: 模型输入，dict
        :param training_args: 训练配置参数
        :param return_outputs:是否返回模型的输出
        :return:
        """
        raise NotImplemented


class TargetLMLoss(Loss):
    '''

    '''
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        # 交叉熵：pre = [B,S],label=[B]
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs, training_args, return_outputs=False):
        '''
        1: 输入:每条样本经过collator后的input_ids/attention_mask/target_mask 都为[B,S]
        2: 输入经过model解码后，形成
        '''
        # ------输入三要素[B,S]-----
        # 输入的token ids，形状为[B, S]。
        input_ids = inputs['input_ids']
        # 注意力掩码，形状为[B, S]。自有元素为1，填补的为0
        attention_mask = inputs['attention_mask']
        # 目标掩码，形状为[B, S]，用于指示哪些位置是目标。设置label计算loss的位置
        # ！！！只有QA中的A及后面的</s>设置为1
        target_mask = inputs['target_mask']


        # ------模型前馈预测 ------
        # [B,S]---[B,S,E]---[B, S, C]，其中E是嵌入维度，C是词汇表大小
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        # [B,S]
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)


        # -----(！！！重点！！！)对logits/label进行错位，logits截掉最后一个eos_token_id，label截掉第一个。目的达到输出/预测位置的对应
        # contiguous深拷贝
        # 对logits进行处理以匹配labels的形状
        # 需要移除序列中的最后一个token，因为我们是预测下一个token
        # logits的形状从[B, S, C]变为[B, S-1, C]
        shift_logits = logits[..., :-1, :].contiguous()

        # 对labels进行处理以匹配logits的形状
        # labels的形状从[B, S]变为[B, S-1]，label从第2位开始
        # 也就是logit第一位为第一个词的输出，label第一位为第二个词的真实token_id
        shift_labels = labels[..., 1:].contiguous()


        # -----将logits和labels展平，以便可以用交叉熵损失计算
        # shift_logits的形状从[B, S-1, C]变为[B*(S-1), C]
        # shift_labels的形状从[B, S-1]变为[B*(S-1)]
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss
