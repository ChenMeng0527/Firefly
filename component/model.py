import transformers
from typing import Tuple, Union
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from component.loss import TargetLMLoss
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BloomForCausalLM(transformers.BloomForCausalLM):
    """
    继承自BloomForCausalLM，区别在于只计算target部分的loss

    返回:
        - Tuple[torch.Tensor] 或 CausalLMOutputWithCrossAttentions:
        根据`return_dict`的值返回不同的数据结构。当`return_dict=False`时，返回一个元组。
        当`return_dict=True`时，返回一个`CausalLMOutputWithCrossAttentions`对象。
    """

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        #  用于语言模型的标签。请注意，标签在模型内部是**移位的**，即你可以设置`labels = input_ids`。
        #  索引在`[-100, 0, ..., config.vocab_size]`之间选择。所有设置为`-100`的标签都会被忽略（掩蔽），
        #  仅为`[0, ..., config.vocab_size]`内的标签计算损失。
        labels=None,  # (batch_size, sequence_length)

        # 一个二进制掩码，指定哪些位置的损失应该被计算。 (batch_size, sequence_length)
        target_mask=None,

        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        return_loss=False,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 根据配置决定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用父类的transformer模块进行前向传播
        transformer_outputs = self.transformer(
                                                input_ids,  # (batch_size, sequence_length)
                                                past_key_values=past_key_values,  # List[Tuple[torch.Tensor]] 由键值对组成的列表，用于解码器的交叉注意力层
                                                attention_mask=attention_mask,  # (batch_size, sequence_length) 其中1表示注意，0表示不注意
                                                position_ids=position_ids,  # (batch_size, sequence_length)
                                                head_mask=head_mask,  # `(num_heads,)` or `(num_layers, num_heads)`
                                                inputs_embeds=inputs_embeds,  # 代替输入ID序列的嵌入输入。
                                                use_cache=use_cache,  # 是否使用缓存机制。
                                                output_attentions=output_attentions,  # 是否返回注意力权重。
                                                output_hidden_states=output_hidden_states,  # 是否返回隐藏状态。
                                                return_dict=return_dict,  # 是否以`transformers.file_utils.ModelOutput`对象的形式返回输出。
                                            )
        # 获取transformer模块的输出，即隐藏状态
        hidden_states = transformer_outputs[0]

        # 将隐藏状态通过语言模型的头部（lm_head）得到logits（未归一化的概率分布）
        lm_logits = self.lm_head(hidden_states)

        loss = None
        # 如果是需要输出loss
        # 如果需要返回损失，并且提供了目标掩码，则计算损失
        if return_loss:
            # 创建损失函数实例，忽略索引为填充token ID的损失
            loss_fn = TargetLMLoss(ignore_index=self.config.pad_token_id)
            # 计算损失，仅考虑目标掩码指定的部分
            loss = loss_fn(lm_logits, input_ids, target_mask)

        # 如果不返回字典形式的输出，则将输出组合成一个元组
        if not return_dict:
            # 将logits和其他transformer输出（不包括隐藏状态和注意力权重）组合起来
            output = (lm_logits,) + transformer_outputs[1:]
            # 如果有损失，则将损失添加到元组的开头
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典形式的输出，则构造一个CausalLMOutputWithCrossAttentions对象
        return CausalLMOutputWithCrossAttentions(
                                                loss=loss,
                                                logits=lm_logits,
                                                past_key_values=transformer_outputs.past_key_values,
                                                hidden_states=transformer_outputs.hidden_states,
                                                attentions=transformer_outputs.attentions,
                                                )

