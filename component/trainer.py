import transformers
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers.utils import (
    logging,
)
from typing import Optional
import os
import torch


logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class Trainer(transformers.Trainer):
    """
    主要修改逻辑：通过传入compute_loss，支持自定义loss计算方式
    """
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,  # 模型对象，可以是预训练模型或者任何nn.Module的子类
            args: TrainingArguments = None,  # 训练参数，包含了训练过程中的配置
            data_collator: Optional[DataCollator] = None,  # 数据整理器，用于将样本批量化
            train_dataset: Optional[Dataset] = None,  # 训练数据集
            eval_dataset: Optional[Dataset] = None,  # 评估数据集
            tokenizer: Optional[PreTrainedTokenizerBase] = None,  # 分词器，用于文本的编码
            model_init: Callable[[], PreTrainedModel] = None,  # 分词器，用于文本的编码
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,  # 用于计算评估指标的函数
            callbacks: Optional[List[TrainerCallback]] = None,  # 训练过程中的回调函数列表
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),  # 优化器和学习率调度器的元组
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,  # 在计算指标前对模型的logits进行预处理的函数
            compute_loss=None,  # 自定义损失计算函数
    ):
        super(Trainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.loss_func = compute_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写loss的计算方式
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        return self.loss_func(model, inputs, self.args, return_outputs)



class LoRATrainer(Trainer):
    """
    修改checkkpoint的保存逻辑，只保存lora
    1：保存模型  self.model.save_pretrained()
    2：保存token  self.tokenizer.save_pretrained()
    3: 保存参数  torch.save(self.args, file_path)
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        # ---创建输出地址文件
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # ---保存模型
        # 保存lora权重和配置
        self.model.save_pretrained(output_dir,  # 输出地址
                                   state_dict=state_dict,  # 参数
                                   safe_serialization=self.args.save_safetensors
                                   )

        # ---保存token
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # ---保存参数
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))