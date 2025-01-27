import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["WANDB_PROJECT"] = "gliner_finetuning"
# os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "none"

import json
from typing import Optional
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

from utils import seed_everything
import argparse
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    train_data: str
    test_data: str
    output_dir: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    gradient_accumulation_steps: int
    save_steps: Optional[int]


def main(config: TrainingConfig):
    seed_everything(41)

    print("Running process with pid:", os.getpid())

    model = GLiNER.from_pretrained(
        "data/model",
        # _attn_implementation="flash_attention_2",
        max_length=2048,
    )
    print(model)

    # use it for better performance, it mimics original implementation but it's less memory efficient
    data_collator = DataCollator(
        model.config, data_processor=model.data_processor, prepare_labels=True
    )

    # train_dataset = json.load(open("LMRData/train_location.json"))
    train_dataset = json.load(open(config.train_data))
    test_dataset = json.load(open(config.test_data))

    print("Dataset Size:", len(train_dataset), len(test_dataset))

    print(train_dataset[:5])

    save_steps = (
        int(
            len(train_dataset)
            * 0.5
            / (config.batch_size * config.gradient_accumulation_steps)
        )
        // (2)
    ) * 2 if config.save_steps is None else config.save_steps

    training_args = TrainingArguments(
        run_name="fine_tune_gliner_large",
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        weight_decay=0.0001,
        others_lr=0.5e-6,
        others_weight_decay=0.0001,
        lr_scheduler_type="constant",  # linear cosine
        warmup_ratio=0,  # .1
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        num_train_epochs=config.num_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=save_steps // 2,
        save_total_limit=10,
        dataloader_num_workers=0,
        report_to="wandb",
        metric_for_best_model="loss",
        loss_reduction="sum",
        ddp_find_unused_parameters=True,
        load_best_model_at_end=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=save_steps//2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(4)],
    )

    trainer.train()

    trainer.evaluate()


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description="Fine-tune GLiNER model")
        parser.add_argument(
            "--train_data",
            type=str,
            default="data/data/train.json",
            help="Path to the training data",
        )
        parser.add_argument(
            "--test_data",
            type=str,
            default="data/data/test.json",
            help="Path to the test data",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="data/models",
            help="Directory to save the model",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="Batch size for training and evaluation",
        )
        parser.add_argument(
            "--num_epochs", type=int, default=5, help="Number of training epochs"
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-6,
            help="Learning rate for training",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of gradient accumulation steps",
        )
        parser.add_argument(
            "--save_steps",
            type=int,
            default=None,
            help="Number of steps to save the model",
        )
        return parser.parse_args()

    args = parse_args()
    config = TrainingConfig(
        train_data=args.train_data,
        test_data=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
    )
    main(config)
