import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "gliner_finetuning"
# os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "none"

import argparse
import random
from glob import glob
import json

from transformers import AutoTokenizer, EarlyStoppingCallback
import torch

from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
from utils import GLiNERConfigArgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--log_dir", type=str, default="data/models/")
    parser.add_argument("--compile_model", type=bool, default=False)
    parser.add_argument("--freeze_language_model", type=bool, default=False)
    parser.add_argument("--new_data_schema", type=bool, default=False)
    args = parser.parse_args()

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    config: GLiNERConfigArgs = load_config_as_namespace(args.config)
    config.log_dir = args.log_dir

    print("Start loading dataset...")

    files = glob(os.path.join(config.train_data))
    data = [json.load(open(f, "r")) for f in files]
    train_data = sum(data, start=[])

    files = glob(os.path.join(config.val_data_dir))
    data = [json.load(open(f, "r")) for f in files]
    test_data = sum(data, start=[])

    random.shuffle(train_data)

    print("Dataset is splitted...", len(train_data), len(test_data))

    if config.prev_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(config.prev_path)
        model = GLiNER.from_pretrained(config.prev_path)
        model_config = model.config
    else:
        model_config = GLiNERConfig(**vars(config))
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

        words_splitter = WordsSplitter(model_config.words_splitter_type)

        model = GLiNER(model_config, tokenizer=tokenizer, words_splitter=words_splitter)

        if not config.labels_encoder:
            model_config.class_token_index = len(tokenizer)
            tokenizer.add_tokens(
                [model_config.ent_token, model_config.sep_token], special_tokens=True
            )
            model_config.vocab_size = len(tokenizer)
            model.resize_token_embeddings(
                [model_config.ent_token, model_config.sep_token],
                set_class_token_index=False,
                add_tokens_to_tokenizer=False,
            )

    if args.compile_model:
        torch.set_float32_matmul_precision("medium")
        model.to(device)
        model.compile_for_training()

    if args.freeze_language_model:
        model.model.token_rep_layer.bert_layer.model.requires_grad_(False)
    else:
        model.model.token_rep_layer.bert_layer.model.requires_grad_(True)

    if args.new_data_schema:
        train_dataset = GLiNERDataset(
            train_data, model_config, tokenizer, words_splitter
        )
        test_dataset = GLiNERDataset(test_data, model_config, tokenizer, words_splitter)
        data_collator = DataCollatorWithPadding(model_config)
    else:
        train_dataset = train_data
        test_dataset = test_data
        data_collator = DataCollator(
            model.config, data_processor=model.data_processor, prepare_labels=True
        )

    save_steps = int(0.25 * len(train_dataset) // config.train_batch_size)

    training_args = TrainingArguments(
        output_dir=config.log_dir,
        run_name="gliner-modern-bert",
        learning_rate=float(config.lr_encoder),
        weight_decay=float(config.weight_decay_encoder),
        others_lr=float(config.lr_others),
        others_weight_decay=float(config.weight_decay_other),
        lr_scheduler_type=config.scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        max_grad_norm=config.max_grad_norm,
        # max_steps=config.num_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        save_steps=save_steps,
        logging_steps=save_steps // 2,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=8,
        use_cpu=False,
        report_to="wandb",
        bf16=True,
        load_best_model_at_end=True,
        num_train_epochs=config.num_train_epochs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(3)],
    )
    trainer.train()
