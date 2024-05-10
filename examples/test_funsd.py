#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import pickle

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

# import layoutlmft.data.datasets.funsd
import layoutlmft.data.funsd
import transformers
from layoutlmft.data import DataCollatorForKeyValueExtraction
# from layoutlmft.data.data_args import DataTrainingArguments
# from layoutlmft.models.model_args import ModelArguments
# from layoutlmft.trainers import FunsdTrainer as Trainer
from gat_utils import *
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    set_seed,
    DefaultFlowCallback,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from datetime import datetime
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
import torch
from pdb import set_trace as bp

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

class LoggerLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        control.should_log = False
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs) # using your custom logger


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default='funsd', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})





def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, rem_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        path_config = json.load(open(rem_args[-1],'r'))
        test_dir_name = rem_args[0].split('_')[-1]
        results_path = path_config["results_path"]
        # dir1_split = dir1.split('-')
        sz, sd = rem_args[1].split('-')[-1], rem_args[2].split('-')[-1]
        heuristic = rem_args[3].split('-')[-1]
        # sz, sd = dir1_split[-2], dir1_split[-1]
        test_dir =os.path.join(results_path, test_dir_name)
        pickle_path = ''
        try:
            mod_type = rem_args[4].split('-')[-1]
            if mod_type=='rotate':
                rotation_flag = True
                rotation_angle = int(rem_args[5].split('-')[-1])
            elif mod_type=='scale':
                scale_flag = True
                scale_factor = int(rem_args[5].split('-')[-1])
            elif mod_type=='shift':
                shift_flag = True
                shift = int(rem_args[5].split('-')[-1])

        except:
            rotation_angle = 8
            rotation_flag = False
            scale_flag = False
            scale_factor = 4
            shift_flag = False
            shift = 10

        training_args.logging_steps = 100
        # training_args.per_device_eval_batch_size = 1
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    timestamp = int(datetime.timestamp(datetime.now()))
    
    logging.basicConfig(
        filename='logs/test-lmv3-gat-closest_{}.log'.format(timestamp),
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # level=logging.INFO
    )
    logger.setLevel(logging.INFO)# if is_main_process(training_args.local_rank) else logging.WARN)
    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    datasets = load_dataset(os.path.abspath(layoutlmft.data.funsd.__file__), cache_dir=model_args.cache_dir)
    few_shot_info = json.loads(open(path_config['few_shot_info'],'r').read())
    image_to_id_dict = json.loads(open(path_config['image_to_id_dict'],'r').read())

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
        
    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(test_dir, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(test_dir)
    model = AutoModelForTokenClassification.from_pretrained(test_dir, config=config)
    
    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )
    
   
    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Get width height of images
    ids_to_train = [int(image_to_id_dict[x]) for x in few_shot_info[sz][sd]]

    image_to_id_train_json = json.load(open(path_config['image_to_id_train']))
    id_to_image_train_json = {j:i for i, j in image_to_id_train_json.items()}
    image_to_id_test_json = json.load(open(path_config['image_to_id_test']))
    id_to_image_test_json = {j:i for i, j in image_to_id_test_json.items()}
    width_height_train, width_height_test = get_widths_heights(id_to_image_train_json, id_to_image_test_json)
    width_height_train = [width_height_train[i] for i in ids_to_train]


    if data_args.visual_embed:
        imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])

        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    bp()
    gat_config_params = json.load(open("data/gat_params.json",'r'))

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, type1, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        if type1=='test':
            width_height_list = [width_height_test[i] for i in tokenized_inputs['overflow_to_sample_mapping']]
        else:
            width_height_list = [width_height_train[i] for i in tokenized_inputs['overflow_to_sample_mapping']]
        labels = []
        bboxes = []
        images = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)

            if data_args.visual_embed:
                ipath = examples["image_path"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes

        if data_args.visual_embed:
            tokenized_inputs["images"] = images
        
        '''Update the bounding boxes based on the type of image manipulation: rotation, scaling or shifting'''
        
        if type1=='test' and rotation_flag:
            print("Rotation : ", rotation_angle)
            tokenized_inputs["bbox"] = get_rotated_bboxes(bboxes, width_height_list, rotation_angle)
        
        if type1=='test' and scale_flag:
            # If the scale factor is divisible by 2, then height and width are scaled down equally
            print("Scale : ", scale_factor)
            if scale_factor!=2:  
                width_height_list = [(x[0]/(scale_factor), x[1]/(scale_factor/2)) for x in width_height_list]
            else:
                width_height_list = [(x[0]/scale_factor, x[1]) for x in width_height_list]  
            tokenized_inputs["bbox"] = get_scaled_bboxes(bboxes, scale_factor, width_height_list)
     
        if type1=='test' and shift_flag:
            print("Shift : ", shift)
            tokenized_inputs["bbox"] = get_shifted_bboxes(bboxes, shift, width_height_list)
        
        # pickle_gat_stuff(tokenized_inputs, pickle_path)
        all_adjs = []
        theta = gat_config_params['theta']
        adj_save_path = gat_config_params['adj_save_path']

        if heuristic == 'angles':
            '''K-nearest neighbors at multiple angles heuristic'''
            ## Try-except block to load pickled adjacency matrix for the test set
            try:
                all_adjs = pickle.load(open(os.path.join(adj_save_path,f'funsd_{type1}_adjs_{theta}_angles_v2_new_full.pkl'),'rb'))
                tokenized_inputs["adjs"] = all_adjs
                return tokenized_inputs
            except:
                theta_values = [x*theta for x in range(1,360//theta + 1)]
                for theta1 in theta_values:
                    adjs = np.array(get_adjs_new_angles_v2(tokenized_inputs, pickle_path, type1, theta1, width_height_list))
                    all_adjs.append(adjs)
                
                all_adjs = np.array(all_adjs)
                all_adjs = np.transpose(all_adjs, (1,0,2,3))
                
                if type1=='test':
                    if not os.path.exists(adj_save_path):
                            os.makedirs(adj_save_path)
                    pickle.dump(all_adjs, open(os.path.join(adj_save_path,f'funsd_{type1}_adjs_{theta}_angles_v2_new_full.pkl'),'wb'))
                
                tokenized_inputs["adjs"] = all_adjs
                return tokenized_inputs
        elif heuristic == 'nearest':
            '''K-nearest neighbors in space heuristic'''
            ## Try-except block to load pickled adjacency matrix for the test set
            try:
                adjs = pickle.load(open(os.path.join(adj_save_path,f'funsd_{type1}_adjs_closest_new_full.pkl'),'rb'))
                tokenized_inputs["adjs"] = adjs
                return tokenized_inputs
            except:
                adjs = np.array(get_adjs_new(tokenized_inputs, pickle_path, type1))
                if type1=='test':
                    if not os.path.exists(adj_save_path):
                            os.makedirs(adj_save_path)
                    pickle.dump(adjs, open(os.path.join(adj_save_path,f'funsd_{type1}_adjs_closest_new_full.pkl'),'wb'))
                tokenized_inputs["adjs"] = adjs
                return tokenized_inputs
        else:
            return tokenized_inputs


    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")

        train_dataset = datasets["train"].select(ids_to_train)
        # train_dataset = datasets["train"]

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file= data_args.overwrite_cache, fn_kwargs = {'type1':'train'},
        )

    if training_args.do_eval:
        # if "validation" not in datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["test"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file= data_args.overwrite_cache, fn_kwargs = {'type1':'test'},
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file= data_args.overwrite_cache, fn_kwargs = {'type1':'test'},
        )
        

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
    # Initialize our Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [LoggerLogCallback]
    )
    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
