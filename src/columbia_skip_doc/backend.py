import logging
import os

import datapipeline as pipeline
import yaml
import torch
import argparse

from utils import setup_logging
from constants import PRJ_ROOT_STR

from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForGeneration
from openprompt.utils.metrics import generation_metric

from transformers import AdamW, get_linear_schedule_with_warmup

FILE_NAME = __name__

_logger = setup_logging(logging.DEBUG, FILE_NAME)


class OpenPromptPipeline:
    def __init__(self, model_type, model_name):
        """Constructor for OpenPromptPipeline class

        Args:
            model_type (str): the model type - see OpenPrompt documentation for supported models in openpromt.plms.__init__.py
            model_name (str): the model name as seen on huggingface.co's model page
        """
        self.model_type = model_type
        self.model_name = model_name

    def load_model(self):
        """Uses OpenPrompt function to load a pretrained model, the model's tokenizer, the model's config from huggingface,
        and OpenPrompt's wrapper class object for the model's objective.

        Returns:
            plm: huggingface pretrained model
            tokenizer: huggingface tokenizer for the model
            model_config: huggingface model config
            wrapper_class: OpenPrompt wrapper class for the model's objective
        """
        return load_plm(self.model_type, self.model_name)

    def generate_data_loaders(
        self,
        data_dict,
        template,
        tokenizer,
        tokenizer_wrapper_class,
        max_seq_length=512,
        batch_size=5,
        teacher_forcing=False,
        predict_eos_token=True,
        truncate_method="tail",
    ):
        """Generates OpenPrompt data loaders for the train, validation, and test datasets

        Args:
            data_dict (dict): dictionary containing the three splits of input data
            template (Template): a derived class of :obj:`Template`
            tokenizer (PretrainedTokenizer): the pretrained tokenizer
            tokenizer_wrapper_class (TokenizerWrapper): the class of tokenizer wrapper
            max_seq_length (int, optional): the max sequence length of the input ids. It's used to truncate sentences. Defaults to 512.
            batch_size (int, optional): the batch_size of data loader. Defaults to 5.
            teacher_forcing (bool, optional): whether to fill the mask with target text. Set to True in training generation model.. Defaults to False.
            predict_eos_token (bool, optional): whether to predict the <eos> token. Suggest to set to True in generation. Defaults to True.
            truncate_method (str, optional): the truncate method to use. select from `head`, `tail`, `balanced`. Defaults to "head".

        Returns:
            train_dataloader: OpenPrompt data loader for the train dataset
            val_dataloader: OpenPrompt data loader for the validation dataset
            test_dataloader: OpenPrompt data loader for the test dataset
        """
        train_dataloader = PromptDataLoader(
            dataset=data_dict["train"],
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=tokenizer_wrapper_class,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            teacher_forcing=True,
            predict_eos_token=predict_eos_token,  # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
            truncate_method=truncate_method,
        )

        validation_dataloader = PromptDataLoader(
            dataset=data_dict["validation"],
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=tokenizer_wrapper_class,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            teacher_forcing=teacher_forcing,
            predict_eos_token=predict_eos_token,
            truncate_method=truncate_method,
        )

        test_dataloader = PromptDataLoader(
            dataset=data_dict["test"],
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=tokenizer_wrapper_class,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            teacher_forcing=teacher_forcing,
            predict_eos_token=predict_eos_token,
            truncate_method=truncate_method,
        )

        return train_dataloader, validation_dataloader, test_dataloader

    def create_template(
        self,
        tokenizer,
        model,
        prompt_text,
        prompt_type="prefix_tuning",
        using_decoder_past_key_values=False,
    ):
        """Creates OpenPrompt template object

        Args:
            tokenizer (PretrainedTokenizer): the pretrained tokenizer
            model (PretrainedModel): the pretrained model
            prompt_text (str): the prompt text
            prompt_type (str, optional): the prompt type. Defaults to "prefix_tuning".
            using_decoder_past_key_values (bool, optional): whether to use decoder past key values. Defaults to False.

        Returns:
            template: OpenPrompt template object
        """
        if prompt_type == "prefix_tuning":
            template = PrefixTuningTemplate(
                model=model,
                tokenizer=tokenizer,
                text=prompt_text,
                using_decoder_past_key_values=using_decoder_past_key_values,
            )
        elif prompt_type == "manual":
            template = ManualTemplate(
                tokenizer=tokenizer,
                text=prompt_text,
            )
        else:
            raise ValueError(
                f"prompt_type {prompt_type} not supported. Please choose from prefix_tuning or manual"
            )

        return template

    def create_prompt_model_for_generation(
        self, plm, tokenizer, template, freeze_plm=True, use_cuda=False
    ):
        """Creates OpenPrompt prompt model

        Args:
            plm (PretrainedModel): the pretrained model
            tokenizer (PretrainedTokenizer): the pretrained tokenizer
            template (Template): a derived class of :obj:`Template`
            freeze_plm (bool, optional): whether to freeze the pretrained model. Defaults to True.
            use_cuda (bool, optional): whether to use CUDA. Defaults to False.

        Returns:
            prompt_model: OpenPrompt prompt model
        """
        prompt_model = PromptForGeneration(
            plm=plm,
            tokenizer=tokenizer,
            template=template,
            freeze_plm=freeze_plm,
        )

        if use_cuda:
            prompt_model = prompt_model.cuda()

        return prompt_model

    def setup_training(
        self, no_decay_list, template, learning_rate=5e-5, eps=1e-8, batch_size=5
    ):
        """Sets up the optimizer for the prompt model

        Args:
            no_decay_list (list): list of parameters that don't need weight decay
            template (Template): a derived class of :obj:`Template`
            learning_rate (float, optional): the learning rate. Defaults to 5e-5.
            eps (float, optional): the epsilon. Defaults to 1e-8.
            batch_size (int, optional): the batch size. Defaults to 5.

        Returns:
            optimizer: the optimizer, steps and scheduler
        """
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in template.named_parameters()
                    if (not any(nd in n for nd in no_decay_list)) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in template.named_parameters()
                    if any(nd in n for nd in no_decay_list) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
        tot_step = len(train_dataloader) * batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

        return optimizer, tot_step, scheduler

    def train_model(
        self,
        train_dataloader,
        prompt_model,
        template,
        optimizer,
        scheduler,
        use_cuda
    ):
        """
        Trains the prompt model

        Args:
            train_dataloader (PromptDataLoader): the train dataloader
            prompt_model (PromptForGeneration): the prompt model
            template (Template): a derived class of :obj:`Template`
            optimizer (optimizer): the optimizer
            scheduler (scheduler): the scheduler
            use_cuda (bool): whether to use CUDA

        Returns:
            the trained prompt model
        """
        global_step = 0
        tot_loss = 0
        log_loss = 0
        # 5 epochs recommended
        for epoch in range(5):
            prompt_model.train()
            for _, inputs in enumerate(train_dataloader):
                global_step += 1
                if use_cuda:
                    inputs = inputs.cuda()
                loss = prompt_model(inputs)
                loss.backward()
                tot_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(template.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if global_step % 500 == 0:
                    _logger.info(
                        f"Epoch {epoch}, global_step {global_step} average loss: {(tot_loss - log_loss) / 500} lr: {scheduler.get_last_lr()[0]}",
                    )
                    log_loss = tot_loss

            return prompt_model

    def evaluate_model(
        self,
        dataloader,
        prompt_model,
        use_cuda,
        generation_arguments,
        model_name,
        metadata,
    ):
        """Evaluate the model using the sentence_bleu metric

        Args:
            train_dataloader (PromptDataLoader): the train dataloader
            prompt_model (PromptForGeneration): the prompt model
            use_cuda (bool): whether to use CUDA
            generation_arguments (dict): the generation arguments
            model_name (str): the model name
            metadata (dict): the metadata about the training and scoring metrics
        """
        generated_sentence = []
        groundtruth_sentence = []
        prompt_model.eval()

        for _, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            _, output_sentences = prompt_model.generate(inputs, **generation_arguments)
            for idx, q in enumerate(inputs["input_ids"]):
                _logger.info(tokenizer.decode(q, skip_special_tokens=True))
                _logger.info(f"Model generated: {output_sentences[idx]}")
            generated_sentence.extend(output_sentences)
            groundtruth_sentence.extend(inputs["tgt_text"])
        score = generation_metric(
            generated_sentence, groundtruth_sentence, "sentence_bleu"
        )
        metadata["model_score"] = score
        _logger.info(f"Model score: {score}")

        try:
            output_dir = os.path.join(
                PRJ_ROOT_STR, "src", "columbia_skip_doc", "models", model_name.split("/")[1]
            )

            output_score_dir = os.path.join(
                PRJ_ROOT_STR,
                "src",
                "columbia_skip_doc",
                "model_params",
                model_name.split("/")[1],
            )
        except IndexError:
            output_dir = os.path.join(
                PRJ_ROOT_STR, "src", "columbia_skip_doc", "models", model_name
            )

            output_score_dir = os.path.join(
                PRJ_ROOT_STR,
                "src",
                "columbia_skip_doc",
                "model_params",
                model_name
            )

        if not os.path.exists(output_score_dir):
            os.makedirs(output_score_dir)
            with open(os.path.join(output_score_dir, "params.yaml"), "w") as f:
                yaml.dump(metadata, f)
            self.save_model(prompt_model, output_dir=output_dir)
        else:
            with open(os.path.join(output_score_dir, "params.yaml"), "r") as f:
                old_score = yaml.safe_load(f)["model_score"]
            if score > old_score:
                _logger.info(
                    "New score is better than old score, saving model run metadata"
                )
                self.save_model(prompt_model, output_dir=output_dir)
            else:
                _logger.info(
                    "Old score is better than new score, not saving model run metadata"
                )

        return

    def save_model(self, prompt_model, output_dir):
        """Saves the prompt model and tokenizer

        Args:
            prompt_model (PromptForGeneration): the prompt model
            output_dir (str): the output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prompt_model.plm.transformer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--use_cuda", type=bool, default=True)
    args = parser.parse_args()

    # To debug CUDA issues
    # CUDA_LAUNCH_BLOCKING = 1

    #### PARAMETERS ####

    # Only use if you have a GPU
    use_cuda = args.use_cuda
    model_type = "gpt2"
    model_name = "gpt2"
    train_split = 0.7
    validation_split = 0.1
    test_split = 0.2
    prompt_type = "prefix_tuning"
    max_sequence_length = 640
    truncate_method = "tail"
    batch_size = 3
    freeze_plm = False
    learning_rate = 5e-5
    eps = 1e-8
    epochs = 5

    #######################

    data_fp = os.path.join(
        PRJ_ROOT_STR, "data", "All-2479-Answers-retrieved-from-MedQuAD.csv"
    )
    data_classes_fp = os.path.join(
        PRJ_ROOT_STR,
        "data",
        "All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt",
    )

    dp = pipeline.DataPipeline(
        data_fp,
        data_classes_fp,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
    )
    data_df = dp.read_and_clean_data(id_col_data="AnswerID", id_col_data_classes="id")
    data_dict = dp.split_data_into_dictionary(data_df)

    op = OpenPromptPipeline(model_type, model_name)
    plm, tokenizer, model_config, WrapperClass = op.load_model()
    _logger.info(f"Loaded model: {plm.name_or_path}")

    template_text = """
          Question: {'placeholder':'text_a', 'shortenable': 'True'} You are a doctor in a clinic. Answer the question and provide a plan of action if needed. {'mask'}
    """

    template = op.create_template(
        tokenizer=tokenizer,
        model=plm,
        prompt_text=template_text,
        prompt_type=prompt_type,
    )

    wrapped_example = template.wrap_one_example(data_dict["train"][0])
    _logger.info(
        f"Created template object for prompt! Example wrapped in template: {wrapped_example}"
    )

    wrapped_tokenizer = WrapperClass(
        max_seq_length=max_sequence_length,
        tokenizer=tokenizer,
        truncate_method=truncate_method,
    )
    _logger.info(f"Initialized tokenizer!")

    train_dataloader, validation_dataloader, test_dataloader = op.generate_data_loaders(
        data_dict=data_dict,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_sequence_length,
        batch_size=batch_size,
        truncate_method=truncate_method,
    )
    _logger.info(f"Generated PromptDataLoader objects!")

    prompt_model = op.create_prompt_model_for_generation(
        plm=plm,
        tokenizer=tokenizer,
        template=template,
        freeze_plm=freeze_plm,
        use_cuda=use_cuda,
    )

    _logger.info(f"Created prompt model!")

    optimizer, tot_step, scheduler = op.setup_training(
        no_decay_list=["bias", "LayerNorm.weight"],
        template=template,
        learning_rate=learning_rate,
        eps=eps,
        batch_size=batch_size,
    )

    _logger.info(f"Setup training!")

    prompt_model = op.train_model(
        train_dataloader=train_dataloader,
        prompt_model=prompt_model,
        template=template,
        optimizer=optimizer,
        scheduler=scheduler,
        use_cuda=use_cuda
    )

    _logger.info(f"Trained model!")

    generation_arguments = {
        "max_length": 884,
        "max_new_tokens": None,
        "min_length": 5,
        "temperature": 1.0,
        "do_sample": False,
        "top_k": 0,
        "top_p": 0.9,
        "repetition_penalty": 5.0,
        "num_beams": 3,
    }

    metadata = {
        "dataset": "medquad",
        "train_split": train_split,
        "validation_split": validation_split,
        "test_split": test_split,
        "template": template_text,
        "prompt_type": prompt_type,
        "max_sequence_length": max_sequence_length,
        "truncate_method": truncate_method,
        "batch_size": batch_size,
        "freeze_plm": freeze_plm,
        "learning_rate": learning_rate,
        "eps": eps,
        "generation_metrics": generation_arguments,
        "epochs": epochs,
    }

    op.evaluate_model(
        test_dataloader,
        prompt_model,
        use_cuda,
        generation_arguments,
        model_name,
        metadata,
    )

    _logger.info(f"Evaluation complete!")
