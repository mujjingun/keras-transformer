import argparse
from typing import Optional
import random
import os

from keras.models import load_model
# noinspection PyPep8Naming
from keras import optimizers
from keras import callbacks
from keras import losses

from keras_transformer.bert import (
    BatchGeneratorForBERT, masked_perplexity,
    MaskedPenalizedSparseCategoricalCrossentropy)

from . import wikitext
from .bpe import BPEEncoder
from .utils import (
    load_optimizer_weights, contain_tf_gpu_mem_usage, CosineLRSchedule)
from .models import transformer_bert_model_simplified

BERT_SPECIAL_TOKENS = ['[SEP]', '[CLS]', '[MASK]']

# Penalty for confidence of the output distribution, as described in
# "Regularizing Neural Networks by Penalizing Confident Output Distributions"
# (https://arxiv.org/abs/1701.06548)
CONFIDENCE_PENALTY = 0.1


def stream_bpe_token_ids(text: str, encoder: BPEEncoder):
    for line in text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        # the encoder is supposed to add <SEQ> and </SEQ>
        for token_id, token in encoder(clean_line):
            yield token_id


def wikitext_bert_generator(
        dataset_name: str, encoder: BPEEncoder,
        batch_size: int, sequence_length: int) -> BatchGeneratorForBERT:
    text = wikitext.read_wikitext_file(dataset_name)
    token_ids = list(stream_bpe_token_ids(text, encoder))

    def sampler(size):
        start = random.randint(0, len(token_ids) - size - 1)
        return token_ids[start: start + size]

    sep_token_id, cls_token_id, mask_token_id = [
        encoder.vocabulary.token_to_id[token]
        for token in BERT_SPECIAL_TOKENS]
    generator = BatchGeneratorForBERT(
        sampler=sampler,
        dataset_size=len(token_ids),
        sep_token_id=sep_token_id,
        cls_token_id=cls_token_id,
        mask_token_id=mask_token_id,
        first_normal_token_id=encoder.vocabulary.first_normal_token_id,
        last_normal_token_id=encoder.vocabulary.last_normal_token_id,
        sequence_length=sequence_length,
        batch_size=batch_size)
    return generator


def main(model_save_path: str,
         model_name: str,
         tensorboard_log_path: Optional[str],
         num_epochs: int,
         learning_rate: float,
         batch_size: int,
         max_seq_length: int,
         word_embedding_size: int,
         load_weights_only: bool,
         show_model_summary: bool,
         beta_1: float,
         beta_2: float,
         transformer_dropout: float,
         embedding_dropout: float,
         l2_reg_penalty: float):
    contain_tf_gpu_mem_usage()
    encoder = wikitext.build_wikitext_bpe_encoder(
        special_tokens=BERT_SPECIAL_TOKENS)

    def compile_new_model():
        optimizer = optimizers.Adam(
            lr=learning_rate, beta_1=beta_1, beta_2=beta_2)
        _model = transformer_bert_model_simplified(
            use_universal_transformer=(model_name == 'universal'),
            max_seq_length=max_seq_length,
            vocabulary_size=encoder.vocabulary_size(),
            word_embedding_size=word_embedding_size,
            transformer_depth=4,
            num_heads=8,
            transformer_dropout=transformer_dropout,
            embedding_dropout=embedding_dropout,
            l2_reg_penalty=l2_reg_penalty)
        _model.compile(
            optimizer,
            loss=[
                MaskedPenalizedSparseCategoricalCrossentropy(
                    CONFIDENCE_PENALTY),
                losses.binary_crossentropy],
            metrics={'word_predictions': masked_perplexity})
        return _model

    if os.path.exists(model_save_path):
        if load_weights_only:
            print('Loading weights from', model_save_path)
            model = compile_new_model()
            model.load_weights(model_save_path,
                               skip_mismatch=True, by_name=True)
            load_optimizer_weights(model, model_save_path)
        else:
            print('Loading the whole model from', model_save_path)
            model = load_model(
                model_save_path,
                custom_objects={
                    'masked_perplexity': masked_perplexity,
                })
    else:
        model = compile_new_model()

    if show_model_summary:
        model.summary(120)

    lr_scheduler = callbacks.LearningRateScheduler(
        CosineLRSchedule(lr_high=learning_rate, lr_low=1e-8,
                         initial_period=num_epochs),
        verbose=1)
    model_callbacks = [
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss', save_best_only=True, verbose=True),
        lr_scheduler,
    ]
    if tensorboard_log_path:
        model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path))

    training_batches = wikitext_bert_generator(
        wikitext.TRAINING_SET_NAME, encoder, batch_size, max_seq_length)
    validation_batches = wikitext_bert_generator(
        wikitext.VALIDATION_SET_NAME, encoder, batch_size, max_seq_length)
    model.fit_generator(
        generator=training_batches.generate_batches(),
        steps_per_epoch=training_batches.steps_per_epoch,
        epochs=num_epochs,
        callbacks=model_callbacks,
        validation_data=validation_batches.generate_batches(),
        validation_steps=validation_batches.steps_per_epoch,
    )
    # Evaluation using test set
    print('-' * 80)
    test_batches = wikitext_bert_generator(
        wikitext.TEST_SET_NAME, encoder, batch_size, max_seq_length)
    test_metrics = model.evaluate_generator(
        test_batches.generate_batches(),
        test_batches.steps_per_epoch)
    for metric_name, metric_value in zip(model.metrics_names, test_metrics):
        print(f'Test {metric_name}:', metric_value)

def bench(x):
    return main(
        model_save_path="lm_model.h5",
        model_name='universal',
        tensorboard_log_path=None,
        num_epochs=2,
        learning_rate=x[1], # 2e-4
        batch_size=32,
        max_seq_length=256,
        word_embedding_size=x[2], # 512
        load_weights_only=True,
        show_model_summary=True,
        beta_1=x[3], # 0.9
        beta_2=x[4], # 0.999
        transformer_dropout=x[5], # 0.1
        embedding_dropout=x[6], # 0.6
        l2_reg_penalty=x[7]) # 1e-4
