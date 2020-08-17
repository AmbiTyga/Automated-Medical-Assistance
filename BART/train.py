import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import time
import os
from bartLM import BartLM

from allennlp.nn import util

from transformers import AdamW, get_linear_schedule_with_warmup, BartModel, BartConfig

max_grad_norm = 1.0


def train_model(
        epochs=10,
        num_gradients_accumulation=4,
        batch_size=4,
        gpu_id=0,
        lr=4e-5,
        load_dir='/content/BART_CheckPoints'
):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    # ------------------------LOAD MODEL-----------------
    print('load the model....')
    model = BartLM()
    model = model.to(device)
    print('load success')
    # ------------------------END LOAD MODEL--------------

    # ------------------------LOAD TRAIN DATA------------------
    train_data = torch.load("/content/train_data.pth")
    train_dataset = TensorDataset(*train_data)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    val_data = torch.load("/content/validate_data.pth")
    val_dataset = TensorDataset(*val_data)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size)
    # ------------------------END LOAD TRAIN DATA--------------

    # ------------------------SET OPTIMIZER-------------------
    num_train_optimization_steps = len(train_dataset) * epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, \
        lr=lr, \
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, \
        num_warmup_steps=num_train_optimization_steps // 10, \
        num_training_steps=num_train_optimization_steps
    )
    # ------------------------END SET OPTIMIZER--------------

    # ------------------------START TRAINING-------------------
    update_count = 0

    start = time.time()
    print('start training....')
    for epoch in range(epochs):
        # ------------------------training------------------------
        model.train()
        losses = 0
        times = 0
        print('\n' + '-' * 20 + f'epoch {epoch}' + '-' * 20)
        for batch in tqdm(train_dataloader):
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder, mask_decoder = batch

            logits = model(input_ids=encoder_input, output_ids=decoder_input, input_mask=mask_encoder,
                           output_mask=mask_decoder)

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder[:, 1:].contiguous()
            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            loss.backward()

            losses += loss.item()

            times += 1
            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        end = time.time()
        print('\n'+'-'*20 + f'epoch {epoch}' + '-'*20)
        print(f'\n')
        print(f'loss: {losses / times}')
        start = end

        # ------------------------validate------------------------
        model.eval()

        perplexity = 0
        batch_count = 0
        print('start calculate the perplexity....')

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = [item.to(device) for item in batch]

                encoder_input, decoder_input, mask_encoder, mask_decoder = batch

                logits = model(input_ids=encoder_input, output_ids=decoder_input, input_mask=mask_encoder,
                               output_mask=mask_decoder)

                out = logits[:, :-1].contiguous()
                target = decoder_input[:, 1:].contiguous()
                target_mask = mask_decoder[:, 1:].contiguous()
                loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
                perplexity += np.exp(loss.item())
                batch_count += 1

        print(f'\nValidate perplexity: {perplexity / batch_count}')

        torch.save(model.state_dict(),
                   os.path.join(os.path.abspath('.'), load_dir, "model-" + str(epoch) + ".pth"))

    # ------------------------END TRAINING-------------------


if __name__ == '__main__':
    train_model()
