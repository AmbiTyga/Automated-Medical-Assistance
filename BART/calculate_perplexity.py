import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from allennlp.nn import util
from bartLM import BartLM

from transformers import BartModel, BartConfig


def calculate_perplexity(
        batch_size=1,
        gpu_id=0,
        decoder_path="/content/BART_CheckPoints/model-10.pth"
):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    # ------------------------LOAD MODEL-----------------
    print('load the model....')

    model = BartLM()
    model.load_state_dict(torch.load(decoder_path, map_location='cuda'))
    model = model.to(device)
    model.eval()

    print('load success')
    # ------------------------END LOAD MODEL--------------

    # ------------------------LOAD VAL DATA------------------
    val_data = torch.load("/content/validate_data.pth")
    val_dataset = TensorDataset(*val_data)

    train_data = torch.load("/content/train_data.pth")
    train_dataset = TensorDataset(*train_data)

    test_data = torch.load("/content/test_data.pth")
    test_dataset = TensorDataset(*test_data)

    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
    # ------------------------END LOAD VAL DATA--------------

    # ------------------------START VAL-------------------
    perplexity = 0
    batch_count = 0
    print('start calculate the train perplexity....')

    with torch.no_grad():
        for batch in train_dataloader:
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

    print(f'train perplexity: {perplexity / batch_count}')

    perplexity = 0
    batch_count = 0
    print('start calculate the validate perplexity....')

    with torch.no_grad():
        for batch in val_dataloader:
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

    print(f'validate perplexity: {perplexity / batch_count}')

    perplexity = 0
    batch_count = 0
    print('start calculate the test perplexity....')

    with torch.no_grad():
        for batch in test_dataloader:
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

    print(f'test perplexity: {perplexity / batch_count}')

    # ------------------------END VAL-------------------



if __name__ == '__main__':
    calculate_perplexity()
