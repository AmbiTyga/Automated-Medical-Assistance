## Sumit no need to run this
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

# import fire

# uses allennlp modules
from allennlp.nn import util

# imports chinese gpt
from transformers import *


def calculate_perplexity(
    batch_size=1,
    gpu_id=0,
    model_path='./BERT/model-10.pth'
    ):
    # make sure your model is on GPU
    device = torch.device(f"cuda:{gpu_id}")

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    # encoder = BartModel.from_pretrained("facebook/bart-base")
    # encoder = encoder.to(device)
    # encoder.eval()

    bert_encoder = BertConfig.from_pretrained('bert-base-uncased')
    bert_decoder = BertConfig.from_pretrained('bert-base-uncased',is_decoder = True)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(bert_encoder,bert_decoder)
    model = EncoderDecoderModel(config)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path,map_location='cuda'))
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------


    #------------------------LOAD VAL DATA------------------
    val_data = torch.load("/content/validate_data.pth")
    val_dataset = TensorDataset(*val_data)

    train_data = torch.load("/content/train_data.pth")
    train_dataset = TensorDataset(*train_data)

    test_data = torch.load("/content/test_data.pth")
    test_dataset = TensorDataset(*test_data)

    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
    #------------------------END LOAD VAL DATA--------------


    #------------------------START VAL-------------------
    perplexity = 0
    batch_count = 0
    print('start calculate the train perplexity....')

    with torch.no_grad():
        for batch in train_dataloader:
            batch = [item.to(device) for item in batch]

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            logits = model(input_ids = encoder_input,attention_mask = mask_encoder_input,
                           decoder_input_ids = decoder_input, decoder_attention_mask = mask_decoder_input)
            
            out = logits[0][:, :-1].contiguous()
                
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()
            
            
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

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            logits = model(input_ids = encoder_input,attention_mask = mask_encoder_input,
                           decoder_input_ids = decoder_input, decoder_attention_mask = mask_decoder_input)
            
            out = logits[0][:, :-1].contiguous()
                
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

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

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = batch

            logits = model(input_ids = encoder_input,attention_mask = mask_encoder_input,
                           decoder_input_ids = decoder_input, decoder_attention_mask = mask_decoder_input)
            
            out = logits[0][:, :-1].contiguous()
                
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(out, target, target_mask, average="token")
            perplexity += np.exp(loss.item())
            batch_count += 1


    print(f'test perplexity: {perplexity / batch_count}')


    #------------------------END VAL-------------------


 if __name__ == '__main__':
     calculate_perplexity()
