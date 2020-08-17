import torch.nn as nn
from transformers import BartModel, BartConfig, modeling_bart


class BartLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BartConfig.from_pretrained('facebook/bart-large', use_cache=False)

        bart = BartModel(self.config)
        self.encoder = bart.encoder
        self.decoder = bart.decoder
        self.linear = nn.Linear(1024, 50265, bias=False)

    def forward(self, input_ids, output_ids, input_mask, output_mask):
        # encoder_hidden_states: [batch_size, max_length, hidden_size]
        encoder_hidden_states = self.encoder(input_ids=input_ids, attention_mask=input_mask)
        # out: [batch_size, max_length, hidden_size]
        decoder_input_ids, decoder_padding_mask, causal_mask = modeling_bart._prepare_bart_decoder_inputs(self.config,
                                                                                                          input_ids=output_ids)
        out, _, _, _ = self.decoder(input_ids=decoder_input_ids, encoder_padding_mask=input_mask,
                                    decoder_padding_mask=decoder_padding_mask, decoder_causal_mask=causal_mask,
                                    encoder_hidden_states=encoder_hidden_states[0])
        out = self.linear(out)
        return out
