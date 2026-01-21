import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from transformers import GenerationMixin
from torch import nn
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from vq import RQVAE
from layers import *


@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    rank_logits: Optional[torch.FloatTensor] = None
    seq_latents: Optional[torch.FloatTensor] = None
    seq_project_latents: Optional[torch.FloatTensor] = None
    dec_latents: Optional[torch.FloatTensor] = None
        
        
class Model(nn.Module, GenerationMixin):
    def __init__(self, config, model, n_items, code_length=1, code_number=256):
        super().__init__()
        self.model = model
        # Handle missing _supports_cache_class attribute for compatibility
        self._supports_cache_class = getattr(model, '_supports_cache_class', False)
        self.config = model.config
        self.base_model_prefix = "model"
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        
        # Detect model type: decoder-only (GPT-2) vs encoder-decoder (T5)
        self.is_decoder_only = not hasattr(model, 'get_encoder')
        if self.is_decoder_only:
            # GPT-2 doesn't have encoder, use the transformer as the "encoder" for compatibility
            self.get_encoder = lambda: model.transformer
        else:
            # T5 has encoder
            self.get_encoder = model.get_encoder
            
        self.device = model.device
        self.can_generate = lambda: True

        self.hidden_size = model.config.hidden_size
        self.semantic_hidden_size = config.get('semantic_hidden_size')
        self.n_items = n_items
        self.code_length = code_length
        self.code_number = code_number
        self.num_beams = config['num_beams']
        
        self.semantic_embedding = nn.Embedding(self.n_items, self.semantic_hidden_size)
        self.semantic_embedding.requires_grad_(False)
        
        self.token_embeddings = nn.ModuleList([nn.Embedding(self.code_number, self.hidden_size) for i in range(self.code_length)])
        self.token_embeddings.requires_grad_(True)
        
        # Start token embedding for decoder (shared between T5 and GPT-2)
        # For T5, this supplements model.shared; for GPT-2, this replaces wte usage
        self.start_token_embedding = nn.Embedding(1, self.hidden_size)
        self.start_token_embedding.requires_grad_(True)
        
        enc_adapter_layers = config['layers']
        enc_adapter_layers = [self.hidden_size] + [config['e_dim']]
        self.enc_adapter = MLPLayers(layers=enc_adapter_layers)

        dec_adapter_layers = config['layers'][::-1]
        dec_adapter_layers = [self.hidden_size] + [self.semantic_hidden_size]
        self.dec_adapter = MLPLayers(layers=dec_adapter_layers)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):
        if self.is_decoder_only:
            # GPT-2: decoder-only model
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            # T5: encoder-decoder model
            return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs, "attention_mask": attention_mask}

    def _shift_right(self, input_ids):
        # Use appropriate token based on model type
        if self.is_decoder_only:
            # GPT-2: use bos_token_id or 0 as default
            pad_token_id = getattr(self.config, 'bos_token_id', 0)
        else:
            # T5: use pad_token_id
            pad_token_id = self.config.pad_token_id
        
        if pad_token_id is None:
            pad_token_id = 0

        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids], dim=-1)

        return shifted_input_ids
    
    def get_input_embeddings(self, input_ids, attention_mask):
        attention_mask_flatten = attention_mask.reshape(-1)

        # Use input_ids device instead of self.device for correct device placement
        device = input_ids.device
        inputs_embeds = torch.zeros(*input_ids.shape, self.hidden_size, device=device)
        
        # Clamp input_ids to valid range [0, code_number-1] to avoid index out of bounds
        # Replace -1 with 0 first, then clamp all values
        input_ids_clamped = input_ids.clone()
        input_ids_clamped[input_ids_clamped == -1] = 0
        input_ids_clamped = torch.clamp(input_ids_clamped, 0, self.code_number - 1)
        
        for i in range(self.code_length):
            inputs_embeds[:, i::self.code_length] = self.token_embeddings[i](input_ids_clamped[:, i::self.code_length])
        
        inputs_embeds = inputs_embeds.view(-1, self.hidden_size)
        # Use start token embedding for padding (works for both architectures)
        pad_embed = self.start_token_embedding.weight[0].to(device)
        inputs_embeds[~attention_mask_flatten] = pad_embed
        inputs_embeds = inputs_embeds.view(input_ids.shape[0], -1, self.hidden_size)

        return inputs_embeds
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, decoder_input_ids=None,
                decoder_inputs_embeds=None, encoder_outputs=None, **kwargs):
        
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        if decoder_input_ids is None and labels is None:
            decoder_input_ids = torch.zeros(input_ids.size(0), self.code_length).long().to(input_ids.device)
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = []
            for i in range(min(decoder_input_ids.shape[1], self.code_length)):
                if i==0:
                    # Use start token embedding (works for both T5 and GPT-2)
                    # Map all start token IDs to 0 since we only have 1 start embedding
                    start_ids = torch.zeros(decoder_input_ids.size(0), dtype=torch.long, device=decoder_input_ids.device)
                    decoder_inputs_embeds.append(self.start_token_embedding(start_ids))
                else:
                    code_embedding = self.token_embeddings[i-1]  # 0~255
                    # Clamp decoder_input_ids to valid range to avoid index out of bounds
                    decoder_ids_clamped = torch.clamp(decoder_input_ids[:, i], 0, self.code_number - 1)
                    decoder_inputs_embeds.append(code_embedding(decoder_ids_clamped))
            decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)

        if self.is_decoder_only:
            # GPT-2: Concatenate input and decoder embeddings for causal LM
            batch_size = inputs_embeds.size(0)
            input_len = inputs_embeds.size(1)
            dec_len = decoder_inputs_embeds.size(1)
            
            # Ensure both embeddings are on the same device
            if inputs_embeds.device != decoder_inputs_embeds.device:
                decoder_inputs_embeds = decoder_inputs_embeds.to(inputs_embeds.device)
            
            # Combine input and decoder embeddings
            combined_embeds = torch.cat([inputs_embeds, decoder_inputs_embeds], dim=1)
            # Convert attention_mask to float for GPT-2 compatibility
            combined_attention_mask = torch.cat([
                attention_mask.float(),
                torch.ones(batch_size, dec_len, dtype=torch.float, device=attention_mask.device)
            ], dim=1)
            
            # Create position_ids explicitly to ensure correct device
            seq_length = combined_embeds.size(1)
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=combined_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Clamp position_ids to valid range to avoid index out of bounds
            # GPT-2 wpe has n_positions embeddings
            max_position = self.model.config.n_positions - 1
            position_ids = torch.clamp(position_ids, 0, max_position)
            
            model_outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            
            # Extract hidden states
            all_hidden_states = model_outputs.hidden_states[-1]
            
            # Decoder outputs: last code_length positions
            decoder_outputs = all_hidden_states[:, input_len:input_len+dec_len, :]
            
            # Sequence latents: mean pooling over input sequence
            seq_latents = all_hidden_states[:, :input_len, :].clone()
            seq_latents[~attention_mask] = 0
            seq_last_latents = torch.sum(seq_latents, dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
            
        else:
            # T5: Use encoder-decoder architecture
            model_outputs = self.model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                output_hidden_states=True,
                encoder_outputs=encoder_outputs
            )

            decoder_outputs = model_outputs.decoder_hidden_states[-1]
            
            # Sequence latents from encoder
            seq_latents = model_outputs.encoder_last_hidden_state.clone()
            seq_latents[~attention_mask] = 0
            seq_last_latents = torch.sum(seq_latents, dim=1) / attention_mask.sum(dim=1).unsqueeze(1)

        # Compute code logits (same for both architectures)
        code_logits = []
        for i in range(min(decoder_inputs_embeds.shape[1], self.code_length)):
            centroid = self.token_embeddings[i].weight.t()
            code_logits.append(torch.matmul(decoder_outputs[:, i], centroid))
        
        code_logits = torch.stack(code_logits, dim=1) # (batch, code_len, code_num)
        
        # Project sequence latents
        seq_project_latents = self.enc_adapter(seq_last_latents)
        
        # Decoder latents (first position)
        dec_latents = decoder_outputs[:, 0, :].clone()
        dec_latents = self.dec_adapter(dec_latents)
        
        outputs = QuantizeOutput(
            logits=code_logits,
            seq_latents=seq_last_latents,
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents
        )
        return outputs
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, n_return_sequences: int = 1,
                 prefix_allowed_tokens_fn=None) -> torch.Tensor:
        """
        Generates sequences using beam search algorithm.

        Args:
            batch (dict): A dictionary containing input_ids and attention_mask.
            n_return_sequences (int): The number of sequences to generate.

        Returns:
            torch.Tensor: The generated sequences.
        """
        if prefix_allowed_tokens_fn is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=self.code_length+1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )
        else:
            outputs = self.my_beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.code_length+1,
                num_beams=self.num_beams,
                num_return_sequences=n_return_sequences,
                return_score=False
            )
        outputs = outputs[:, 1:].reshape(-1, n_return_sequences, self.code_length)
        return outputs

    def my_beam_search(
        self,
        input_ids,
        attention_mask,
        max_length=6,
        num_beams=1,
        num_return_sequences=1,
        return_score=False
    ):
        """
        Adapted from Hugging Face's implementation.

        Perform beam search to generate sequences using the specified model. 

        *** This implementation does not include stopping conditions based on end-of-sequence (EOS) tokens. Instead, the
        sequence generation is controlled solely by the `max_length` parameter. ***

        Note: In scenarios where the generation should explicitly detect and respond to EOS tokens 
        to terminate the sequence early, this function would need modifications. In the current setup,
        setting `max_length` to a suitable fixed value (e.g., 6) can serve the purpose by limiting
        the maximum sequence length.

        Parameters:
        - input_ids (torch.Tensor): Tensor of input ids.
        - attention_mask (torch.Tensor): Tensor representing the attention mask.
        - max_length (int): Maximum length of the sequence to be generated; controls when to stop extending the sequence.
        - num_beams (int): Number of beams for beam search.
        - num_return_sequences (int): Number of sequences to return.
        - return_score (bool): If True, returns a tuple of (sequences, scores) where 'scores' are the average log likelihood of the returned sequences.

        Returns:
        - torch.Tensor: The final decoder input ids from the beam search, or a tuple of (decoder_input_ids, scores) if 'return_score' is True.

        Example usage:
        # Assuming the model, input_ids, and attention_mask are predefined:
        sequences = beam_search(model, input_ids, attention_mask, max_length=6, num_beams=5, num_return_sequences=5)
        """

        batch_size = input_ids.shape[0]

        # Prepare beam search inputs
        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )
        
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        # For T5: Store encoder_outputs to prevent running full forward path repeatedly
        # For GPT-2: encoder_outputs will be None and we pass inputs_embeds each time
        encoder_outputs = None
        if not self.is_decoder_only:
            with torch.no_grad():
                encoder_outputs = self.get_encoder()(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )

        # Beam search loop
        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                if self.is_decoder_only:
                    # GPT-2: pass input_ids and attention_mask each time
                    outputs = self.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids
                    )
                else:
                    # T5: use cached encoder_outputs
                    outputs = self.forward(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids
                    )

            decoder_input_ids, beam_scores = self.beam_search_step(
                outputs.logits,
                decoder_input_ids,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams
            )

        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], \
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)

        return decoder_input_ids[selection_mask.view(-1), :]

    def prepare_beam_search_inputs(self, input_ids, attention_mask, batch_size, num_beams):
        """
        Adapted from Hugging Face's implementation.

        Prepares and duplicates the input data for beam search decoding.

        This function initializes decoder input IDs and beam scores, creates an offset for beam indices, 
        and expands the input_ids and attention_mask tensors to accommodate the specified number of beams for each instance in the batch.

        Parameters:
        - input_ids (torch.Tensor): The input IDs tensor of shape (batch_size, sequence_length) used for the encoder part of the model.
        - attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, sequence_length) indicating to the model which tokens should be attended to.
        - batch_size (int): The number of instances per batch in the input data.
        - num_beams (int): The number of beams to use in beam search. This expands the input data and scores accordingly.

        Returns:
        - input_ids (torch.Tensor): The expanded input IDs tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - attention_mask (torch.Tensor): The expanded attention mask tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - initial_decoder_input_ids (torch.Tensor): The initialized decoder input IDs for each beam, shape (batch_size * num_beams, 1).
        - initial_beam_scores (torch.Tensor): The initialized scores for each beam, flattened to a single dimension, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): An offset for each beam index to assist in reordering beams during the search, shape (batch_size * num_beams,).

        Each input sequence is replicated 'num_beams' times to provide separate candidate paths in beam search. Beam scores are initialized with 0 for the first beam and a very low number (-1e9) for others to ensure the first token of each sequence is chosen from the first beam.
        """

        decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=input_ids.device, dtype=torch.long)
        # Use appropriate start token based on model type
        if self.is_decoder_only:
            start_token_id = getattr(self.config, 'bos_token_id', 0)
        else:
            start_token_id = self.config.decoder_start_token_id
        
        if start_token_id is None:
            start_token_id = 0
        
        initial_decoder_input_ids = decoder_input_ids * start_token_id

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9  # Set a low score for all but the first beam to ensure the first beam is selected initially
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))

        beam_idx_offset = torch.arange(batch_size, device=input_ids.device).repeat_interleave(num_beams) * num_beams

        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset


    def beam_search_step(self, logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
        """
        Adapted from Hugging Face's implementation.

        Executes one step of beam search, calculating the next set of input IDs based on logits from a model.

        This function expands the current beam, calculates scores for all possible next tokens, selects the top tokens for each beam, and prepares the input IDs for the next iteration of the model. It utilizes logits output by the model to determine the most likely next tokens and updates the beam scores.

        Parameters:
        - logits (torch.Tensor): Logits returned from the model, shape (batch_size * num_beams, sequence_length, vocab_size).
        - decoder_input_ids (torch.Tensor): Current decoder input IDs, shape (batch_size * num_beams, current_sequence_length).
        - beam_scores (torch.Tensor): Current scores for each beam, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): Index offsets for each beam to handle batches correctly, shape (batch_size * num_beams,).
        - batch_size (int): Number of sequences being processed in a batch.
        - num_beams (int): Number of beams used in the beam search.

        Returns:
        - decoder_input_ids (torch.Tensor): Updated decoder input IDs after adding the next tokens, shape (batch_size * num_beams, current_sequence_length + 1).
        - beam_scores (torch.Tensor): Updated scores for each beam, shape (batch_size * num_beams,).

        The function selects the top `2 * num_beams` tokens from the logits based on their scores, reshapes and adjusts them based on the existing beam scores, and determines the next tokens to add to each beam path. The updated paths are then returned for use in the next iteration of the beam search.
        """
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)  # Calculate log softmax over the last dimension

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        # beam_idx_offset: beam_idx contains sequence indicies relative to each individual batch. We need to offset the indicies to retrieve the correct sequence in the corresponding batch
        # for example, when batch_size = 2, beam_size = 3, beam_idx_offset = [0, 0, 0, 3, 3, 3]
        decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        return decoder_input_ids, beam_scores

    