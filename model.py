import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import GenerationMixin
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from layers import MLPLayers


@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    suffix_logits: Optional[torch.FloatTensor] = None
    position_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    rank_logits: Optional[torch.FloatTensor] = None
    seq_latents: Optional[torch.FloatTensor] = None
    seq_project_latents: Optional[torch.FloatTensor] = None
    dec_latents: Optional[torch.FloatTensor] = None
    qs_loss: Optional[torch.FloatTensor] = None
        
        
class Model(nn.Module, GenerationMixin):
    def __init__(self, config, model, n_items, code_length=1, code_number=256):
        super().__init__()
        self.model = model
                                                                          
        self._supports_cache_class = getattr(model, '_supports_cache_class', False)
        self.config = model.config
        self.base_model_prefix = "model"
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
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
        
        token_vocab_sizes = [self.code_number] * self.code_length
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.hidden_size)
            for vocab_size in token_vocab_sizes
        ])
        self.token_embeddings.requires_grad_(True)
        
        enc_adapter_layers = config['layers']
        enc_adapter_layers = [self.hidden_size] + [config['e_dim']]
        self.enc_adapter = MLPLayers(layers=enc_adapter_layers)

        dec_adapter_layers = config['layers'][::-1]
        dec_adapter_layers = [self.hidden_size] + [self.semantic_hidden_size]
        self.dec_adapter = MLPLayers(layers=dec_adapter_layers)

                                                                                                        
                                                                                         
        self.qs_projector = nn.Linear(config['e_dim'], self.hidden_size)

                                   
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs, "attention_mask": attention_mask}

    def _shift_right(self, input_ids):
        pad_token_id = self.config.pad_token_id

        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), pad_token_id, device=input_ids.device)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids], dim=-1)

        return shifted_input_ids
    
    def get_input_embeddings(self, input_ids, attention_mask):
        attention_mask_flatten = attention_mask.reshape(-1)

        inputs_embeds = torch.zeros(*input_ids.shape, self.hidden_size, device=self.device)
        input_ids[input_ids==-1] = 0
        for i in range(self.code_length):
            inputs_embeds[:, i::self.code_length] = self.token_embeddings[i](input_ids[:, i::self.code_length])
        
        inputs_embeds = inputs_embeds.view(-1, self.hidden_size)
        inputs_embeds[~attention_mask_flatten] = self.model.shared.weight[0]
        inputs_embeds = inputs_embeds.view(input_ids.shape[0], -1, self.hidden_size)

        return inputs_embeds

    def get_mixture_input_embeddings(
        self, semantic_probabilities, suffix_ids, attention_mask
    ):
        """Build item-major history embeddings without integer semantic IDs."""
        if semantic_probabilities.dim() != 4:
            raise ValueError(
                "semantic_probabilities must have shape [batch, history, level, code]"
            )
        batch_size, history_length, semantic_levels, code_number = (
            semantic_probabilities.shape
        )
        if semantic_levels != self.code_length - 1:
            raise ValueError(
                f"expected {self.code_length - 1} semantic levels, got {semantic_levels}"
            )
        if suffix_ids.shape != (batch_size, history_length):
            raise ValueError("suffix_ids must match the batch/history dimensions")
        if attention_mask.shape != (batch_size, history_length):
            raise ValueError("attention_mask must match the batch/history dimensions")

        item_embeddings = []
        for level in range(semantic_levels):
            if code_number != self.token_embeddings[level].num_embeddings:
                raise ValueError("assignment and recommender codebook sizes differ")
            item_embeddings.append(
                torch.matmul(
                    semantic_probabilities[:, :, level],
                    self.token_embeddings[level].weight,
                )
            )
        item_embeddings.append(
            self.token_embeddings[-1](suffix_ids.clamp_min(0))
        )
        inputs_embeds = torch.stack(item_embeddings, dim=2).reshape(
            batch_size, history_length * self.code_length, self.hidden_size
        )
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand(
            batch_size, history_length, self.code_length
        ).reshape(batch_size, history_length * self.code_length)
        pad_embedding = self.model.shared.weight[0].view(1, 1, -1)
        inputs_embeds = torch.where(
            expanded_attention_mask.unsqueeze(-1), inputs_embeds, pad_embedding
        )
        return inputs_embeds, expanded_attention_mask

    def get_differentiable_decoder_inputs(self, target_probabilities):
        """Teacher-force with target mixtures instead of detached argmax IDs."""
        if target_probabilities.dim() != 3:
            raise ValueError(
                "target_probabilities must have shape [batch, level, code]"
            )
        if target_probabilities.shape[1] != self.code_length - 1:
            raise ValueError(
                f"expected {self.code_length - 1} target levels"
            )
        batch_size = target_probabilities.shape[0]
        start_ids = torch.full(
            (batch_size,),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=target_probabilities.device,
        )
        decoder_embeddings = [self.model.shared(start_ids)]
        for level in range(self.code_length - 1):
            decoder_embeddings.append(
                torch.matmul(
                    target_probabilities[:, level],
                    self.token_embeddings[level].weight,
                )
            )
        return torch.stack(decoder_embeddings, dim=1)

    def compute_differentiable_code_losses(
        self, logits, target_probabilities, suffix_ids, suffix_logits=None
    ):
        """Return semantic and hard-suffix contributions on the old CE scale."""
        semantic_levels = self.code_length - 1
        if suffix_logits is None:
            if logits.shape[1] != self.code_length:
                raise ValueError("recommender logits must contain every code position")
            semantic_logits = logits[:, :semantic_levels]
            suffix_logits = logits[:, semantic_levels]
        else:
            if logits.shape[1] == self.code_length:
                semantic_logits = logits[:, :semantic_levels]
            elif logits.shape[1] == semantic_levels:
                semantic_logits = logits
            else:
                raise ValueError("semantic logits must contain every RQ position")
        if target_probabilities.shape[:2] != (
            logits.shape[0], semantic_levels
        ):
            raise ValueError("target probabilities do not match recommender logits")
        semantic_nll = -(
            target_probabilities
            * F.log_softmax(semantic_logits.float(), dim=-1)
        ).sum(dim=-1)
        denominator = logits.shape[0] * self.code_length
        semantic_loss = semantic_nll.sum() / denominator
        suffix_loss = F.cross_entropy(
            suffix_logits.float(),
            suffix_ids.detach().long(),
            reduction="sum",
        ) / denominator
        return semantic_loss, suffix_loss
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, decoder_input_ids=None,
                decoder_inputs_embeds=None, encoder_outputs=None, quantizer_latent=None,
                token_indices=None, qs_beta=0.25, **kwargs):
        
        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

        if decoder_inputs_embeds is None:
            if decoder_input_ids is None and labels is None:
                decoder_input_ids = torch.zeros(
                    input_ids.size(0), self.code_length
                ).long().to(input_ids.device)
            elif decoder_input_ids is None and labels is not None:
                decoder_input_ids = self._shift_right(labels)

            decoder_inputs_embeds = []
            for i in range(min(decoder_input_ids.shape[1], self.code_length)):
                if i==0:
                    code_embedding = self.model.shared
                else:
                    code_embedding = self.token_embeddings[i-1]         
                decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i]))
            decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)


        model_outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs
        )

        decoder_outputs = model_outputs.decoder_hidden_states[-1]

        code_logits = []
        for i in range(min(decoder_inputs_embeds.shape[1], self.code_length)):
            centroid = self.token_embeddings[i].weight.t()
            code_logits.append(torch.matmul(decoder_outputs[:, i], centroid))
        
        position_logits = tuple(code_logits)
        stacked_logits = torch.stack(position_logits, dim=1)
        suffix_logits = (
            stacked_logits[:, -1]
            if len(position_logits) == self.code_length
            else None
        )
        
        seq_latents = model_outputs.encoder_last_hidden_state.clone()
                      
        seq_latents[~attention_mask] = 0
        seq_last_latents = torch.sum(seq_latents, dim=1) / attention_mask.sum(dim=1).unsqueeze(1)
        seq_project_latents = self.enc_adapter(seq_last_latents)
        
        dec_latents = model_outputs.decoder_hidden_states[-1].clone()
        dec_latents = dec_latents[:,0,:]
        dec_latents = self.dec_adapter(dec_latents)

        qs_loss = None
        if quantizer_latent is not None and token_indices is not None:
            token_embs = torch.stack([
                self.token_embeddings[i](token_indices[:, i])
                for i in range(token_indices.shape[1])
            ], dim=1).mean(dim=1)
            z_projected = self.qs_projector(quantizer_latent)
            qs_loss = F.mse_loss(z_projected, token_embs.detach()) + \
                qs_beta * F.mse_loss(z_projected.detach(), token_embs)
        
        outputs = QuantizeOutput(
            logits=stacked_logits,
            suffix_logits=suffix_logits,
            position_logits=position_logits,
            seq_latents=seq_last_latents,
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents,
            qs_loss=qs_loss,
        )
        return outputs
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, n_return_sequences: int = 1,
                 prefix_allowed_tokens_fn=None) -> torch.Tensor:
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

        batch_size = input_ids.shape[0]

                                    
        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )
        
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)

                                                                               
        with torch.no_grad():
            encoder_outputs = self.get_encoder()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )

                          
        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.forward(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )

            decoder_input_ids, beam_scores = self.beam_search_step(
                outputs.position_logits[-1].unsqueeze(1),
                decoder_input_ids,
                beam_scores,
                beam_idx_offset,
                batch_size,
                num_beams
            )

                                                                             
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], \
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)

        return decoder_input_ids[selection_mask.view(-1), :]

    def prepare_beam_search_inputs(self, input_ids, attention_mask, batch_size, num_beams):

        decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=self.device, dtype=torch.long)
        initial_decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9                                                                                             
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))

        beam_idx_offset = torch.arange(batch_size, device=self.device).repeat_interleave(num_beams) * num_beams

        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset


    def beam_search_step(self, logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)                                                 

        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

                                                                                                                                                                                            
                                                                                               
        decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        return decoder_input_ids, beam_scores
