from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import gelu
from transformers import RobertaPreTrainedModel, RobertaModel, XLMRobertaConfig
from transformers.utils import ModelOutput


class RobertaForTranslationAndQEOutput(ModelOutput):
    """
    class for outputs of token classification and regression models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        translation_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        token_qe_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    translation_logits: Optional[torch.FloatTensor] = None
    token_qe_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForTranslationAndQE(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.translation_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.qe_head = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        translation_labels: Optional[torch.LongTensor] = None,
        token_qe_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], RobertaForTranslationAndQEOutput]:
        r"""
        translation_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        token_qe_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        loss_fct = CrossEntropyLoss()
        loss = None

        if translation_labels is not None:
            translation_logits = self.translation_head(sequence_output)
            loss = loss_fct(translation_logits.view(-1, self.vocab_size), translation_labels.view(-1))
            if not return_dict:
                output = (translation_logits, ) + outputs[2:]
                return ((loss,) + output) if (loss is not None) else output

            return RobertaForTranslationAndQEOutput(
                loss=loss,
                translation_logits=translation_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        if token_qe_labels is not None:
            token_qe_logits = self.qe_head(sequence_output)
            loss = loss_fct(token_qe_logits.view(-1, self.num_labels), token_qe_labels.view(-1))
            if not return_dict:
                output = (token_qe_logits, ) + outputs[2:]
                return ((loss,) + output) if (loss is not None) else output

            return RobertaForTranslationAndQEOutput(
                loss=loss,
                token_qe_logits=token_qe_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class XLMRobertaForTranslationAndQE(RobertaForTranslationAndQE):
    """
    This class overrides [`RobertaForTranslationAndQE`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
