from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import gelu
from transformers import RobertaPreTrainedModel, RobertaModel, XLMRobertaConfig
from transformers.utils import ModelOutput


class XLMRobertaForTokenClassificationAndRegressionConfig(XLMRobertaConfig):
    def __init__(self, reg_lambda=0.5, **kwargs):
        super().__init__(**kwargs)
        self.reg_lambda = reg_lambda


class RobertaTokenRegressionHead(nn.Module):
    """Head for token-level regression tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForTokenClassificationAndRegressionOutput(ModelOutput):
    """
    class for outputs of token classification and regression models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        token_cls_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        token_reg_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Regression scores.
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
    token_cls_logits: torch.FloatTensor = None
    token_reg_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForTokenClassificationAndRegression(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.reg_head = nn.Linear(config.hidden_size, 1)        # 简单reg头
        # self.reg_head = RobertaTokenRegressionHead(config)    # 复杂reg头

        self.reg_lambda = config.reg_lambda

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
        token_cls_labels: Optional[torch.LongTensor] = None,
        token_reg_labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], RobertaForTokenClassificationAndRegressionOutput]:
        r"""
        cls_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        reg_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token regression loss. Indices should be in `[0, ..., config.num_labels - 1]`.
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
        token_cls_logits = self.classifier(sequence_output)
        token_reg_logits = self.reg_head(sequence_output)

        loss = None

        token_cls_loss = None
        if token_cls_labels is not None:
            loss_fct = CrossEntropyLoss()
            token_cls_loss = loss_fct(token_cls_logits.view(-1, self.num_labels), token_cls_labels.view(-1))

        token_reg_loss = None
        if token_reg_labels is not None:
            loss_fct = MSELoss(reduction='none')
            token_reg_loss_temp = loss_fct(token_reg_logits.view(-1), token_reg_labels.view(-1))
            token_reg_mask = token_reg_labels.ge(0).view(-1)
            token_reg_loss = torch.mean(token_reg_loss_temp * token_reg_mask)
            limit = 5
            limit_tensor = torch.ones_like(token_reg_loss) * limit
            token_reg_loss = torch.where(token_reg_loss > limit, limit_tensor, token_reg_loss)

        if token_cls_loss is not None and token_reg_loss is not None:
            loss = token_cls_loss + token_reg_loss * self.reg_lambda
        elif token_cls_loss is not None:
            loss = token_cls_loss

        if not return_dict:
            output = (token_cls_logits, token_reg_logits, ) + outputs[2:]
            return ((loss,) + output) if (token_cls_loss is not None and token_reg_loss is not None) else output

        return RobertaForTokenClassificationAndRegressionOutput(
            loss=loss,
            token_cls_logits=token_cls_logits,
            token_reg_logits=token_reg_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLMRobertaForTokenClassificationAndRegression(RobertaForTokenClassificationAndRegression):
    """
    This class overrides [`RobertaForTokenClassificationAndRegression`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig

