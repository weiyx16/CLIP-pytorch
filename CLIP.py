import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class QuickGELU(nn.Module):
    def __init__(self):
        super(QuickGELU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
        super(MultiheadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.in_proj_weight = nn.Parameter(torch.ones((3 * self.all_head_size, self.all_head_size)), requires_grad=True)
        self.in_proj_bias = nn.Parameter(torch.zeros((3 * self.all_head_size)), requires_grad=True)

        self.dropout_att = nn.Dropout(attention_probs_dropout_prob, inplace=True)
        # self.dropout_hid = nn.Dropout(hidden_dropout_prob)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.query_scale = torch.tensor(0.125, dtype=torch.float64)


    def transpose_for_scores(self, x):
        x = x.contiguous().view(-1, x.size(1)*self.num_attention_heads, self.attention_head_size)
        return x.transpose(0, 1)

    def forward(self, hidden_states):
        # len*bs*(3*hidden_size)
        mixed_layer = torch.matmul(hidden_states, self.in_proj_weight.t()) + self.in_proj_bias
        mixed_query_layer, mixed_key_layer, mixed_value_layer = torch.chunk(mixed_layer, 3, -1)
        # Special case: 0.125 for both visual and text
        mixed_query_layer = torch.mul(mixed_query_layer, self.query_scale)

        # (head_num*bs)*len*head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.bmm(query_layer, key_layer.transpose(1, 2))
        # TODO: No softmax sqrt(d)
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.num_attention_heads == 8:
            # text branch need attention mask
            attention_mask = torch.tril(torch.ones(1, hidden_states.size(0), hidden_states.size(0)), diagonal=0)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype, device=hidden_states.device)
            attention_mask = (1.0 - attention_mask) * -10000000.0
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout_att(attention_probs)
        context_layer = torch.bmm(attention_probs, value_layer)

        context_layer = context_layer.transpose(0, 1).contiguous()
        context_layer = context_layer.view(hidden_states.size())

        context_layer = self.out_proj(context_layer)
        # context_layer = self.dropout_hid(context_layer)
        return context_layer


class ResidualAttentionBlock(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = MultiheadAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)
        self.ln_1 = nn.LayerNorm(hidden_size, eps=1e-5)

        self.mlp = nn.Sequential(OrderedDict([  
                                    ('c_fc', nn.Linear(hidden_size, 4*hidden_size)),  
                                    ('gelu', QuickGELU()),  
                                    ('c_proj', nn.Linear(4*hidden_size, hidden_size)),  
                                ]))
        # self.dropout_hid = nn.Dropout(hidden_dropout_prob)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        # different layernorm location: https://github.com/openai/gpt-2/blob/master/src/model.py#L123
        # https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
        self.ln_1 = self.ln_1.float()
        hidden_states_to_ln = self.ln_1(hidden_states.type(torch.float32)).type(hidden_states.dtype)
        
        attn_outputs = self.attn(hidden_states_to_ln)
        hidden_states = attn_outputs + hidden_states

        self.ln_2 = self.ln_2.float()
        hidden_states_to_ln = self.ln_2(hidden_states.type(torch.float32)).type(hidden_states.dtype)

        mlp_outputs = self.mlp(hidden_states_to_ln)
        # mlp_outputs = self.dropout_hid(mlp_outputs)
        hidden_states = mlp_outputs + hidden_states
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, num_hidden_layers=12, hidden_size=768, num_attention_heads=12, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
        super(Transformer, self).__init__()
        layers = [ResidualAttentionBlock(hidden_size=hidden_size, 
                                        num_attention_heads=num_attention_heads, 
                                        attention_probs_dropout_prob=attention_probs_dropout_prob, 
                                        hidden_dropout_prob=hidden_dropout_prob) for _ in range(num_hidden_layers)]
        self.resblocks = nn.Sequential(*layers)
    

    def forward(self, hidden_states):
        embedding = self.resblocks(hidden_states)
        return embedding


class VisualTransformer(nn.Module):
    """This support for VIT-32 with 224*224 input only (seq:7*7+1=50)
    `input`: (bs*49) * 3 * 32 * 32
    """
    def __init__(self, patch_number=49, patch_size=32, proj_size=512, num_hidden_layers=12, hidden_size=768, num_attention_heads=12, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
        super(VisualTransformer, self).__init__()
        self.class_embedding =  nn.Parameter(torch.ones((hidden_size)), requires_grad=True)
        self.positional_embedding = nn.Parameter(torch.ones((patch_number+1, hidden_size)), requires_grad=True)
        self.conv1 = nn.Conv2d(3, hidden_size, patch_size, bias=False)
        self.ln_pre = nn.LayerNorm(hidden_size, eps=1e-5)
        self.transformer = Transformer(num_hidden_layers=num_hidden_layers,
                                        hidden_size=hidden_size, 
                                        num_attention_heads=num_attention_heads, 
                                        attention_probs_dropout_prob=attention_probs_dropout_prob, 
                                        hidden_dropout_prob=hidden_dropout_prob)
        self.ln_post = nn.LayerNorm(hidden_size, eps=1e-5)
        self.proj = nn.Parameter(torch.ones((hidden_size, proj_size)), requires_grad=True)

        self.patch_number = patch_number
        self.hidden_size = hidden_size
    

    def forward(self, input):
        # input: (49*bs)*3*32*32 to (49*bs)*768
        input = input.type(self.class_embedding.dtype)
        input = self.conv1(input)
        # bs*49*768
        vis_emb = input.reshape(-1, self.patch_number, self.hidden_size)
        # bs*1*768
        cls_emb = self.class_embedding.unsqueeze(0).unsqueeze(1).repeated(vis_emb.size(0), 1, 1)
        # bs*50*768
        seq_emb = torch.cat((cls_emb, vis_emb), dim=1)
        # 1*50*768
        pos_emb = self.positional_embedding.unsqueeze(0)
        # bs*50*768
        input_emb = torch.add(seq_emb, pos_emb)
        self.ln_pre = self.ln_pre.float()
        input_emb = self.ln_pre(input_emb.type(torch.float32)).type(input_emb.dtype)
        
        # bs*50*768
        output_emb = self.transformer(input_emb)
        # bs*768
        output = output_emb[:, 0, :].view(-1, output_emb.size(-1))
        # bs*512
        self.ln_post = self.ln_post.float()
        output = self.ln_post(output.type(torch.float32)).type(input_emb.dtype)
        return torch.matmul(output, self.proj)


class CLIP(nn.Module):
    def __init__(self, patch_number=49, patch_size=32, proj_size=512, num_hidden_layers=12, visual_hidden_size=768, text_hidden_size=512, visual_num_attention_heads=12, text_num_attention_heads=8, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
        super(CLIP, self).__init__()
        self.visual = VisualTransformer(patch_number=patch_number,
                                        patch_size=patch_size,
                                        proj_size=proj_size,
                                        num_hidden_layers=num_hidden_layers,
                                        hidden_size=visual_hidden_size, 
                                        num_attention_heads=visual_num_attention_heads, 
                                        attention_probs_dropout_prob=attention_probs_dropout_prob, 
                                        hidden_dropout_prob=hidden_dropout_prob)
        self.transformer = Transformer(num_hidden_layers=num_hidden_layers,
                                        hidden_size=text_hidden_size, 
                                        num_attention_heads=text_num_attention_heads, 
                                        attention_probs_dropout_prob=attention_probs_dropout_prob, 
                                        hidden_dropout_prob=hidden_dropout_prob)

        self.context_length = nn.Parameter(torch.tensor(77), requires_grad=False)
        self.input_resolution = nn.Parameter(torch.tensor(224), requires_grad=False)
        self.vocab_size = nn.Parameter(torch.tensor(49408), requires_grad=False)
        self.token_embedding = nn.Embedding(int(self.vocab_size.item()), text_hidden_size)
        self.ln_final = nn.LayerNorm(text_hidden_size, eps=1e-5)
        self.positional_embedding = nn.Parameter(torch.ones((int(self.context_length.item()), text_hidden_size)), requires_grad=True)
        self.text_projection = nn.Parameter(torch.ones((text_hidden_size, text_hidden_size)), requires_grad=True)
        # TODO: why is learnable?
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)

    def encode_text(self, input):
        """
        `input`: bs*self.context_length
        """
        # bs*77*512
        token_emb = self.token_embedding(input)

        # bs*77
        # TODO: maybe not solid to pick the argmax one...
        seq_end = torch.argmax(input, dim=1, keepdim=True)

        grid_ind, grid_pos = torch.meshgrid(torch.arange(seq_end.size(0), dtype=torch.long, device=token_emb.device),
                                            torch.arange(self.context_length.item(), dtype=torch.long, device=token_emb.device))
        # 1*77*512
        pos_emb = self.positional_embedding.unsqueeze(0)

        # bs*77*512 to permute 77*bs*512
        input_emb = torch.add(token_emb, pos_emb).permute(1, 0, 2)

        # bs*77*512
        output_emb = self.transformer(input_emb).permute(1, 0, 2)
        
        #   For computational efficiency, the max sequence length was capped at 76. The text sequence is bracketed with [SOS] and [EOS] tokens 
        # and the activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text
        # which is layer normalized and then linearly projected into the multi-modal embedding space.
        self.ln_final = self.ln_final.float()
        output = self.ln_final(output_emb.type(torch.float32)).type(token_emb.dtype)

        output = torch.matmul(output[grid_pos == seq_end].view(-1, output.size(-1)), self.text_projection.to(device=token_emb.device))
        return output

    def encode_image(self, image):
        """
        `image`: (bs*49)*3*32*32
        `output`: bs*512
        """
        output = self.visual(image)
        return output

    def forward(self, image, input):
        text_latents = self.encode_text(input)
        image_latents = self.encode_image(image)
        logit_scale = self.logit_scale.exp()

        text_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, image_latents))
        labels = torch.arange(input.size(0), device = image_latents.device)
        sim_i_2_t = torch.matmul(torch.mul(logit_scale, image_latents), torch.t(text_latents))
        sim_t_2_i = torch.matmul(torch.mul(logit_scale, text_latents), torch.t(image_latents))
        
        loss_t_2_i = F.cross_entropy(sim_t_2_i, labels)
        loss_i_2_t = F.cross_entropy(sim_i_2_t, labels)
        
        return sim_i_2_t, sim_t_2_i, loss_i_2_t, loss_t_2_i