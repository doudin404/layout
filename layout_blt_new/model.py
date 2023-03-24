import torch
import torch.nn as nn
import torch.nn.functional as F


class BLTConfig:
    """ base BLT config, params common to all GPT versions"""
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class BLT(nn.Module):
    def __init__(self, mconf):
        super(BLT, self).__init__()
        self.mconf = mconf
        self.embedder = LayoutEmbed(
            embed_size=mconf.embed_size,
            hidden_dropout_prob=mconf.hidden_dropout_prob,
            vocab_size=mconf.vocab_size,
            max_position_embeddings=mconf.max_position_embeddings,
            pad_token=mconf.pad_token
        )
        """mask_lens = torch.ceil(lens * mask_rate).type(torch.int64)"""
        self.bert_layers = nn.Sequential(*[BertLayer(intermediate_size=mconf.intermediate_size,
                                                     embed_size=mconf.embed_size,
                                                     hidden_dropout_prob=mconf.hidden_dropout_prob,
                                                     n_head=mconf.n_head,
                                                     attention_probs_dropout_prob=mconf.attention_probs_dropout_prob) for i in range(mconf.n_layer)])
        self.bert_mlm = BertMlmLayer(
            embed_size=mconf.embed_size,
            vocab_size=mconf.vocab_size
        )

    def configure_optimizers(self, train_config):
        """
        一个用来选择需要weight_decay的变量的函数
        """
        # 定义优化器的超参数
        weight_decay = train_config.weight_decay
        lr = train_config.learning_rate
        betas = train_config.betas

        # 创建AdamW优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        """
        
        decay_params = []
        no_decay_params = []
        for pn, p in self.named_parameters():
            if pn.endswith("weight") and isinstance(self._modules[pn.split(".")[0]], (torch.nn.Linear,)):
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        optim_groups = [{"params": decay_params, "weight_decay": train_config.weight_decay},
                        {"params": no_decay_params, "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        """
        
        return optimizer

    def forward(self, input_ids, targets=None, masks=None):
        input_ids = input_ids.long()
        pad_mask = (input_ids == self.mconf.pad_token)

        input_embeddings = self.embedder(input_ids)
        layer_input = input_embeddings

        for bert_layer in self.bert_layers:
            layer_input = bert_layer(layer_input, pad_mask)

        layer_output = layer_input

        logits = self.bert_mlm(
            layer_output)

        if targets is None:
            loss = 0
        else:
            if masks is not None:
                masks = masks.view(-1)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1))[masks], targets.view(-1)[masks])
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


class LayoutEmbed(nn.Module):
    def __init__(self, embed_size, hidden_dropout_prob, vocab_size, max_position_embeddings, pad_token, hidden_size=None):
        '''
        布局嵌入。

        有四种类型的布局生成嵌入。

            word_embedder:布局序列令牌的嵌入。
            asset_embeder:布局序列中不同元素的正序嵌入。
            backword_embedder:布局序列中不同元素的逆序嵌入。
            attribute_embedder:属于不同元素的相同属性将共享相同的嵌入，

        属性:
        embed_size:嵌入维数。
        hidden_dropout_prob:丢弃率。
        vocab_size:词汇量。
        max_position_embeddings:布局序列中的最大位置。
        embed_size:嵌入维数投影长度
        (上述改动以及原代码中的位置编码方案都不如最简单的词编码+位置编码方案)
        '''
        layout_dim = 2

        super(LayoutEmbed, self).__init__()
        self.embed_size = embed_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.layout_dim = layout_dim
        self.hidden_size = hidden_size
        self.pad_token = pad_token
        # Token embeddings.
        self.word_embedder = nn.Embedding(vocab_size, embed_size)

        # Position embeddings.
        #self.position_embedder = nn.Embedding(
        #    max_position_embeddings, embed_size)
        self.position_ids = torch.arange(max_position_embeddings).unsqueeze(0)

        # backword_embddings.
        self.backword_embdder = nn.Embedding(
            max_position_embeddings//(self.layout_dim * 2 + 1)+1, embed_size)

        # Asset segment embeddings.
        self.asset_embedder = nn.Embedding(
            max_position_embeddings//(self.layout_dim * 2 + 1)+1, embed_size)

        self.attribute_embedder = nn.Embedding(#特征嵌入
            self.layout_dim * 2 + 1, embed_size)

        self.layer_norm = nn.LayerNorm(embed_size)

        if self.hidden_size:
            self.dense = nn.Linear(embed_size, hidden_size)

    def forward(self, input_ids):
        device = input_ids.device
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length].to(device)
        asset_ids = position_ids // (self.layout_dim * 2 + 1)
        asset_num = torch.sum(
            input_ids != self.pad_token, dim=1).unsqueeze(-1) // (self.layout_dim * 2 + 1)
        attribute_ids = position_ids % (self.layout_dim * 2 + 1)


        backword_ids=torch.clamp(asset_num - asset_ids, min=0)
        backword_embeddings=self.backword_embdder(backword_ids)



        word_embeddings = self.word_embedder(input_ids)
        #position_embeddings = self.position_embedder(position_ids)
        asset_embeddings = self.asset_embedder(asset_ids)
        attribute_embeddings = self.attribute_embedder(attribute_ids)
        #asset_num_embeddings = self.backword_embdder(asset_num)
        input_embeddings = word_embeddings + attribute_embeddings +  asset_embeddings + backword_embeddings
        #input_embeddings = word_embeddings + attribute_embeddings +  asset_embeddings + backword_embeddings
             #+ asset_embeddings + asset_num_embeddings +position_embedding

        input_embeddings = self.layer_norm(input_embeddings)
        if self.hidden_size:
            input_embeddings = self.dense(input_embeddings)
        input_embeddings = F.dropout(
            input_embeddings, p=self.hidden_dropout_prob, training=self.training)

        return input_embeddings


class BertMlmLayer(nn.Module):
    """Bert layer for masked token prediction."""

    def __init__(self, embed_size, vocab_size):
        super(BertMlmLayer, self).__init__()
        self.embed_size = embed_size
        self.mlm_dense = nn.Linear(embed_size, self.embed_size)
        self.mlm_ln = nn.LayerNorm(self.embed_size)
        self.decoder_layer = nn.Linear(embed_size,vocab_size)

    def forward(self, last_layer):
        mlm_hidden = self.mlm_dense(last_layer)
        mlm_hidden = F.gelu(mlm_hidden)
        mlm_hidden = self.mlm_ln(mlm_hidden)
        
        logits = self.decoder_layer(mlm_hidden)

        return logits


class BertLayer(nn.Module):
    """A single Bert layer."""

    def __init__(self, embed_size, intermediate_size, hidden_dropout_prob,
                 n_head, attention_probs_dropout_prob):
        super(BertLayer, self).__init__()

        self.bert_attention = BertAttention(
            embed_size=embed_size,
            hidden_dropout_prob=hidden_dropout_prob,
            n_head=n_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob)

        self.bert_mlp = BertMlp(
            embed_size=embed_size,
            hidden_dropout_prob=hidden_dropout_prob,
            intermediate_size=intermediate_size)

    def forward(self, layer_input, pad_mask):
        attention_output = self.bert_attention(layer_input, pad_mask)
        layer_output = self.bert_mlp(attention_output)
        return layer_output


class BertMlp(nn.Module):
    """BERT MLP layer that is part of each BERT layer."""

    def __init__(self, embed_size, hidden_dropout_prob, intermediate_size):
        super(BertMlp, self).__init__()
        self.intermediate_output = nn.Linear(embed_size, intermediate_size)
        self.layer_output = nn.Linear(intermediate_size, embed_size)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, attention_output):
        # Bert intermediate layer.
        intermediate_output = self.intermediate_output(attention_output)
        intermediate_output = torch.nn.functional.gelu(intermediate_output)

        # Bert output layer.
        layer_output = self.layer_output(intermediate_output)
        layer_output = F.dropout(
            layer_output, p=self.hidden_dropout_prob, training=self.training)
        layer_output = self.layer_norm(layer_output + attention_output)

        return layer_output


class BertAttention(nn.Module):
    def __init__(self, embed_size, hidden_dropout_prob, n_head, attention_probs_dropout_prob):
        super(BertAttention, self).__init__()
        self.hidden_dropout_prob = hidden_dropout_prob
        self.self_attention = nn.MultiheadAttention(
            embed_size, n_head, dropout=attention_probs_dropout_prob, batch_first=True)
        self.attention_probs_dropout_prob=attention_probs_dropout_prob
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, layer_input, pad_mask):


        attention_output, _ = self.self_attention(
            layer_input, layer_input, layer_input, key_padding_mask=pad_mask)

        attention_output = F.dropout(
            attention_output, p=self.hidden_dropout_prob, training=self.training)

        attention_output = self.layer_norm(attention_output + layer_input)

        return attention_output
