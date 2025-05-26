# Adapted from https://github.com/yd-kwon/POMO
# Original Author: Yoon Dae Kwon
# License: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

use_bias = True


class TSPModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.encoder = TSP_Encoder(config.embedding_dim, config.head_num, config.qkv_dim, config.ff_hidden_dim, config.encoder_layer_num)
        self.decoder = TSP_Decoder(config.embedding_dim, config.head_num, config.qkv_dim, config.latent_dim, config.logit_clipping)
        self.encoded_nodes = None
        # shape: (batch, problem, embedding)

        self.fc1 = nn.Linear(config.embedding_dim, 2 * config.embedding_dim)
        self.fc2 = nn.Linear(2 * config.embedding_dim, config.hidden_dim)
        self.mu_fc = nn.Linear(config.hidden_dim, config.latent_dim)
        self.log_var_fc = nn.Linear(config.hidden_dim, config.latent_dim)

        self.mu_act = Generalized_SoftClipping(s1=-10.0, s2=10.0, s=5.0, threshold=20)
        self.log_var_act = Generalized_SoftClipping(s1=-10.0, s2=10.0, s=5.0, threshold=20)

    def encode(self, reset_state):
        node_xy = reset_state.node_xy.to(self.device)
        # shape: (batch, problem, 2)

        self.encoded_nodes = self.encoder(node_xy)
        # shape: (batch, problem, embedding)
        self.decoder.set_kv(self.encoded_nodes)

        self.graph_embedding = self.encoded_nodes.mean(dim=1)
        # shape: (batch, embedding)
        hidden = torch.relu(self.fc2(torch.relu(self.fc1(self.graph_embedding))))
        self.mu = self.mu_fc(hidden)
        self.log_var = self.log_var_fc(hidden)

        self.mu = 10*torch.tanh(self.mu)
        #self.mu = self.mu_act(self.mu)
        self.log_var = -F.softplus(self.log_var)
        #self.log_var = self.log_var_act(self.log_var)

        mu = self.mu.unsqueeze(1).repeat(1, self.config.K, 1)
        log_var = self.log_var.unsqueeze(1).repeat(1, self.config.K, 1)
        epsilon = torch.randn_like(log_var)
        self.z = mu + torch.exp(0.5*log_var) * epsilon

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        K = state.BATCH_IDX.size(1)

        if state.current_node is None:
            if self.config.problem_size == K:
                selected = torch.arange(K)[None, :].expand(batch_size, K).to(self.device)
                prob = torch.ones(size=(batch_size, K)).to(self.device)

            else:
                selected = torch.randint(low=0, high=self.config.problem_size, size=(batch_size, K)).to(self.device)
                prob = torch.ones(size=(batch_size, K)).to(self.device)

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, K, embedding)
            self.decoder.set_q1(encoded_first_node, self.z)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node.to(self.device))
            # shape: (batch, K, embedding)
            probs = self.decoder(encoded_last_node, self.z, ninf_mask=state.ninf_mask)
            # shape: (batch, K, problem)

            if self.training and self.config.eval_type == 'sampling':
                while True:  
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * K, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, K)
                    # shape: (batch, K)
                    prob = probs[state.BATCH_IDX, state.K_IDX, selected].reshape(batch_size, K)
                    # shape: (batch, K)
                    if (prob != 0).all():
                        break

            elif self.training and self.config.eval_type == 'greedy':
                selected = probs.argmax(dim=2)
                # shape: (batch, K)
                prob = probs[state.BATCH_IDX, state.K_IDX, selected].reshape(batch_size, K)

            else:
                test_eval_type = 'greedy'
                if test_eval_type == 'greedy':
                    selected = probs.argmax(dim=2)
                    # shape: (batch, K)
                    prob = probs[state.BATCH_IDX, state.K_IDX, selected].reshape(batch_size, K) 

                if test_eval_type == 'sampling':
                    while True:  
                        with torch.no_grad():
                            selected = probs.reshape(batch_size * K, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, K)
                        # shape: (batch, K)
                        prob = probs[state.BATCH_IDX, state.K_IDX, selected].reshape(batch_size, K)
                        # shape: (batch, K)
                        if (prob != 0).all():
                            break


        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, K)

    batch_size = node_index_to_pick.size(0)
    K = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, K, embedding_dim)
    # shape: (batch, K, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, K, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, ff_hidden_dim, encoder_layer_num):
        super().__init__()

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim) for _ in range(encoder_layer_num)])

    def forward(self, node_xy):
        # node_xy.shape: (batch, problem, 2)

        embedded_node = self.embedding(node_xy)
        # shape: (batch, problem, embedding)

        out = embedded_node
        # shape: (batch, problem, embedding)
        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, ff_hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=use_bias)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=use_bias)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=use_bias)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)
        self.add_n_normalization_2 = AddAndInstanceNormalization(embedding_dim)

    def forward(self, input1):
        # input1.shape: (batch, problem, embedding)
        head_num = self.head_num

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, latent_dim, logit_clipping):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.logit_clipping = logit_clipping

        self.Wq_first = nn.Linear(embedding_dim+latent_dim, head_num * qkv_dim, bias=use_bias)
        self.Wq_last = nn.Linear(embedding_dim+latent_dim, head_num * qkv_dim, bias=use_bias)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=use_bias)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=use_bias)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.head_num

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1, z):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or K
        head_num = self.head_num
        input_cat = torch.cat((encoded_q1, z), dim=2)
        self.q_first = reshape_by_heads(self.Wq_first(input_cat), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, z, ninf_mask):
        # encoded_last_node.shape: (batch, K, embedding)
        # z.shape: (batch, K, latent_dim)
        # ninf_mask.shape: (batch, K, problem)

        head_num = self.head_num

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, z), dim=2)
        # shape = (batch, group, embedding+latent_dim)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, K, qkv_dim)

        q = self.q_first + q_last
        # shape: (batch, head_num, K, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, K, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, K, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, K, problem)

        sqrt_embedding_dim = self.embedding_dim**(1/2)
        logit_clipping = self.logit_clipping

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, K, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, K, problem)

        return probs
    

########################################
# DECODER FOR INFERENCE
########################################

class TSP_Decoder_modified(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, latent_dim, logit_clipping):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.logit_clipping = logit_clipping

        self.Wq_first = nn.Linear(embedding_dim+latent_dim, head_num * qkv_dim, bias=use_bias)
        self.Wq_last = nn.Linear(embedding_dim+latent_dim, head_num * qkv_dim, bias=use_bias)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=use_bias)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=use_bias)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape = (batch, problem+1, embedding)
        head_num = self.head_num

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape = (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2).detach()
        self.single_head_key.requires_grad = True
        # shape = (batch, embedding, problem+1)

    def set_q1(self, encoded_q1, z):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or K
        # z.shape: (batch, K, latent_dim)
        head_num = self.head_num
        self.encoded_q1 = encoded_q1
        input_cat = torch.cat((self.encoded_q1, z), dim=2)
        self.q_first = reshape_by_heads(self.Wq_first(input_cat), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, z, ninf_mask):
        # encoded_last_node.shape: (batch, K, embedding)
        # z.shape: (batch, K, latent_dim)
        # ninf_mask.shape: (batch, K, problem)
        head_num = self.head_num

        with torch.no_grad():

            #  Multi-Head Attention
            #######################################################
            input_cat = torch.cat((encoded_last_node, z), dim=2)
            # shape = (batch, K, embedding+latent_dim)

            q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
            # shape: (batch, head_num, K, qkv_dim)

            q = self.q_first + q_last
            # shape: (batch, head_num, K, qkv_dim)

            out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
            # shape: (batch, K, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch, K, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key.detach())
        # shape = (batch, K, problem)

        sqrt_embedding_dim = self.embedding_dim**(1/2)
        logit_clipping = self.logit_clipping

        score_scaled = score / sqrt_embedding_dim
        # shape = (batch, K, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch, K, problem)

        return probs


def replace_decoder(model, state, config):
    model.decoder = TSP_Decoder_modified(config.embedding_dim, config.head_num, config.qkv_dim, config.latent_dim, config.logit_clipping)
    model.decoder.load_state_dict(state_dict=state, strict=False)
    return model


########################################
# NN SUB CLASS / FUNCTIONS
########################################

class Generalized_SoftClipping(nn.Module):
    def __init__(self, s1: float = -50.0, s2: float = 50.0, s: float = 1.0, threshold: float = 20.0) -> None:
        super(Generalized_SoftClipping, self).__init__()
        self.s1 = s1  
        self.s2 = s2
        self.s = s
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1 = F.softplus(x - self.s1, beta=self.s, threshold=self.threshold)
        term2 = F.softplus(x - self.s2, beta=self.s, threshold=self.threshold)
        return term1 - term2 + self.s1


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ff_hidden_dim = ff_hidden_dim

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
