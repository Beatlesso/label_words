import warnings

import torch


class Predictor:
    def __init__(self, label_id_dict, pad_token_id, task_name, tokenizer, layer,
                 naive_class_embs=None,
                 naive_final_emb=None) -> None:
        self.naive_class_embs = naive_class_embs
        self.naive_final_emb = naive_final_emb
        self.label_id_dict = label_id_dict
        self.pad_token_id = pad_token_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.layer = layer

        # 根据不同的任务，给预测输出加上前缀
        if task_name == 'sst2':
            self.prefix_idxs = [tokenizer.encode('Sentiment', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'agnews':
            self.prefix_idxs = [tokenizer.encode('Answer', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'trec':
            self.prefix_idxs = [tokenizer.encode(' Type', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'emo':
            self.prefix_idxs = [tokenizer.encode('Emotion', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        else:
            raise NotImplementedError(f"task_name: {task_name}")

    def get_pos(self, inputs):
        label_id_dict = self.label_id_dict
        pad_token_id = self.pad_token_id
        final_pos = (inputs['input_ids'] != pad_token_id).int().sum(-1) - 1
        device = inputs['input_ids'].device
        bsz, sql = inputs['input_ids'].shape
        class_poss = []
        for idx in label_id_dict.values():
            # 类别编号
            class_idx = idx
            # 将 prefix 和 class_idx 编码到一起形成新的复合 class_idx，表示标签词
            for offset, prefix_idx in enumerate(reversed(self.prefix_idxs)):
                class_idx += prefix_idx * 100000 ** (offset + 1)
            input_ids = inputs['input_ids'].detach().clone()
            
            '''
            对于句子 "I love natural language processing"，如果我们将其划分为二元（2-gram）片段
            会得到以下的片段序列：
                "I love"
                "love natural"
                "natural language"
                "language processing"
            下面相当于将 input_ids 变成类似于 n-gram 的复合形式，从而在输入序列中匹配复合 id
            '''
            input_ids[:, 1:] += inputs['input_ids'][:, :-1] * 100000
            input_ids[:, 2:] += inputs['input_ids'][:, :-2] * 100000 * 100000
            # 通过比较输入序列和复合 id，得到每个标签词在输入序列中的位置
            # (1, sql) -> (bsz, sql)  
            # 然后使用 input_ids == class_idx 在 input_ids 的每个bsz上进行元素选择，得到其在sql上的下标
            class_pos = torch.arange(sql, device=device).unsqueeze(0).repeat(bsz, 1)[
                input_ids == class_idx].squeeze()
            class_poss.append(class_pos)
        # 返回每个标签词的位置 和 每个batch上inputs的最后一个位置
        # (cnt, bsz), (bsz)
        return class_poss, final_pos

    def _cal_all_key_and_values_of_class(self, inputs, past_key_values, one_class_one_list=False,
                                         include_final=False):
        class_poss, final_pos = self.get_pos(inputs)
        # 标签词是否包含最后一个位置
        if include_final:
            class_poss.append(final_pos)

        # 取出 和 class_poss 对应的 ker_or_value，并将其作为class_vecs返回
        def get_vecs(ker_or_value, class_poss):
            batch_idx = torch.arange(inputs['input_ids'].shape[0])
            class_vecs = []
            for poss in class_poss:
                class_vec = ker_or_value[batch_idx, :, poss, :]
                class_vecs.append(class_vec.unsqueeze(-2))
            if not one_class_one_list:
                class_vecs = torch.cat(class_vecs, dim=-2)
            return class_vecs

        key_and_values = []
        # 从每一层的 past_key_values 取出对应 class_poss 的 past_key_values
        for layer in range(0, self.layer):
            key_and_values.append(tuple([get_vecs(_, class_poss) for _ in past_key_values[layer]]))
        return key_and_values  # tuple of tuple of tensor (bsz, n_head, num_class, d_head)

    def cal_all_key_and_values_of_class(self, inputs, results, one_class_one_list=False,
                                        include_final=False):
        past_key_values = results.past_key_values
        key_and_values = self._cal_all_key_and_values_of_class(inputs, past_key_values,
                                                               one_class_one_list=one_class_one_list,
                                                               include_final=include_final)
        return key_and_values  # tuple of tuple of tensor (bsz, n_head, num_class, d_head)

    def get_attention(self, inputs, results, layer):
        class_poss, final_pos = self.get_pos(inputs)
        batch_idx = torch.arange(inputs['input_ids'].shape[0])
        scores = []
        for class_pos in class_poss:
            attention = results.attentions[layer][batch_idx, :, final_pos, class_pos]
            score = attention
            if class_pos.numel() == 1:
                score = score.sum(-1)
            else:
                score = score.sum()
            if inputs['input_ids'].shape[0] != 1:
                warnings.warn(f'Only support batch_size=1 now!')
            scores.append(score.unsqueeze(0))
        scores = torch.cat(scores, dim=0)
        return scores

    def cal_all_sim_attn(self, inputs, results):
        sims = []
        for layer in range(0, self.layer):
            sim = self.get_attention(inputs=inputs, results=results, layer=layer)
            sims.append(sim.unsqueeze(1))
        sims = torch.cat(sims, dim=1)
        sims = sims.reshape(inputs['input_ids'].shape[0], -1)
        return sims
