from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from icl.utils.other import dict_to


class LMForwardAPI(nn.Module):
    def __init__(self, model, model_name, tokenizer, label_dict: Dict[int, str], device='cuda:0'):
        ''' 
            label_dict = {0: ' Negative', 1: ' Positive'} 
        '''
        super().__init__()
        self._use_past_key_values = False
        self._past_key_values = None
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.calibration_probs = None
        self.use_calibration_probs = False
        self.probs_from_results_fn = None
        self.results_args: dict = {}
        # label_map Dict[int, int]，tokenizer.encode 一步到位将字符串编码为 输入
        # 所以它是 文本编码后的词典id 到 label数字 的映射
        self.label_map = {tokenizer.encode(v, add_special_tokens=False)[0]: k for k, v in
                          label_dict.items()}
        self.position_offset = 0

        assert model_name in ['gpt2-xl', 'gpt-j-6b']

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'LMForwardAPI: set device to {device}')
        self.model = self.model.to(device)
        if self.past_key_values:
            self.past_key_values = self.past_key_values  # will reset device

    def cal_logits(self, inputs, **kwargs):
        '''根据inputs，获取最终预测token的logits'''
        self.model.eval()
        # 把字典值都放到device上
        inputs = dict_to(inputs, self.device)

        if self.use_past_key_values:
            past_key_values = self.get_past_key_values(inputs)
            kwargs['past_key_values'] = past_key_values
            inputs['attention_mask'] = self.get_mask_with_past_key_values(inputs['attention_mask'])
            # 获取这两张模型的位置编码方法
            if self.model_name in ['gpt-j-6b','gpt2-xl']:
                bsz, sql = inputs['input_ids'].shape
                position_ids = torch.arange(sql, dtype=torch.long, device=self.device).repeat(bsz, 1)
                position_ids = position_ids + self.position_offset
                kwargs['position_ids'] = position_ids

        # 获取 forward 的结果
        results = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            **kwargs,
        )
        # logits (bsz, sql, n_class)
        logits = results['logits']
        # find last position before pad tokens    input_ids(bsz, sql)
        input_ids = inputs['input_ids']
        # eos_token_id  结束符的id
        eos_token_id: int = self.tokenizer.eos_token_id
        is_not_eos = input_ids != eos_token_id
        prediction_pos = is_not_eos.sum(dim=1) - 1
        is_not_eos = is_not_eos.float()

        # check all eos_tokens are at the end
        assert (is_not_eos[:, :-1] - is_not_eos[:, 1:] >= 0).all()
        # get logits for the last position
        # 从 logits 中提取出每个序列中最后一个非结束符位置的预测结果，形成一个新的 logits (bsz, num_class) 张量
        logits = logits[torch.arange(input_ids.shape[0]), prediction_pos, :]
        # 返回 logits 和forward结果results
        return logits, results

    def _cal_probs(self, logits):
        # 取出对应用作最后分类输出的logits，并用softmax归一化
        interest_index = list(self.label_map.keys())
        logits = logits[:, interest_index]
        probs = F.softmax(logits, dim=-1)
        if self.use_calibration_probs:
            assert self.calibration_probs is not None
            probs = probs / self.calibration_probs
        # 分别返回归一化后和归一化前的结果
        return probs, logits

    def cal_probs(self, inputs, **kwargs):
        logits, results = self.cal_logits(inputs, **kwargs)
        probs, logits = self._cal_probs(logits)
        return probs, logits, results

    def cal_probs_from_results(self, inputs, results):
        return self.probs_from_results_fn(inputs, results)

    @property
    def past_key_values(self):
        return self._past_key_values

    @past_key_values.setter
    def past_key_values(self, past_key_values):
        ''' 根据传入的 past_key_values 设置 self._past_key_values'''
        if past_key_values is not None:
            assert isinstance(past_key_values, tuple)
            assert isinstance(past_key_values[0], tuple)
            assert len(past_key_values[0]) == 2
            assert isinstance(past_key_values[0][0], torch.Tensor)
            assert past_key_values[0][0].shape[0] == 1
            self._past_key_values = tuple(
                tuple(t.to(self.device) for t in tup) for tup in past_key_values)
        else:
            self._past_key_values = None

    @property
    def use_past_key_values(self):
        return self._use_past_key_values

    @use_past_key_values.setter
    def use_past_key_values(self, use_past_key_values):
        self._use_past_key_values = use_past_key_values

    
    def get_mask_with_past_key_values(self, mask):
        '''将原本mask，拼接上past_key_values的mask后返回'''
        if self.past_key_values is None:
            raise ValueError('past_key_values is None, please set it first')
        batch_size = mask.shape[0]
        past_key_values_len = self.past_key_values[0][0].shape[2]
        mask = torch.cat(
            [torch.ones(batch_size, past_key_values_len, dtype=torch.bool, device=self.device),
             mask], dim=1)
        return mask

    def get_past_key_values(self, inputs):
        '''从 self.past_key_values 取出对应的 past_key_values，并且扩张到对应 batch_size尺寸'''
        if self.past_key_values is None:
            raise ValueError('past_key_values is None, please set it first')
        batch_size = inputs['input_ids'].shape[0]
        past_key_values = ()
        for layer_key, layer_value in self.past_key_values:
            past_key_values += (
                                   layer_key.expand(batch_size, -1, -1, -1),
                                   layer_value.expand(batch_size, -1, -1, -1)),

        return past_key_values

    @torch.no_grad()
    def forward_no_grad(self, inputs):
        
        ori_logits, results = self.cal_logits(inputs, **self.results_args)
        probs, logits = self._cal_probs(ori_logits)
        probs_from_results = self.cal_probs_from_results(inputs, results)
        probs_from_results['ori_logits'] = ori_logits
        return probs, probs_from_results

    def forward(self, **kwargs):
        '''
            ori_logits有所有的num_class，而logits只有对应lable的数值
            根据inputs获取最终预测分类任务的logits和prob
        '''
        ori_logits, results = self.cal_logits(kwargs, **self.results_args)
        probs, logits = self._cal_probs(ori_logits)
        result = {'probs': probs, 'logits': logits, 'results': results}
        if self.probs_from_results_fn:
            probs_from_results = self.cal_probs_from_results(kwargs, results)
            result['probs_from_results'] = probs_from_results
        result['ori_logits'] = ori_logits
        return result
