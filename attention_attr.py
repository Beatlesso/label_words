import pickle
import warnings
from dataclasses import dataclass, field
from typing import List
import os
import numpy as np
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
import torch
import torch.nn.functional as F

from icl.lm_apis.lm_api_base import LMForwardAPI
from icl.utils.data_wrapper import wrap_dataset, tokenize_dataset
from icl.utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from icl.utils.prepare_model_and_tokenizer import load_model_and_tokenizer, \
    get_label_id_dict_for_args
from icl.utils.random_utils import set_seed
from icl.utils.other import load_args, set_gpu, sample_two_set_with_shot_per_class
from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoModelForCausalLM, \
    AutoTokenizer
from icl.utils.load_local import convert_path_old, load_local_model_or_tokenizer, \
    get_model_layer_num
from icl.util_classes.arg_classes import AttrArgs
from icl.util_classes.predictor_classes import Predictor
from transformers import HfArgumentParser
from datasets import concatenate_datasets
from datasets.utils.logging import disable_progress_bar
import icl.analysis.attentioner_for_attribution
from icl.analysis.attentioner_for_attribution import AttentionAdapter, \
    GPT2AttentionerManager
from icl.utils.other import dict_to

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

hf_parser = HfArgumentParser((AttrArgs,))
args: AttrArgs = hf_parser.parse_args_into_dataclasses()[0]

set_gpu(args.gpu)
if args.sample_from == 'test':
    dataset = load_huggingface_dataset_train_and_test(args.task_name)
else:
    raise NotImplementedError(f"sample_from: {args.sample_from}")

model, tokenizer = load_model_and_tokenizer(args)
args.label_id_dict = get_label_id_dict_for_args(args, tokenizer)

# model = model.half()

model = LMForwardAPI(model=model, model_name=args.model_name, tokenizer=tokenizer,
                     device='cuda:0',
                     label_dict=args.label_dict)


num_layer = get_model_layer_num(model=model.model, model_name=args.model_name)
predictor = Predictor(label_id_dict=args.label_id_dict, pad_token_id=tokenizer.pad_token_id,
                      task_name=args.task_name, tokenizer=tokenizer, layer=num_layer)

# 准备分析数据集
def prepare_analysis_dataset(seed):
    if args.sample_from == 'test':
        if len(dataset['test']) < args.actual_sample_size:
            args.actual_sample_size = len(dataset['test'])
            warnings.warn(
                f"sample_size: {args.sample_size} is larger than test set size: {len(dataset['test'])},"
                f"actual_sample_size is {args.actual_sample_size}")
        # 获取测试集并且打乱顺序
        test_sample = dataset['test'].shuffle(seed=seed).select(range(args.actual_sample_size))
    else:
        raise NotImplementedError(f"sample_from: {args.sample_from}")
    # 禁用进度条
    disable_progress_bar()
    # 获取训练集当做示例
    demonstration = dataset['train']
    # 类别数量
    class_num = len(set(demonstration['label']))
    np_labels = np.array(demonstration['label'])
    # 根据每个类别的标签，获取对应的数据索引
    ids_for_demonstrations = [np.where(np_labels == class_id)[0] for class_id in range(class_num)]
    demonstrations_contexted = []
    np.random.seed(seed)

    for i in range(len(test_sample)):
        demonstration_part_ids = []
        for _ in ids_for_demonstrations:
            demonstration_part_ids.extend(np.random.choice(_, args.demonstration_shot))
        demonstration_part = demonstration.select(demonstration_part_ids)
        # 随机打乱顺序
        demonstration_part = demonstration_part.shuffle(seed=seed)
        # 将选定的训练集样本与当前测试样本合并，创建一个新的数据样本
        # 训练集样本作为prompt
        demonstration_part = wrap_dataset(test_sample.select([i]), demonstration_part,
                                          args.label_dict,
                                          args.task_name)
        demonstrations_contexted.append(demonstration_part)
    demonstrations_contexted = concatenate_datasets(demonstrations_contexted)
    # 过滤掉超过 tokenizer 的最大长度限制的样本
    demonstrations_contexted = demonstrations_contexted.filter(
        lambda x: len(tokenizer(x["sentence"])['input_ids']) <= tokenizer.max_len_single_sentence)
    # 使用 tokenizer 对数据集的 sentence 进行处理，将文本转换为模型可以理解的输入表示形式。
    demonstrations_contexted = tokenize_dataset(demonstrations_contexted, tokenizer=tokenizer)
    return demonstrations_contexted


demonstrations_contexted = prepare_analysis_dataset(args.seeds[0])

if args.model_name in ['gpt2-xl']:
    attentionermanger = GPT2AttentionerManager(model.model)
else:
    raise NotImplementedError(f"model_name: {args.model_name}")

# output_dir 模型预测和检查点的输出目录
training_args = TrainingArguments("./output_dir", remove_unused_columns=False,
                                  per_device_eval_batch_size=1,
                                  per_device_train_batch_size=1)
trainer = Trainer(model=model, args=training_args)
analysis_dataloader = trainer.get_eval_dataloader(demonstrations_contexted)


for p in model.parameters():
    p.requires_grad = False

# 获取比例
def get_proportion(saliency, class_poss, final_poss):
    saliency = saliency.detach().clone().cpu()
    # torch.hstack：按照水平（列）顺序堆叠张量 (cnt, bsz) -> (cnt * bsz)
    class_poss = torch.hstack(class_poss).detach().clone().cpu()
    final_poss = final_poss.detach().clone().cpu()
    # saliency 的维度必须是2D或3D且在3D情况下第一维的大小为1
    assert len(saliency.shape) == 2 or (len(saliency.shape) == 3 and saliency.shape[0] == 1)
    # 三维移除第一维
    if len(saliency.shape) == 3:
        saliency = saliency.squeeze(0)
    saliency = saliency.numpy()
    # 将显著性图的对角线元素置为0，消除自身对自身影响的计算
    np.fill_diagonal(saliency, 0)
    # 计算其它token对标签词信息流的重要性
    proportion1 = saliency[class_poss, :].sum()
    # 计算标签词对最终预测位置信息流的重要性
    proportion2 = saliency[final_poss, class_poss].sum()
    # 其它信息流的重要性
    proportion3 = saliency.sum() - proportion1 - proportion2

    N = int(final_poss)
    # 信息流总数
    sum3 = (N + 1) * N / 2 - sum(class_poss) - len(class_poss)
    # 计算每个重要性的流均值
    # 这里每个标签词都要对 p_k 都有 j < p_k 的信息流流向它，所以信息流的个数为 sum(class_poss)
    proportion1 = proportion1 / sum(class_poss)
    proportion2 = proportion2 / len(class_poss)
    proportion3 = proportion3 / sum3
    proportions = np.array([proportion1, proportion2, proportion3])
    return proportions


pros_list = []
# tqdm 是进度条
for idx, data in tqdm(enumerate(analysis_dataloader)):
    # 将 dict 的内容都放到字典上
    data = dict_to(data, model.device)
    print(data['input_ids'].shape)
    attentionermanger.zero_grad()
    # 同于将字典中的键值对一个个地传递给方法
    output = model(**data)
    label = data['labels']
    loss = F.cross_entropy(output['logits'], label)
    loss.backward()
    # 获取每个标签词在不同batch上的位置，每个batch上inputs的最后一个位置
    # (cnt, bsz), (bsz)
    class_poss, final_poss = predictor.get_pos({'input_ids': attentionermanger.input_ids})

    print("~" * 30)
    print(np.array(torch.hstack(class_poss).detach().clone().cpu()).shape)
    print(np.array(final_poss.detach().clone().cpu()).shape)
    print("~" * 30)
    break

    pros = []




    # 计算每个Transformer层的pro，放入pros
    for i in range(len(attentionermanger.attention_adapters)):
        # 取第i个attention_adapters的grad
        saliency = attentionermanger.grad(use_abs=True)[i]
        pro = get_proportion(saliency, class_poss, final_poss)
        pros.append(pro)
    pros = np.array(pros)
    pros = pros.T
    pros_list.append(pros)

pros_list = np.array(pros_list)

os.makedirs(os.path.dirname(args.save_file_name), exist_ok=True)
with open(args.save_file_name, 'wb') as f:
    pickle.dump(pros_list, f)
