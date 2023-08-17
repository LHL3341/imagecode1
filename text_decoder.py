import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from extras import convert_sents_to_features, BertLayer

def mask_token(inputs,mlm_probability=0.15):
    mask = torch.ones(inputs.shape[0])
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = mask.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    """
    prob_data = torch.full(labels.shape, args.mlm_probability) 
    	是生成一个labels一样大小的矩阵,里面的值默认是0.15.
    torch.bernoulli(prob_data),从伯努利分布中抽取二元随机数(0或者1),
    	prob_data是上面产生的是一个所有值为0.15(在0和1之间的数),
    	输出张量的第i个元素值,将以输入张量的第i个概率值等于1.
    	(在这里 即输出张量的每个元素有 0.15的概率为1, 0.85的概率为0. 15%原始数据 被mask住)
    """
    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).bool()
    """
    mask_indices通过bool()函数转成True,False
    下面对于85%原始数据 没有被mask的位置进行赋值为-1
    """
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    """
    对于mask的数据,其中80%是赋值为MASK.
    这里先对所有数据以0.8概率获取伯努利分布值, 
    然后 和maksed_indices 进行与操作,得到Mask 的80%的概率 indice, 对这些位置赋值为MASK 
    """
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    mask[indices_replaced] = 0

    # 10% of the time, we replace masked input tokens with random word
    """
    对于mask_indices剩下的20% 在进行提取,取其中一半进行random 赋值,剩下一般保留原来值. 
    """
    #indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    #inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    """
    最后返回 mask之后的input 和 label.
    inputs 为原文+Mask+radom 单词
    labels 为 1 和 -1. 其中1是Mask的位置, -1是没有mask的位置
    """
    return mask.cuda()

class TextDecoder(nn.Module):
    def __init__(self, config, layers=8,vocab_size=50265):
        super(TextDecoder, self).__init__()

        # embedding for BERT, sum of positional, segment, token embeddings
        #self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList([BertLayer(config) for _ in range(layers)])
        self.mask_token = nn.parameter.Parameter()
        self.fc = nn.Linear(1024,vocab_size)


    def forward(self,text,cls_token,mask):
        mask[0] = 1
        mask[-1] = 1
        restored_text = text * mask.unsqueeze(1)
        combined_input = torch.cat([cls_token.unsqueeze(0),restored_text]).unsqueeze(0)
        b, s,_ = combined_input.shape
        attention_mask = torch.ones((b,1,1,s)).cuda()
        x = self.transformer_blocks[0](combined_input, attention_mask)
        for layer_module in self.transformer_blocks[1:]:
            x = layer_module(x, attention_mask)
        x = x + combined_input
        x = x[:,1:,:].reshape(-1,1024)
        logits = self.fc(x.half())
        return logits
    
def compute_txt_loss(logits, target_tokens, mask):
    # Compute cross-entropy loss only for masked tokens
    mask = ~mask.bool()
    masked_logits = logits * mask.unsqueeze(-1)
    loss = nn.CrossEntropyLoss(reduction='mean')(masked_logits.view(-1, logits.shape[-1]), target_tokens[:,1:].view(-1))
    return loss