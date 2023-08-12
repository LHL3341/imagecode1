import torch

def mask_token(image_patches, tokens):# TODO 参考BERT
    """
    image_patches: patch_num * hidden_size
    tokens: token_num * hidden_size
    mask: token_num的0,1矩阵
    """
    mask = torch.zeros_like(tokens).cuda()
    return mask

class TextDecoder(torch.nn.Module):
    def __init__(self):
        super(TextDecoder, self).__init__()
        pass

    def forward(self, image, text,mask):

        return loss
    