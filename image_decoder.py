import torch

def mask_image(image_patches, tokens, mask_rate=0.5):# TODO 参考MAE
    """
    image_patches: patch_num * hidden_size
    tokens: token_num * hidden_size
    mask: patch_num的0,1矩阵
    """
    mask = torch.zeros_like(image_patches).cuda()
    return mask

class ImageDecoder(torch.nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()
        pass

    def forward(self, image, text,mask):
        
        return loss