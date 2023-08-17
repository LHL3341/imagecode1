import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import patches
from extras import convert_sents_to_features, BertLayer

def mask_image(image_patches, num_patches=256, mask_rate=0.5):  # TODO 参考MAE
    """
    image_patches: patch_num * hidden_size
    tokens: token_num * hidden_size
    mask: patch_num的0,1矩阵
    """
    device = 'cuda'
    # 根据 mask 比例计算需要 mask 掉的 patch 数量
    # num_patches = (h // self.patch_h) * (w // self.patch_w)
    num_masked = int(mask_rate * num_patches)
    image_patches = image_patches.unsqueeze(0)
    # Shuffle:生成对应 patch 的随机索引
    # torch.rand() 服从均匀分布(normal distribution)
    # torch.rand() 只是生成随机数，argsort() 是为了获得成索引
    # (b, n_patches)
    b=1
    shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
    # mask 和 unmasked patches 对应的索引
    mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

    # 对应 batch 维度的索引：(b,1)
    #batch_ind = torch.arange(b, device=device).unsqueeze(-1)
    # 利用先前生成的索引对 patches 进行采样，分为 mask 和 unmasked 两组
    #mask_patches, unmask_patches = image_patches[batch_ind, mask_ind],image_patches[batch_ind, unmask_ind]
    mask = torch.zeros(image_patches.shape[:-1]).cuda()
    mask[0,unmask_ind] = 1
    return mask


class ImageDecoder(torch.nn.Module):
    def __init__(
            self, config,layers = 8,
            mask_ratio=0.75, decoder_depth=1,encoder_dim=1024,
            num_decoder_heads=8, decoder_dim_per_head=64
    ):
        super(ImageDecoder, self).__init__()
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        # Encoder(这里 CW 用 ViT 实现)
        #self.encoder = encoder
        #self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w

        # 由于原生的 ViT 有 cls_token，因此其 position embedding 的倒数第2个维度是：
        # 实际划分的 patch 数量加上 1个 cls_token
        #num_patches_plus_cls_token, encoder_dim = encoder.pos_embed.shape[-2:]
        # Input channels of encoder patch embedding: patch size**2 x 3
        # 这个用作预测头部的输出通道，从而能够对 patch 中的所有像素值进行预测
        #num_pixels_per_patch = encoder.patch_embed.weight.size(1)

        decoder_dim = config.hidden_size
        # Encoder-Decoder：Encoder 输出的维度可能和 Decoder 要求的输入维度不一致，因此需要转换
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # Mask token
        # 社会提倡这个比例最好是 75%
        self.mask_ratio = mask_ratio
        # mask token 的实质：1个可学习的共享向量
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))

        # Decoder：实质就是多层堆叠的 Transformer
        self.transformer_blocks = nn.ModuleList([BertLayer(config) for _ in range(layers)])
        #self.decoder = Transformer(
        #    decoder_dim,
        #    decoder_dim * 4,
        #    depth=decoder_depth,
        #    num_heads=num_decoder_heads,
        #    dim_per_head=decoder_dim_per_head,
        #)
        # 在 Decoder 中用作对 mask tokens 的 position embedding
        # Filter out cls_token 注意第1个维度去掉 cls_token
        self.decoder_pos_embed = nn.Embedding(256, decoder_dim)

        # Prediction head 输出的维度数等于1个 patch 的像素值数量
        self.head = nn.Linear(decoder_dim, 256*3)

    def forward(self, image, cls_token, mask):
        image = image.unsqueeze(0)
        cls_token = cls_token.unsqueeze(0)
        # 对编码后的 tokens 维度进行转换，从而符合 Decoder 要求的输入维度
        mask_id = torch.where(mask==0)[1]
        unmask_id = torch.where(mask==1)[1]
        # 由于 mask token 实质上只有1个，因此要对其进行扩展，从而和 masked patches 一一对应
        # (decoder_dim)->(b, n_masked, decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(1, mask_id.shape[-1], 1)
        # 为 mask tokens 加入位置信息
        mask_tokens += self.decoder_pos_embed(mask_id)

        encoded_tokens = image[:,mask_id]
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)
        shuffle_indices = torch.cat([mask_id,unmask_id]).argsort()
        # 将 mask tokens 与 编码后的 tokens 拼接起来
        # (b, n_patches, decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        # Un-shuffle：恢复原先 patches 的次序
        dec_input_tokens = torch.empty_like(concat_tokens).cuda()
        dec_input_tokens[0, shuffle_indices] = concat_tokens
        # 将全量 tokens 喂给 Decoder 解码
        b, s,_ = dec_input_tokens.shape
        attention_mask = torch.ones((b,1,1,s)).cuda()
        x = self.transformer_blocks[0](dec_input_tokens, attention_mask)
        for layer_module in self.transformer_blocks[1:]:
            x = layer_module(x, attention_mask)
        x = x + dec_input_tokens
        #decoded_tokens = self.transformer_blocks(dec_input_tokens)
        # 取出解码后的 mask tokens
        dec_mask_tokens = x[0, mask_id, :]
        # 预测 masked patches 的像素值
        # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
        pred_mask_pixel_values = self.head(dec_mask_tokens.half())
        return pred_mask_pixel_values
        
def compute_img_loss(pred_mask_pixel_values, mask_patches):
    # loss 计算
    loss = F.mse_loss(pred_mask_pixel_values, mask_patches.half())
    return loss

def to_patches(image):
    c,h,w = image.shape
    num_patches = (h // 16) * (w // 16)
    # (b, c=3, h, w)->(b, n_patches, patch_size**2 * c)
    patches = image.view(
        c,
        h // 16, 16, 
        w // 16, 16
    ).permute(1, 3, 2, 4, 0).reshape(num_patches, -1)

    return patches