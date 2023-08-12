# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import os
import random
import wandb
import torch
from torch import autograd
from torchvision import transforms
import tqdm
import clip
from torch import nn, optim
from PIL import Image
from pathlib import Path
from collections import defaultdict
import sys
from volta_src.config import BertConfig
from volta_src.embeddings import BertLayerNorm
from volta_src.encoders import GeLU
from extras import convert_sents_to_features, BertLayer
import argparse
from OFA.transformers.src.transformers.models.ofa.tokenization_ofa import OFATokenizer
from OFA.transformers.src.transformers.models.ofa.modeling_ofa import OFAModel


from image_decoder import ImageDecoder,mask_image
from text_decoder import TextDecoder,mask_token



random.seed(10)
torch.manual_seed(10)


trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
resolution = 256
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def find_best_matches(text_features, photo_features):
    similarities = (photo_features @ text_features.T).squeeze(1)
    best_photo_idx = (-similarities).argsort()
    similarities = -similarities
    similarities.sort()
    return best_photo_idx, similarities


def convert_models_to_fp32(model):
    for p in model.parameters():
        if p.grad is not None:
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()


class ContextualCLIP(torch.nn.Module):
    def __init__(self, bert_config, args):
        super(ContextualCLIP, self).__init__()
        #self.clip, self.preprocess = clip.load(vision_backbone, device=device, jit=False)
        self.preprocess = patch_resize_transform
        self.OFA_encoder = OFAModel.from_pretrained('OFA-large/checkpoints')
        self.OFA_encoder.decoder = None

        self.img_decoder = ImageDecoder() # TODO 图像解码器，参考MAE
        self.txt_decoder = TextDecoder() # TODO 文本解码器，参考BERT

        config = BertConfig.from_dict(bert_config)
        self.fusion = args.fusion
        hidden_size = 1024
        config.hidden_size =  hidden_size
        config.num_attention_heads = 8
        self.transformer = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers)])
        self.transformer.cuda()
        self.prediction_layer = nn.Linear(config.hidden_size, 1).cuda()
        self.batch_size = 1
        self.logit_scale = float(args.logit_scale)
        self.frozen_clip = args.frozen_clip
        self.add_input = args.add_input
        self.positional = args.positional
        self.rloss = nn.CrossEntropyLoss()
        if args.positional:
            self.positional_emb = torch.nn.Embedding(10,hidden_size).cuda()

    def forward(self, images, text, pos_mask,img_idx, output_attn=False):
        """
        images : 10,3,256,256
        text : 1, text_length+2
        pos_mask : 10,1
        """
        if self.frozen_clip:
            with torch.no_grad():
                input_ids_context = text[:1].repeat(10, 1)
                gen = self.OFA_encoder(input_ids_context, patch_images=images,
                                       output_attentions=True,output_hidden_states=True)
        else:
            input_ids_context = text[:1].repeat(10, 1)
            gen = self.OFA_encoder(input_ids_context, patch_images=images,
                                   output_attentions=True,output_hidden_states=True)
            
        all_hidden_states = gen.last_hidden_state

        attn_map = gen.attentions[-1] #各层注意力图，可供gradCAM
        #attn_map = attn_map.mean(dim=1)
        #attn_map = attn_map[img_idx,256:-1,:256].reshape(-1,16,16)

        image_tokens = all_hidden_states[:,:256] # 10*256*1024
        cls_tokens = all_hidden_states[:, 256]   # 10*1*1024
        text_tokens = all_hidden_states[:,256:]  # 10*text_len*1024

        x_ = torch.unsqueeze(cls_tokens,dim=0)
        if self.positional:
            embs = self.positional_emb(torch.arange(10).cuda())
            embs = embs * pos_mask
            x_pos = x_ + embs
        else:
            x_pos = x_
        attention_mask = torch.ones((self.batch_size,1,1,10)).cuda()
        x = self.transformer[0](x_pos, attention_mask)
        for layer_module in self.transformer[1:]:
            x = layer_module(x, attention_mask)
        if self.add_input:
            x = x + x_

        preds = self.prediction_layer(x.half())

        ground_truth = torch.tensor([img_idx]).long().cuda()
        retrieval_loss = self.rloss(preds, ground_truth.unsqueeze(dim=0))

        mask_img = False
        mask_txt = False
        use_gradcam = False

        if output_attn:
            attn_grad = torch.autograd.grad(outputs=retrieval_loss, inputs=attn_map,retain_graph=True)
            attn_grad,attn_map = attn_grad[0].mean(dim=1),attn_map.mean(dim=1)
            attn_grad,attn_map = attn_grad[img_idx,256:-1,:256].reshape(-1,256),attn_map[img_idx,256:-1,:256].reshape(-1,256)
            grad_cam = attn_map*attn_grad

        img_recon_loss = 0
        txt_recon_loss = 0
        # x为十对图文的[cls]，b*10*1024(c_h)
        if mask_img:
            mask = mask_image(image_tokens,text_tokens)
            img_recon_loss = self.img_decoder(image_tokens,x,mask)
        if mask_txt:
            mask = mask_token(image_tokens,text_tokens)
            txt_recon_loss = self.txt_decoder(text_tokens,x,mask)
        
        loss = img_recon_loss + txt_recon_loss + retrieval_loss

        if output_attn:
            return preds, loss, grad_cam,gen.attentions, gen.hidden_states 
        return preds, loss

    def encode_images(self, photos_batch):
        photos = [Image.open(photo_file) for photo_file in photos_batch]
        photos_preprocessed = torch.stack([self.preprocess(photo) for photo in photos]).to(device)

        with torch.no_grad():
            photos_features = self.clip.encode_image(photos_preprocessed)
            photos_features /= photos_features.norm(dim=-1, keepdim=True)
        return photos_features.cpu().numpy()

    def encode_text(self, search_query):
        with torch.no_grad():
            text_encoded = self.clip.encode_text(clip.tokenize(search_query, truncate=True).to(device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        return text_encoded.cpu().numpy()
if __name__ == "__main__":
    wandb.init(project='contextualofa', settings=wandb.Settings(start_method="fork"))
    config = wandb.config
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsize", type=int, default=36)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", default= 0.2, type=float)
    parser.add_argument("-m", "--model", type=str, default='ViT-B/16')
    parser.add_argument("--fusion", type=str,default='mult')
    parser.add_argument("-a", "--activation", default='relu')
    parser.add_argument("-s", "--logit_scale", default=1)
    parser.add_argument("--frozen_clip", action="store_true",default=False)
    parser.add_argument("--finetuned_checkpoint_path", default='')
    parser.add_argument("--add_input", action="store_true",default=True)
    parser.add_argument("--positional", action="store_true",default=True)
    parser.add_argument("--head_scheduler", default= 1.0, type=float)
    parser.add_argument("--base_scheduler", default= 1.0, type=float)
    parser.add_argument("--transformer_layers", default=2, type=int)
    parser.add_argument("--all_pos", action="store_true",default=False)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--valid_descr_path', type=str, default='data/valid_data.json')
    parser.add_argument('--train_descr_path', type=str, default='data/train_data.json')
    parser.add_argument('--imgs_path', type=str, default='data/image-sets')

    args = parser.parse_args()
    assert args.fusion in ['concat', 'mult']
    assert args.activation in ['leaky-relu', 'relu', 'gelu']
    wandb.config.update(args)

    OFA_tokenizer = OFATokenizer.from_pretrained('OFA-large/')

    img_dirs = args.imgs_path
    valid_data = json.load(open(args.valid_descr_path, 'r'))
    train_data = json.load(open(args.train_descr_path, 'r'))
    train = []
    for img_dir, data in train_data.items():
        for img_idx, text in data.items():
            train.append((img_dir, img_idx, text))
    valid = []
    for img_dir, data in valid_data.items():
        for img_idx, text in data.items():
            valid.append((img_dir, img_idx, text))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'DEVICE USED: {device}')


    bert_config = json.load(open('imagecode/baselines/clip/vilbert-and-bert-config.json', 'r'))
    contextual_clip = ContextualCLIP(bert_config, args)
    if args.finetuned_checkpoint_path:
        checkpoint = torch.load(args.finetuned_checkpoint_path)
        contextual_clip.load_state_dict(checkpoint['model_state_dict'])
    contextual_clip.cuda()
    config = wandb.config
    wandb.watch(contextual_clip)

    if device == "cpu":
        contextual_clip.float()
    else:
        clip.model.convert_weights(
            contextual_clip)  # Actually this line is unnecessary since clip by default already on float16
        contextual_clip.OFA_encoder.float()

    MAX_EPOCHS = 30

    head_params = list(contextual_clip.transformer.parameters()) + list(contextual_clip.prediction_layer.parameters())
    if args.positional:
        head_params += list(contextual_clip.positional_emb.parameters())
    optimizer = optim.Adam([{"params": contextual_clip.OFA_encoder.parameters()}, {"params": head_params, "lr": config.lr_head}] , lr=config.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.wd)

    lambda1 = lambda epoch: args.base_scheduler ** epoch
    lambda2 = lambda epoch: args.head_scheduler ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    best_val = 0

    for i in range(args.epochs):
        save_model = False
        # EVALUATE
        contextual_clip.eval()
        if i != 0:
            correct = 0
            ranks = defaultdict(int)
            for img_dir, img_idx, text in tqdm.tqdm(valid):
            #    text = [text]
                img_idx = int(img_idx)
                img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
                img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
                images = [Image.open(photo_file) for photo_file in img_files]
                images = torch.stack([contextual_clip.preprocess(photo) for photo in images]).to(device)
                input_ids = OFA_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180)['input_ids'].to(device)

                if "open-images" in str(img_dir):
                    pos_mask = torch.zeros((10,1)).cuda()
                else:
                    pos_mask = torch.ones((10,1)).cuda()
                with torch.no_grad():
                    logits,_ = contextual_clip(images, input_ids, pos_mask, img_idx)
                logits = logits.squeeze()
                pred = torch.argmax(logits).squeeze()
                if img_idx == pred:
                    correct += 1
            print(correct)
            print(len(valid))
            print(ranks)
            acc = correct / len(valid)
            wandb.log({'val_acc': acc})
            if acc > best_val:
                best_val = acc
                save_model = True
                string = ''
                for key, val in list(vars(args).items()):
                    if 'path' not in key:
                        string += f'_{val}'
                torch.save({
                    'epoch': i,
                    'model_state_dict': contextual_clip.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"checkpoints/ofa/CONTEXTUAL_clip_best_{best_val}_{string.replace('/', '')}.pt")
            print('------------------------------')

        print(f'EPOCH: {i}')
        step = 0
        random.shuffle(train)
        contextual_clip.train()
        correct = 0
        for img_dir, img_idx, text in train:
            step += 1
            #text = [text]
            img_idx = int(img_idx)
            img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            images = [Image.open(photo_file) for photo_file in img_files]
            images = torch.stack([contextual_clip.preprocess(photo) for photo in images]).to(device)
            input_ids = OFA_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=180)['input_ids'].to(device)
            if "open-images" in str(img_dir):
                pos_mask = torch.zeros((10,1)).cuda()
            else:
                pos_mask = torch.ones((10,1)).cuda()
            if args.all_pos:
                pos_mask = torch.ones((10,1)).cuda()
            logits,loss = contextual_clip(images, input_ids, pos_mask, img_idx)
            #ground_truth = torch.tensor([img_idx]).long().to(device)  # the index of the correct one
            #loss = loss_txt(logits, ground_truth.unsqueeze(dim=0))
            loss.backward()
            pred = torch.argmax(logits).squeeze()
            if img_idx == pred:
                correct += 1

            if step % config.batchsize == 0:
                print(f'TOTAL LOSS: {loss}')
                print('STEP: ' + str(step))
                wandb.log({'loss': loss})
                if device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(contextual_clip)
                    optimizer.step()
                    clip.model.convert_weights(contextual_clip)
                    contextual_clip.OFA_encoder.float()
                optimizer.zero_grad()
                contextual_clip.zero_grad()
        acc = correct / len(train)
        wandb.log({'train_acc': acc})
        scheduler.step()
