import torch
import torch.nn as nn
import torchvision
import timm
from utils import find_pair
import math
from prompt_encoder import PromptEncoder
from typing import Optional, Tuple
from copy import deepcopy


## ViT-Base + P3DE ##
class FeaturesDeitBase_P3DE(nn.Module):
    """
    backbone: ViT-Base
    """

    def __init__(self, num_cam):
        super(FeaturesDeitBase_P3DE, self).__init__()
        deitbase = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.features = deitbase
        self.num_cam = num_cam
        self.x_max = 20
        self.y_max = 20
        self.x_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.x_max, 1, deitbase.embed_dim))
        self.y_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.y_max, 1, deitbase.embed_dim))
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, deitbase.embed_dim))
        timm.models.layers.trunc_normal_(self.x_cam_token)
        timm.models.layers.trunc_normal_(self.y_cam_token)
        timm.models.layers.trunc_normal_(self.z_cam_token)
    
    def forward_features(self, x, pos, cam):
        device = x.device
        bs = len(x)
        cam_batch = torch.ones(bs, dtype=torch.int64, device=device) * cam[0]

        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        pos_x = ((pos[:,0] + pos[:,2]) * 0.5 * self.x_max).type(torch.int64)
        pos_y = ((pos[:,1] + pos[:,3]) * 0.5 * self.y_max).type(torch.int64)
        x_cam_token = torch.index_select(self.x_cam_token,0,cam_batch*self.x_max + pos_x)
        y_cam_token = torch.index_select(self.y_cam_token,0,cam_batch*self.y_max + pos_y)
        z_cam_token = torch.index_select(self.z_cam_token,0,cam_batch)
        x = torch.cat((x,x_cam_token,y_cam_token,z_cam_token),dim=1)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x = torch.cat((x[:,0],x[:,-3],x[:,-2],x[:,-1]),dim=-1)
        return x

    def forward(self, x, pos, cam):
        output = self.forward_features(x, pos, cam)
        output = output.view(output.size()[0], -1)
        return output



## ViT-Base + P3DE ##
class FeaturesDeitBase_P3DE_depth(nn.Module):
    """
    backbone: ViT-Base
    """

    def __init__(self, num_cam):
        super(FeaturesDeitBase_P3DE_depth, self).__init__()
        deitbase = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.features = deitbase
        self.num_cam = num_cam
        self.x_max = 20
        self.y_max = 20
        self.x_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.x_max, 1, deitbase.embed_dim))
        self.y_cam_token = nn.Parameter(torch.zeros(self.num_cam * self.y_max, 1, deitbase.embed_dim))
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, deitbase.embed_dim))
        timm.models.layers.trunc_normal_(self.x_cam_token)
        timm.models.layers.trunc_normal_(self.y_cam_token)
        timm.models.layers.trunc_normal_(self.z_cam_token)

        self.features.patch_embed = timm.models.layers.PatchEmbed(in_chans=4)
    
    def forward_features(self, x, cam, pos):
        cam_int = cam
        x = self.features.patch_embed(x)
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.features.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.features.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.features.pos_drop(x + self.features.pos_embed)
        pos_x = ((pos[:,0] + pos[:,2]) * 0.5 * self.x_max).type(torch.int64)
        pos_y = ((pos[:,1] + pos[:,3]) * 0.5 * self.y_max).type(torch.int64)
        x_cam_token = torch.index_select(self.x_cam_token,0,cam_int*self.x_max + pos_x)
        y_cam_token = torch.index_select(self.y_cam_token,0,cam_int*self.y_max + pos_y)
        z_cam_token = torch.index_select(self.z_cam_token,0,cam_int)
        x = torch.cat((x,x_cam_token,y_cam_token,z_cam_token),dim=1)
        x = self.features.blocks(x)
        x = self.features.norm(x)
        x = torch.cat((x[:,0],x[:,-3],x[:,-2],x[:,-1]),dim=-1)
        return x

    def forward(self, x, cam, pos):
        output = self.forward_features(x, cam, pos)
        output = output.view(output.size()[0], -1)
        return output


class Multi_View_Predictor_prompt(nn.Module):
    """
    prompt encoder from SAM
    """

    def __init__(self, num_cam, mode='box'):
        super(Multi_View_Predictor_prompt, self).__init__()
        prompt_embed_dim = 256
        self.image_size = 1024
        vit_patch_size = 16
        image_embedding_size = self.image_size // vit_patch_size
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        with open('weights/prompt_vit_h_4b8939.pth', "rb") as f:
            state_dict = torch.load(f, weights_only=True)
        self.prompt_encoder.load_state_dict(state_dict)
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        self.num_cam = num_cam
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, prompt_embed_dim))
        timm.models.layers.trunc_normal_(self.z_cam_token)

        self.pos_encoder = P3DE_net(prompt_embed_dim * 3, 'cuda')
        # self.pos_decoder = Decoder(prompt_embed_dim * 3, 'cuda')
        # num_heads = 4
        # self.multihead_attn = nn.MultiheadAttention(prompt_embed_dim, num_heads, batch_first=True)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=prompt_embed_dim, nhead=num_heads, batch_first=True)

        # self.to_3d = nn.Linear(deitbase.embed_dim, 3)
        # self.observers = nn.Linear(3, 2 * self.num_cam)
        self.mode = mode
        self.num_predictions = 1
        if self.mode == 'box':
            pred_dim = 4 * self.num_cam * self.num_predictions
        else:
            pred_dim = 2 * self.num_cam * self.num_predictions
        self.observers = nn.Linear(prompt_embed_dim * 3, pred_dim)

        self.sigmoid = nn.Sigmoid()
    
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    def apply_coords_torch(
        self, coords: torch.Tensor, 
        original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.image_size
        )
        # coords = deepcopy(coords).to(torch.float)
        coords = coords.clone().to(torch.float)
        coords[..., 0] = coords[..., 0] * new_w
        coords[..., 1] = coords[..., 1] * new_h
        return coords
    
    def predict_bbox(self, x):
        device = x.device
        bs = len(x)

        # preds = self.observers(self.to_3d(x[:,-1])).view(bs, self.num_cam, 2)
        if self.mode == 'box':
            preds = self.observers(x).view(bs, self.num_cam, self.num_predictions, 4)
            # preds = self.observers(self.pos_decoder(x)).view(bs, self.num_cam, 4)
            center_x = preds[:, :, :, 0]
            center_y = preds[:, :, :, 1]
            w = self.sigmoid(preds[:, :, :, 2])
            h = self.sigmoid(preds[:, :, :, 3])
            out_x1 = center_x - w * 0.5
            out_x2 = center_x + w * 0.5
            out_y1 = center_y - h * 0.5
            out_y2 = center_y + h * 0.5
            out_prompts = torch.stack([out_x1, out_y1, out_x2, out_y2], dim=-1) # shape: [bs, num_cam, num_predictions, 4]
        else:
            preds = self.observers(x).view(bs, self.num_cam, self.num_predictions, 2)
            x = preds[:, :, :, 0]
            y = preds[:, :, :, 1]
            out_prompts = torch.stack([x, y], dim=-1) # shape: [bs, num_cam, num_predictions, 2]

        return out_prompts

    def forward(
        self, 
        cam_batch, 
        original_size: Tuple[int, ...], 
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        reid: Optional[torch.Tensor] = None,
        ):
        if point_coords is not None:
            # point_coords.shape: [bs, N, 2]
            point_coords = self.apply_coords_torch(point_coords.unsqueeze(1), original_size)
            points = (point_coords, point_labels)
        else:
            points = None
        if boxes is not None:
            boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
            boxes = boxes.reshape(-1, 4) # shape: [bs, 4]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        # sparse_embeddings.shape: [num_bbox, 2, 256]
        # z_cam_token.shape: [num_bbox, 1, 256]
        z_cam_token = torch.index_select(self.z_cam_token, 0, cam_batch)

        x = torch.cat([sparse_embeddings, z_cam_token], dim=1)
        x = x.reshape(len(sparse_embeddings), -1)
        if reid != None:
            x = x + reid
        out = self.pos_encoder(x)
        # x = sparse_embeddings.reshape(len(sparse_embeddings), -1)
        # x = x + z_cam_token.squeeze(1)
        # out = self.pos_encoder(x)
        # if reid != None:
        #     out = out + reid

        out_prompts = self.predict_bbox(out)

        return out, out_prompts
    
    def encode_decode(self, cam_batch, original_size, prompts, reid=None):
        if prompts.shape[1] == 4:
            fea, preds = self.forward(cam_batch=cam_batch, original_size=original_size, boxes=prompts, reid=reid)
        else:
            prompts_labels = torch.ones((len(prompts), 1), dtype=torch.int, device=prompts.device)
            fea, preds = self.forward(cam_batch=cam_batch, original_size=original_size, point_coords=prompts, point_labels=prompts_labels)
        return fea, preds


class P3DE_net(nn.Module):
    def __init__(self, in_dim, device):
        super(P3DE_net, self).__init__()
        self.layer1 = nn.Linear(in_dim, in_dim)
        self.layer2 = nn.Linear(in_dim, in_dim)
        self.layer3 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.to(device)
    
    def layer_norm(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        eps = 1e-6
        return (x - mean) / (std + eps)

    def forward(self, x):
        x = self.layer1(x) + x
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.layer2(x) + x
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.layer3(x) + x
        x = self.layer_norm(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim, device):
        super(Decoder, self).__init__()
        self.layer1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.to(device)
    
    def layer_norm(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        eps = 1e-6
        return (x - mean) / (std + eps)

    def forward(self, x):
        x = self.layer1(x) + x
        x = self.layer_norm(x)
        x = self.relu(x)
        return x