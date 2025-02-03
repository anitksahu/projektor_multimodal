# projektor/default_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_models import BaseClassifier, BaseEmbedder

# --- Vision-Only: PreActResNet18 Implementation ---
class PreActBlock(nn.Module):
    """Pre-activation basic block."""
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False)
            )
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet18Classifier(BaseClassifier):
    """
    Default vision-only classifier using PreActResNet18.
    """
    def __init__(self, num_classes=10):
        super(PreActResNet18Classifier, self).__init__(num_classes)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(PreActBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(PreActBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(PreActBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(PreActBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * PreActBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class PreActResNet18Embedder(PreActResNet18Classifier, BaseEmbedder):
    """
    Vision-only embedder based on PreActResNet18.
    The final classification layer is removed to return features.
    """
    def __init__(self, num_classes=10):
        super(PreActResNet18Embedder, self).__init__(num_classes)
        self.linear = nn.Identity()

    def forward(self, x):
        return PreActResNet18Classifier.forward(self, x)

# --- Multimodal: CLIP-based Embedder Example ---
# (This implementation uses Hugging Face's CLIPProcessor to handle both image and text.)
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CLIPMultiModalEmbedder(BaseEmbedder):
    """
    A CLIP-based embedder that handles multi-modal input.
    It expects a dictionary with keys "image" and "text". The "text" input can be a string or a list of strings.
    
    This implementation uses the CLIPProcessor to:
       - Preprocess images (resize, normalize, etc.)
       - Tokenize and preprocess text
       
    It then feeds both inputs to the CLIP model and returns a concatenation of the image and text embeddings.
    """
    def __init__(self, device):
        super(CLIPMultiModalEmbedder, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()

    def forward(self, batch):
        """
        Expects batch to be a dictionary with keys:
           "image": a tensor or list of images,
           "text":  a string or list of strings.
        
        Returns:
           A tensor of concatenated image and text embeddings.
        """
        images = batch.get("image", None)
        texts = batch.get("text", None)
        if images is None and texts is None:
            raise ValueError("Batch must contain at least one of 'image' or 'text' keys.")

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        image_embeds = outputs.image_embeds if "image_embeds" in outputs else None
        text_embeds = outputs.text_embeds if "text_embeds" in outputs else None

        if image_embeds is not None:
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        if text_embeds is not None:
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        if (image_embeds is not None) and (text_embeds is not None):
            multimodal_embed = torch.cat([image_embeds, text_embeds], dim=1)
        elif image_embeds is not None:
            multimodal_embed = image_embeds
        else:
            multimodal_embed = text_embeds
        return multimodal_embed

