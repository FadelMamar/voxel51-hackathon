import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
import timm
import numpy as np, torch
from sentence_transformers import SentenceTransformer

from lightning import LightningModule, Trainer
from lightning import LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr


class FoodWasteTorchDataset(Dataset):
    def __init__(self, view, img_tfms,ing2idx,emb_matrix):
        self.view = view
        self.ids = view.values("id")
        self.img_tfms = img_tfms
        self.ing2idx = ing2idx
        self.emb_matrix = emb_matrix
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        s = self.view[self.ids[idx]]
        img = self.img_tfms(Image.open(s.filepath).convert("RGB"))

        # ingredient embeddings (variable-length)
        emb = torch.from_numpy(
            np.stack([self.emb_matrix[self.ing2idx[i]] for i in s["ingredient_name"]])
        )

        # regression target (g of waste per ingredient)
        tgt = torch.zeros(len(self.ing2idx), dtype=torch.float32)
        for ing, amt in zip(s["ingredient_name"], s["return_quantity"]):
            if amt is not None:
                tgt[self.ing2idx[ing]] = amt
        return img, emb, tgt

def collate(batch):
    imgs, embs, tgts = zip(*batch)
    return torch.stack(imgs), list(embs), torch.stack(tgts)


class DataModule(LightningDataModule):

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        self.sbert = SentenceTransformer(MODEL_NAME)
    
    def setup(self, stage=None):
        
        dataset = load_from_hub("Voxel51/food-waste-dataset", overwrite=True)

        all_ing = sorted({ing for s in dataset for ing in s["ingredient_name"]})
        ing2idx = {ing: i for i, ing in enumerate(all_ing)}
        idx2ing = {i: ing for ing, i in ing2idx.items()}
        emb_matrix = self.sbert.encode(all_ing, convert_to_numpy=True)
        NUM_ING = len(ing2idx)

        self.get_weights(dataset)

        val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAdjustSharpness(0.5),
        transforms.RandomAdjustContrast(0.5),
        transforms.RandomAdjustBrightness(0.2),
        transforms.RandomAdjustSaturation(0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_view = dataset.match({"split": "train"})
        test_view  = dataset.match({"split": "test"})
        self.train_dataset = FoodWasteTorchDataset(train_view, train_tfms,ing2idx,emb_matrix)
        self.test_dataset  = FoodWasteTorchDataset(test_view, val_tfms,ing2idx,emb_matrix)
    
    def get_weights(self,train_dataset,num_ing:int,device:str):
        freq = torch.zeros(num_ing, dtype=torch.long)

        for sample in train_dataset:
            freq += (sample[2] > 0)

        w = 1.0 / torch.log(freq.float() + 2)
        w = w / w.mean()
        w = w.to(device)
        return w

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,collate_fn=collate)

class FoodWastePredictor(nn.Module):
    """
    Baseline: ResNet image encoder + pooled SBERT ingredient encoder
    """
    def __init__(self, *, embedding_dim: int = 512, num_ingredients: int):
        super().__init__()

        # ---------- image branch ----------
        self.resnet = timm.create_model("timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k",pretrained=True,num_classes=128)

        # ---------- ingredient branch ----------
        self.ing_mlp = nn.Sequential(                       # (B, 512) → (B, 128)
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

        # ---------- fusion & head ----------
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.q_proj = nn.Linear(128, embedding_dim, bias=False)
        self.regressor   = nn.Linear(128, num_ingredients)  # (B,N_ing)

    def forward(self, images, artikel_emb, return_total=True):
        """
        images: Tensor [B, 3, H, W]
        artikel_emb: Tensor [B, L, 512] or list of [L_i, 512] tensors
        """
        # --- image path ---
        img_feat = self.resnet(images)                      # (B,128)

        # ----- text path -----
        if isinstance(artikel_emb, list):                   # list → pad
            artikel_emb = nn.utils.rnn.pad_sequence(
                artikel_emb, batch_first=True)
        # ing_feat = self.ing_mlp(artikel_emb.mean(dim=1))    # (B,128)
        token_feat = self.ing_mlp(artikel_emb)              # (B,L,128)
        query = self.q_proj(img_feat).unsqueeze(2)
        attn_logits = (artikel_emb @ query).squeeze(2)
        attn_weights = torch.softmax(attn_logits, dim=1)
        ing_feat = (attn_weights.unsqueeze(2) * token_feat).sum(dim=1)  # (B,128)
        # ----- fuse & slot predictions -----
        combined      = torch.cat((img_feat, ing_feat), dim=1)
        fused = self.fusion(combined)                       # (B, 128)
        return F.relu(self.regressor(fused))

class FoodWasteModel(LightningModule):

    def __init__(self,model:nn.Module,weight_vec:torch.Tensor,lr:float=1e-3):
        super().__init__()

        self.lr = lr

        self.model = model
        self.weight_vec = weight_vec
        self.criterion  = nn.MSELoss(reduction='none')

        self.y_true, self.y_pred = [], []

    def forward(self,*args):
        return self.model(*args)
    
    def jitter_targets(self,targets, epsilon=1.0):
        mask  = targets > 0
        noise = torch.empty_like(targets).uniform_(-epsilon, epsilon)
        jittered = targets + noise * mask.float()
        return torch.clamp(jittered, min=0.)

    def training_step(self,batch,batch_idx):

        images, ingredients, targets = batch
        images         = images.to(self.device)
        self.weight_vec = self.weight_vec.to(self.device)
        ingredients    = [seq.to(self.device) for seq in ingredients]
        targets  = targets.to(self.device)
        targets = self.jitter_targets(targets, epsilon=1.)
        outputs = self.forward(images, ingredients)
        tot_pred = outputs.sum(1)
        tot_true = targets.sum(1)
        diff_sq = self.criterion(outputs, targets)
        loss = (diff_sq * self.weight_vec).mean()
        total_loss = loss + 0.001 * F.l1_loss(tot_pred, tot_true)
        self.log("train_loss",loss)
        return loss

    def validation_step(self,batch,batch_idx):
        images, ingredients, targets = batch
        imgs     = images.to(self.device)
        ingredients     = [ing.to(self.device) for ing in ingredients]
        targets  = targets.to(self.device)
        preds = self.forward(imgs, ingredients)
        self.y_true.append(targets.cpu().numpy())
        self.y_pred.append(preds.cpu().numpy())

        return None
    
    def on_validation_epoch_end(self):
        self.y_true = np.concatenate(self.y_true, axis=0)   # (N, num_ing)
        self.y_pred = np.concatenate(self.y_pred, axis=0)
        true_total = self.y_true.sum(axis=1)
        pred_total = self.y_pred.sum(axis=1)

        mae_total  = np.abs(pred_total - true_total).mean()
        rho, _     = spearmanr(true_total, pred_total)
        mse  = mean_squared_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(self.y_true, self.y_pred)
        r2   = r2_score(self.y_true, self.y_pred)
        logs = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}
        logs["MAE_total"] = mae_total
        logs["rho"] = rho

        self.log_dict(logs)

        self.y_true, self.y_pred = [], []

        return None
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    data = DataModule()
    data.setup()
    for _ in data.train_dataloader():
        pass