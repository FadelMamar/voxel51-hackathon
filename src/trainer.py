import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
import timm
import numpy as np, torch
from sentence_transformers import SentenceTransformer

from lightning import LightningModule, Trainer
from lightning import LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger,MLFlowLogger
import mlflow

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer 
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

        # regression target (g of waste per ingredient)
        tgt = torch.zeros(len(self.ing2idx), dtype=torch.float32)
        for ing, amt in zip(s["ingredient_name"], s["return_quantity"]):
            if amt is not None:
                tgt[self.ing2idx[ing]] = amt
        return img, s["ingredient_name"], tgt

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
        self.num_ingredients = len(ing2idx)
        self.ingredients = all_ing

        val_tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToImage(), 
        T.ToDtype(torch.float32, scale=True),
        #T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        train_tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToImage(), 
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(),
        T.RandomRotation(45),
        #T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1,),
        #T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        train_view = dataset.match({"split": "train"})
        test_view  = dataset.match({"split": "test"})
        self.train_dataset = FoodWasteTorchDataset(train_view, train_tfms,ing2idx,emb_matrix)
        self.test_dataset  = FoodWasteTorchDataset(test_view, val_tfms,ing2idx,emb_matrix)
    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,collate_fn=collate)

class FoodWastePredictor(nn.Module):
    """
    Baseline: ResNet image encoder + pooled SBERT ingredient encoder
    """
    def __init__(self, *, labels_list: int):
        super().__init__()

        # ---------- image branch ----------
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:timm/PE-Core-S-16-384')
        self.tokenizer = get_tokenizer('hf-hub:timm/PE-Core-S-16-384')
        self.text = self.tokenizer(labels_list, context_length=self.model.context_length)

        self.num_ingredients = len(labels_list)
        self.model.eval()  # set to eval mode
        for p in self.model.parameters():
            p.requires_grad = False
        #self.text_features = self.model.encode_text(self.text, normalize=True)

        self.preprocess = T.Compose([
            T.Resize((384, 384)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.regressor = nn.LazyLinear(self.num_ingredients)  # (B,N_ing)

    def forward(self, images:torch.Tensor, artikel_emb:list[list[str]], return_total=True):
        """
        images: Tensor [B, 3, H, W]
        artikel_emb: 
        """
        # --- image path ---
        image = self.preprocess(images)
        image_features = self.model.encode_image(image, normalize=True)
        text_features = self.model.encode_text(artikel_emb, normalize=True)
        
        return self.regressor(image_features @ text_features.T)

class FoodWasteModel(LightningModule):

    def __init__(self,model:nn.Module,lr:float=1e-3):
        super().__init__()

        self.lr = lr

        self.model = model
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
        targets  = targets.to(self.device)
        targets = self.jitter_targets(targets, epsilon=1.)
        outputs = self.forward(images, ingredients)
        tot_pred = outputs.sum(1)
        tot_true = targets.sum(1)
        diff_sq = self.criterion(outputs, targets)
        loss = diff_sq .mean()
        total_loss = loss + 0.001 * F.l1_loss(tot_pred, tot_true)
        self.log("train_loss",total_loss,on_epoch=True)
        return loss

    def validation_step(self,batch,batch_idx):
        images, ingredients, targets = batch
        imgs     = images.to(self.device)
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
        logs = {"val_MSE": mse, "val_RMSE": rmse, "val_MAE": mae, "val_R2": r2}
        logs["val_MAE_total"] = mae_total
        logs["val_rho"] = rho

        self.log_dict(logs)

        self.y_true, self.y_pred = [], []

        return None
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    data = DataModule()
    data.setup()
    #for _ in data.train_dataloader():
    #    pass
    model = FoodWasteModel(
        model=FoodWastePredictor(labels_list=data.ingredients),
        lr=1e-3
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

    logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="http://127.0.0.1:5000/")


    trainer = Trainer(max_epochs=10, accelerator="auto", 
                      precision=32,
                      #fast_dev_run=True,
                      callbacks=[
                        ModelCheckpoint(
                            dirpath="checkpoints",
                            monitor="val_MAE",
                            mode="min",
                            save_top_k=1,
                            save_on_train_epoch_end=False,
                        )
                    ],
                    logger=logger,
                      )

    trainer.fit(
        model,
        data,        
    )