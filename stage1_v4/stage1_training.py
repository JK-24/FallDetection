import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# =====================================================
# CONFIG
# =====================================================
DATA_ROOT = r"F:\JK\FallVision_Extracted"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_LEN = 32
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4
NUM_FOLDS = 3

print("====================================")
print("Device:", DEVICE)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
print("====================================\n")

# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.ToTensor()
])

# =====================================================
# MODEL
# =====================================================
class Stage1_3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, (3,7,7), (1,2,2), (1,3,3))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1,3,3),(1,2,2),(0,1,1))

        self.conv2 = nn.Conv3d(64,128,3,padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2,2)

        self.conv3 = nn.Conv3d(128,256,3,padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256,256,3,padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256,2)

    def forward(self,x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

# =====================================================
# DATASET
# =====================================================
class FallVisionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        frame_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))

        if len(frame_paths) == 0:
            return self.__getitem__((idx + 1) % len(self.samples))

        if len(frame_paths) >= CLIP_LEN:
            frame_paths = frame_paths[:CLIP_LEN]
        else:
            frame_paths += [frame_paths[-1]] * (CLIP_LEN - len(frame_paths))

        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            img = transform(img)
            frames.append(img)

        clip = torch.stack(frames, dim=1)
        return clip, label

# =====================================================
# UTIL
# =====================================================
def get_structure():
    structure = {}
    for label_name in ["Fall", "No Fall"]:
        label_root = os.path.join(DATA_ROOT, label_name)
        for context in os.listdir(label_root):
            context_path = os.path.join(label_root, context)
            folders = sorted(os.listdir(context_path))[:NUM_FOLDS]
            if context not in structure:
                structure[context] = {}
            structure[context][label_name] = folders
    return structure

def filter_valid(samples):
    return [(p,l) for p,l in samples if len(glob(os.path.join(p,"*.jpg"))) > 0]

# =====================================================
# MAIN
# =====================================================
def main():

    structure = get_structure()

    for fold in range(NUM_FOLDS-1,0,-1):

        print(f"\n========== FOLD {fold+1} ==========")

        train_samples, test_samples = [], []

        for context in structure:
            for label_name, label in [("Fall",1),("No Fall",0)]:
                folders = structure[context][label_name]
                test_folder = folders[fold]
                train_folders = [f for i,f in enumerate(folders) if i != fold]

                for folder in train_folders:
                    folder_path = os.path.join(DATA_ROOT,label_name,context,folder)
                    for vd in sorted(os.listdir(folder_path)):
                        train_samples.append((os.path.join(folder_path,vd), label))

                folder_path = os.path.join(DATA_ROOT,label_name,context,test_folder)
                for vd in sorted(os.listdir(folder_path)):
                    test_samples.append((os.path.join(folder_path,vd), label))

        train_samples = filter_valid(train_samples)
        test_samples = filter_valid(test_samples)

        train_loader = DataLoader(FallVisionDataset(train_samples),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=torch.cuda.is_available())

        test_loader = DataLoader(FallVisionDataset(test_samples),
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=torch.cuda.is_available())

        model = Stage1_3DCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        checkpoint_path = f"checkpoint_fold{fold+1}.pth"
        csv_path = f"metrics_fold{fold+1}.csv"

        start_epoch = 0
        best_f1 = 0.0

        if os.path.exists(checkpoint_path):
            print("Checkpoint found. Resuming...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint['best_f1']
            print(f"Resuming from epoch {start_epoch}")

        else:
            with open(csv_path,'w',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch","train_loss","train_acc",
                    "test_loss","test_acc",
                    "precision","recall","f1",
                    "specificity","balanced_acc",
                    "roc_auc","TP","TN","FP","FN"
                ])

        for epoch in range(start_epoch, EPOCHS):

            print(f"\nEpoch {epoch+1}/{EPOCHS}")

            # ================= TRAIN =================
            model.train()
            train_loss_total = 0
            train_correct = 0

            for batch_idx,(clips,labels) in enumerate(train_loader):

                if batch_idx % 50 == 0:
                    print(f"  Training batch {batch_idx}/{len(train_loader)}")

                clips,labels = clips.to(DEVICE),labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(clips)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                train_loss_total += loss.item() * clips.size(0)
                train_correct += (outputs.argmax(1)==labels).sum().item()

            train_loss = train_loss_total / len(train_samples)
            train_acc = train_correct / len(train_samples)

            print("  Training complete.")

            # ================= TEST =================
            model.eval()
            test_loss_total = 0
            test_correct = 0
            all_labels, all_preds, all_probs = [],[],[]

            with torch.no_grad():
                for batch_idx,(clips,labels) in enumerate(test_loader):

                    if batch_idx % 50 == 0:
                        print(f"  Testing batch {batch_idx}/{len(test_loader)}")

                    clips,labels = clips.to(DEVICE),labels.to(DEVICE)
                    outputs = model(clips)
                    loss = criterion(outputs,labels)

                    test_loss_total += loss.item() * clips.size(0)
                    probs = torch.softmax(outputs, dim=1)[:,1]
                    preds = outputs.argmax(1)

                    test_correct += (preds==labels).sum().item()
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            test_loss = test_loss_total / len(test_samples)
            test_acc = test_correct / len(test_samples)

            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            specificity = tn/(tn+fp) if (tn+fp)!=0 else 0
            balanced_acc = (recall + specificity)/2
            roc_auc = roc_auc_score(all_labels, all_probs)

            print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f}")

            with open(csv_path,'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1,train_loss,train_acc,
                    test_loss,test_acc,
                    precision,recall,f1,
                    specificity,balanced_acc,
                    roc_auc,tp,tn,fp,fn
                ])

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1
            }, checkpoint_path)

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"best_model_fold{fold+1}.pth")
                print("  New best model saved.")

        print(f"Fold {fold+1} finished. Best F1: {best_f1:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
