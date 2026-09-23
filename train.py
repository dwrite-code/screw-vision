import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ==============================
# 1. 기본 설정
# ==============================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("사용 장치:", DEVICE)


# ==============================
# 2. 이미지 전처리
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# 3. 데이터셋
# ==============================

train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset/val",
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# ==============================
# 4. 클래스 확인
# ==============================

print("클래스:", train_dataset.classes)

NUM_CLASSES = len(train_dataset.classes)


# ==============================
# 5. ResNet-18 불러오기
# ==============================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# 마지막 분류층 수정
model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ==============================
# 6. 손실함수 / Optimizer
# ==============================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==============================
# 7. 학습
# ==============================

EPOCHS = 20

best_accuracy = 0.0


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    # ==========================
    # Validation
    # ==========================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Validation Accuracy: {accuracy * 100:.2f}%"
    )


    # ==========================
    # 가장 좋은 모델 저장
    # ==========================

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes
            },
            "best_model.pth"
        )

        print("⭐ 최고 모델 저장!")


print()
print("학습 완료!")
print("최고 Validation Accuracy:",
      best_accuracy * 100)