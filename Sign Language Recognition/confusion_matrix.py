from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from CNNModel import CNNModel


def to_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor


# Load data
data = pd.read_excel("../Data/alphabet_data.xlsx", header=0)
data.pop("CHARACTER")
groupValue, coordinates = data.pop("GROUPVALUE"), data.copy()

coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
coordinates = torch.from_numpy(coordinates).float()
groupValue = torch.from_numpy(groupValue.to_numpy()).long()

k_folds = 4
epoch = 70
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# To accumulate predictions from all folds
all_true = []
all_pred = []

for fold, (trainIndex, valIndex) in enumerate(kf.split(coordinates)):
    print(f"Training Fold {fold + 1}/{k_folds}")

    training = to_cuda(coordinates[trainIndex])
    groupValueTraining = to_cuda(groupValue[trainIndex])

    validation = to_cuda(coordinates[valIndex])
    groupValueValidation = to_cuda(groupValue[valIndex])

    model = CNNModel()
    model = to_cuda(model)

    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = CrossEntropyLoss()

    # Train this fold
    for epochi in range(1, epoch + 1):
        model.train()
        optimizer.zero_grad()

        outputTrain = model(training)
        lossTrain = criterion(outputTrain, groupValueTraining)

        lossTrain.backward()
        optimizer.step()

    # After training, get predictions for this fold
    model.eval()
    with torch.no_grad():
        outputValid = model(validation).cpu()
        y_pred_fold = torch.argmax(outputValid, dim=1).numpy()
        y_true_fold = groupValueValidation.cpu().numpy()

        # Accumulate for final confusion matrix
        all_true.extend(y_true_fold.tolist())
        all_pred.extend(y_pred_fold.tolist())

# ---------- FINAL CONFUSION MATRIX ----------
cm_final = confusion_matrix(all_true, all_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_final)
disp.plot(xticks_rotation=90)
plt.title("Final Confusion Matrix (All Folds Combined)")
plt.show()
