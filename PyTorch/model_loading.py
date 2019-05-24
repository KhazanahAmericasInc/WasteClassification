from fastai.vision import *
from fastai.metrics import error_rate
from sklearn.metrics import confusion_matrix
############ model loading ################
model_load = cnn_learner(data,models.resnet34,metrics=error_rate)
model_load = model_load.load('waste_classification_2019-05-23 21:56:02')

preds_load = model_load.get_preds(ds_type=DatasetType.Test)

max_idxs = np.asarray(np.argmax(preds_load[0],axis=1))

yhat = []
for max_idx in max_idxs:
    yhat.append(data.classes[max_idx])

y = []

## convert POSIX paths to string first
for label_path in data.test_ds.items:
    y.append(str(label_path))
    
## then extract waste type from file path
pattern = re.compile("([a-z]+)[0-9]+")
for i in range(len(y)):
    y[i] = pattern.search(y[i]).group(1)

cm = confusion_matrix(y,yhat)
print(cm)

correct = 0

for r in range(len(cm)):
    for c in range(len(cm)):
        if (r==c):
            correct += cm[r,c]

accuracy = correct/sum(sum(cm))
accuracy