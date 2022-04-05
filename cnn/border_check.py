import os

from tqdm import tqdm

from train import Model
from utils import read_image


model = Model.load('model_weights.pth', ['border'])

dataset_dir = '../all_border_dataset_2000_annotations_split'

test_dataset = f'{dataset_dir}/test/'
files = sorted(os.listdir(test_dataset), reverse=True)

# files which are annotated -> have borders
border_filenames = [f[:-4] for f in files if f[-4:] == '.xml']
positive_predictions = []
for i in tqdm(range(len(border_filenames))):
    image = read_image(f'{test_dataset}/{border_filenames[i]}.jpg')
    
    prediction = model.predict(image)
    positive_predictions.append(prediction)

# files which are not annotated -> don't have borders
negative_filenames = [f[:-4] for f in files if f[:-4] not in border_filenames]
negative_predictions = []
for i in tqdm(range(len(negative_filenames))):
    image = read_image(f'{test_dataset}/{negative_filenames[i]}.jpg')
    prediction = model.predict(image)
    negative_predictions.append(prediction)

# filter out low confidence predictions
thresh = 0.8
positive_scores = [any(score > thresh) for _, _, score in positive_predictions] # labels, boxes, scores = prediction
negative_scores = [any(score > thresh) for _, _, score in negative_predictions]

for i in range(len(positive_scores)):
    if not positive_scores[i]:
        print(border_filenames[i])

# print results
print(
    f"""TP: {sum(positive_scores)}, TN: {len(negative_scores) - sum(negative_scores)}, """
    f"""FP: {sum(negative_scores)}, FN: {len(positive_scores) - sum(positive_scores)}""")
print(f'TP %: {sum(positive_scores) / len(positive_scores) * 100}%, TN %: {(len(negative_scores) - sum(negative_scores)) / len(negative_scores) * 100}%')
