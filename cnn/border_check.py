from detecto import core, utils
import os
from tqdm import tqdm

model = core.Model.load('model_weights.pth', ['border'])
dataset_dir   = '../app/all_border_dataset_2000_annotations_split'

test_dataset = f'{dataset_dir}/test/'
files = sorted(os.listdir(test_dataset), reverse=True)

# files which are annotated -> have borders
border_filenames = [f[:-4] for f in files if f[-4:] == '.xml']
positive_predictions = []
for i in tqdm(range(len(border_filenames))):
    image = utils.read_image(f'{test_dataset}/{border_filenames[i]}.jpg') 
    prediction = model.predict(image)
    positive_predictions.append(prediction)

# files which are not annotated -> don't have borders
negative_filenames = [f[:-4] for f in files if f[:-4] not in border_filenames]
negative_predictions = []
for i in tqdm(range(len(negative_filenames))):
    image = utils.read_image(f'{test_dataset}/{negative_filenames[i]}.jpg') 
    prediction = model.predict(image)
    negative_predictions.append(prediction)

# filter out low confidence predictions
thresh = 0.7
positive_scores = [any(score > thresh) for _, _, score in positive_predictions]
negative_scores = [any(score > thresh) for _, _, score in negative_predictions]

# print results
print(f"""TP: {sum(positive_scores)}, TN: {len(negative_scores) - sum(negative_scores)}, """
      f"""FP: {sum(negative_scores)}, FN: {len(positive_scores) - sum(positive_scores)}""")
print(f'TP %: {sum(positive_scores) / len(positive_scores) * 100}%, TN %: {(len(negative_scores) - sum(negative_scores)) / len(negative_scores) * 100}%')