import torch
print(torch.cuda.is_available())

from detecto import core

dataset_dir   = '../app/all_border_dataset_2000_annotations_split'
train_dataset = core.Dataset(f'{dataset_dir}/train/')
val_dataset   = core.Dataset(f'{dataset_dir}/val/')

loader = core.DataLoader(train_dataset, batch_size=2, shuffle=True)
model  = core.Model(['border'])
losses = model.fit(loader, val_dataset, epochs=25, lr_step_size=5, learning_rate=0.001, verbose=True)

print(losses)

model.save('model_weights.pth')