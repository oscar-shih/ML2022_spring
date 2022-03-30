import json
import csv
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from util import set_seed, get_cosine_schedule_with_warmup, collate_batch
from dataset import myDataset, InferenceDataset
from model import Classifier

def model_fn(batch, model, criterion, device):
	m, s = 0.3, 30
	mels, labels = batch
	mels = mels.to(device)
	labels = labels.to(device)

	outs, cosine = model(mels)
	m *= F.one_hot(labels, 600)
	cosine -= m
	cosine *= s
	loss = criterion(cosine, labels)

	# Get the speaker id with highest probability.
	preds = outs.argmax(1)
	accuracy = torch.mean((preds == labels).float())

	return loss, accuracy

def get_dataloader(data_dir, batch_size, n_workers):
	dataset = myDataset(data_dir)
	speaker_num = dataset.get_speaker_number()
	# Split dataset into training dataset and validation dataset
	trainlen = int(0.9 * len(dataset))
	lengths = [trainlen, len(dataset) - trainlen]
	trainset, validset = random_split(dataset, lengths)

	train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers, pin_memory=True, collate_fn=collate_batch)
	valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=n_workers, drop_last=True, pin_memory=True, collate_fn=collate_batch)

	return train_loader, valid_loader, speaker_num

def valid(dataloader, model, criterion, device): 
	model.eval()
	running_loss = 0.0
	running_accuracy = 0.0
	pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

	for i, batch in enumerate(dataloader):
		with torch.no_grad():
			loss, accuracy = model_fn(batch, model, criterion, device)
			running_loss += loss.item()
			running_accuracy += accuracy.item()

		pbar.update(dataloader.batch_size)
		pbar.set_postfix(loss=f"{running_loss / (i+1):.2f}", accuracy=f"{running_accuracy / (i+1):.2f}",)

	pbar.close()
	model.train()

	return running_accuracy / len(dataloader)

def train_parse_args():
	config = {
		"data_dir": "./Dataset",
		"save_path": {
			"model1": "./model1.ckpt",
			"model2": "./model2.ckpt",
			"model3": "./model3.ckpt",
			"model4": "./model4.ckpt",
			"model5": "./model5.ckpt",
		},
		"batch_size": 128,
		"n_workers": 0,
		"valid_steps": 2000,
		"warmup_steps": 1000,
		"save_steps": 10000,
		"total_steps": 100000,
		"classifier_hyperparameter":{
			"model1":{
				"input_dim": 100,
				"num_heads": 10,
				"ffn_dim": 256,
				"num_layers": 4,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
			"model2":{
				"input_dim": 100,
				"num_heads": 10,
				"ffn_dim": 256,
				"num_layers": 2,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
			"model3":{
				"input_dim": 100,
				"num_heads": 4,
				"ffn_dim": 256,
				"num_layers": 6,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
			"model4":{
				"input_dim": 100,
				"num_heads": 4,
				"ffn_dim": 786,
				"num_layers": 6,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
			"model5":{
				"input_dim": 100,
				"num_heads": 4,
				"ffn_dim": 512,
				"num_layers": 6,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
		}
	}

	return config

def train(data_dir, save_path, batch_size, n_workers, valid_steps, warmup_steps, total_steps, save_steps, classifier_hyperparameter):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
	train_iterator = iter(train_loader)
	print(f"[Info]: Finish loading data!",flush = True)
	for i in classifier_hyperparameter:
		model = Classifier(classifier_hyperparameter[i], n_spks=speaker_num).to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=2e-5, betas=(0.5, 0.999), amsgrad=True)
		scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
		print(f"[Info]: Finish creating model!",flush = True)

		best_accuracy = -1.0
		best_state_dict = None

		pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

		for step in range(total_steps):
			# Get data
			try:
				batch = next(train_iterator)
			except StopIteration:
				train_iterator = iter(train_loader)
				batch = next(train_iterator)

			loss, accuracy = model_fn(batch, model, criterion, device)
			batch_loss = loss.item()
			batch_accuracy = accuracy.item()

			# Updata model
			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

			# Log
			pbar.update()
			pbar.set_postfix(
				loss=f"{batch_loss:.2f}",
				accuracy=f"{batch_accuracy:.2f}",
				step=step + 1,
			)

			# Do validation
			if (step + 1) % valid_steps == 0:
				pbar.close()

				valid_accuracy = valid(valid_loader, model, criterion, device)

				# keep the best model
				if valid_accuracy > best_accuracy:
					best_accuracy = valid_accuracy
					best_state_dict = model.state_dict()

				pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

			# Save the best model so far.
			if (step + 1) % save_steps == 0 and best_state_dict is not None:
				torch.save(best_state_dict, save_path[i])
				pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

		pbar.close()

def inference_collate_batch(batch):
	feat_paths, mels = zip(*batch)
	return feat_paths, torch.stack(mels)

def inference_parse_args():
	config = {
		"data_dir": "./Dataset",
		"model_path": {
			"model1": "./model1.ckpt",
			"model2": "./model2.ckpt",
			"model3": "./model3.ckpt",
			"model4": "./model4.ckpt",
			"model5": "./model5.ckpt",
		},
		"output_path": "./output.csv",
		"classifier_hyperparameter":{
			"model1":{
				"input_dim": 100,
				"num_heads": 8,
				"ffn_dim": 256,
				"num_layers": 4,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
			"model2":{
				"input_dim": 100,
				"num_heads": 8,
				"ffn_dim": 256,
				"num_layers": 2,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
			"model3":{
				"input_dim": 100,
				"num_heads": 4,
				"ffn_dim": 256,
				"num_layers": 6,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
			"model4":{
				"input_dim": 100,
				"num_heads": 4,
				"ffn_dim": 786,
				"num_layers": 6,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
			"model5":{
				"input_dim": 100,
				"num_heads": 4,
				"ffn_dim": 512,
				"num_layers": 6,
				"depthwise_conv_kernel_size": 31,
				"dropout": 0.5
			},
		}
	}

	return config

def inference(data_dir, model_path, output_path, classifier_hyperparameter):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	mapping_path = Path(data_dir) / "mapping.json"
	mapping = json.load(mapping_path.open())

	dataset = InferenceDataset(data_dir)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, collate_fn=inference_collate_batch)
	print(f"[Info]: Finish loading data!",flush = True)

	speaker_num = len(mapping["id2speaker"])
	model1 = Classifier(classifier_hyperparameter["model1"], n_spks=speaker_num).to(device)
	model1.load_state_dict(torch.load(model_path["model1"]))
	model1.eval()

	model2 = Classifier(classifier_hyperparameter["model2"], n_spks=speaker_num).to(device)
	model2.load_state_dict(torch.load(model_path["model2"]))
	model2.eval()

	model3 = Classifier(classifier_hyperparameter["model3"], n_spks=speaker_num).to(device)
	model3.load_state_dict(torch.load(model_path["model3"]))
	model3.eval()

	model4 = Classifier(classifier_hyperparameter["model4"], n_spks=speaker_num).to(device)
	model4.load_state_dict(torch.load(model_path["model4"]))
	model4.eval()

	model5 = Classifier(classifier_hyperparameter["model5"], n_spks=speaker_num).to(device)
	model5.load_state_dict(torch.load(model_path["model5"]))
	model5.eval()

	print(f"[Info]: Finish creating model!",flush = True)

	results = [["Id", "Category"]]
	for feat_paths, mels in tqdm(dataloader):
		with torch.no_grad():
			mels = mels.to(device)
			outs1, _ = model1(mels)
			outs2, _ = model2(mels)
			outs3, _ = model3(mels)
			outs4, _ = model4(mels)
			outs5, _ = model5(mels)
			outs = (outs1+outs2+outs3+outs4+outs5) / 5
			preds = outs.argmax(1).cpu().numpy()
			for feat_path, pred in zip(feat_paths, preds):
				results.append([feat_path, mapping["id2speaker"][str(pred)]])

	with open(output_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(results)

if __name__ == "__main__":
	set_seed(1126)
	train(**train_parse_args())
	inference(**inference_parse_args())