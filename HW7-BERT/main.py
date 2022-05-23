from genericpath import exists
from urllib import response
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import get_linear_schedule_with_warmup

from tqdm.auto import tqdm
from utils import same_seeds, read_data, evaluate
from data import QA_Dataset

if __name__ == '__main__':
    same_seeds(1126)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16_training = False
    model = BertForQuestionAnswering.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")

    train_questions, train_paragraphs = read_data("hw7_train.json")
    dev_questions, dev_paragraphs = read_data("hw7_dev.json")
    test_questions, test_paragraphs = read_data("hw7_test.json")

    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

    train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

    train_batch_size = 6

    # Note: Do NOT change batch size of dev_loader / test_loader !
    # Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    num_epoch = 3
    validation = True
    logging_step = 100
    learning_rate = 1e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epoch)

    model.train()
    # model.load_state_dict(torch.load("model.ckpt", map_location='cuda'))

    print("Start Training ...")

    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0
        
        for data in tqdm(train_loader):	
            # Load all data into GPU
            data = [i.to(device) for i in data]
            
            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss
            output.loss.backward()
            
            # Optimizer and scheduler step.
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            ##### TODO: Apply linear learning rate decay #####
            
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                        attention_mask=data[2].squeeze(dim=0).to(device))
                    # prediction is correct only if answer text exactly matches
                    dev_acc += evaluate(data, output, tokenizer, dev_paragraphs[dev_questions[i]['paragraph_id']], 
                        dev_paragraphs_tokenized[dev_questions[i]['paragraph_id']].tokens) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
                torch.save(model.state_dict(), f"./model.ckpt")

            model.train()

    print("Saving Model ...")
    model_save_dir = "saved_model_chinese_pretrain_mrc_roberta_wwm_ext_large" 
    model.save_pretrained(model_save_dir)

    print("Evaluating Test Set ...")
    result = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                        attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(evaluate(data, output, tokenizer, test_paragraphs[test_questions[i]['paragraph_id']],
                                test_paragraphs_tokenized[test_questions[i]['paragraph_id']].tokens))

    result_file = "result.csv"
    with open(result_file, 'w') as f:	
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")