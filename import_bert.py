from transformers import BertForSequenceClassification, AutoTokenizer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

fine_tuner = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
fine_tuner.train()

eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
results = evaluate(model, tokenizer, eval_dataset, args)
