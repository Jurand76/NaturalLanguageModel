from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Dane treningowe i walidacyjne
train_data = [
    {"input": ["Pokaż", "wszystkie", "samochody", "wyprodukowane", "po", "2015", "roku."],
     "output": ["SELECT", "*", "FROM", "cars", "WHERE", "year", ">", "2015;"]},
    {"input": ["Wymień", "marki", "i", "modele", "samochodów", "z", "rokiem", "2020."],
     "output": ["SELECT", "make,", "model", "FROM", "cars", "WHERE", "year", "=", "2020;"]}
]

eval_data = [
    {"input": ["Ile", "samochodów", "wyprodukowano", "w", "2015", "roku?"],
     "output": ["SELECT", "COUNT(*)", "FROM", "cars", "WHERE", "year", "=", "2015;"]},
    {"input": ["Pokaż", "wszystkie", "samochody", "marki", "Toyota."],
     "output": ["SELECT", "*", "FROM", "cars", "WHERE", "make", "=", "'Toyota';"]}
]

# Napraw dane wejściowe
def fix_input_output(data):
    for example in data:
        example["input"] = " ".join(example["input"]) if isinstance(example["input"], list) else example["input"]
        example["output"] = " ".join(example["output"]) if isinstance(example["output"], list) else example["output"]
    return data

train_data = fix_input_output(train_data)
eval_data = fix_input_output(eval_data)

print("Train data")
print(train_data)

# Utwórz Dataset
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# Tokenizer i model
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Funkcja przetwarzania danych
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        examples["output"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Przetwórz dane

columns_to_remove = ["input", "output"]   # usuwanie kolumn input i output z datasetu

print("Przetwarzanie train dataset")
train_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=columns_to_remove)
print(train_dataset[0])

print("Przetwarzanie eval dataset")
eval_dataset = eval_dataset.map(preprocess_function, batched=False, remove_columns=columns_to_remove)
print(eval_dataset[0])

# Argumenty treningowe
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Poprawiony parametr
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    remove_unused_columns=False
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Trener
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Trening modelu
print("Trening modelu")
trainer.train()
print("Trening zakończony")
