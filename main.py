from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Inicjalizacja modelu i tokenizera
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Przetwarzanie danych
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

# Przygotowanie zbiorów danych
from datasets import Dataset

data = [
    {"input": "Pokaż wszystkie samochody wyprodukowane po 2015 roku.", "output": "SELECT * FROM cars WHERE year > 2015;"},
    {"input": "Wymień marki i modele samochodów z rokiem 2020.", "output": "SELECT make, model FROM cars WHERE year = 2020;"}
]

train_dataset = Dataset.from_list(data)

for example in train_dataset:
    print(example["input"], example["output"])

eval_data = [
    {"input": "Ile samochodów wyprodukowano w 2015 roku?", "output": "SELECT COUNT(*) FROM cars WHERE year = 2015;"},
    {"input": "Pokaż wszystkie samochody marki Toyota.", "output": "SELECT * FROM cars WHERE make = 'Toyota';"}
]

eval_dataset = Dataset.from_list(eval_data)

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Ustawienia trenera
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    remove_unused_columns=False
)

# Data collator dla przetwarzania sekwencji
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Inicjalizacja trenera
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Trening modelu
trainer.train()