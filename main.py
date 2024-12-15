from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from multiprocessing import freeze_support

import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Dane treningowe i walidacyjne
train_data = [
    {"input": "Pokaż wszystkie samochody wyprodukowane po 2015 roku.",
     "output": "SELECT * FROM cars WHERE year > 2015;"},
    {"input": "Pokaż wszystkie samochody wyprodukowane po 2016 roku.",
     "output": "SELECT * FROM cars WHERE year > 2016;"},
    {"input": "Pokaż wszystkie samochody wyprodukowane po 2018 roku.",
     "output": "SELECT * FROM cars WHERE year > 2018;"},
    {"input": "Pokaż samochody marki Ford.",
     "output": "SELECT * FROM cars WHERE make = 'Ford';"},
    {"input": "Pokaż samochody marki Opel.",
     "output": "SELECT * FROM cars WHERE make = 'Opel';"},
    {"input": "Pokaż samochody marki Audi.",
     "output": "SELECT * FROM cars WHERE make = 'Audi';"},
    {"input": "Pokaż ile samochodów wyprodukowano w 2020 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2020;"},
    {"input": "Pokaż ile samochodów wyprodukowano w 2010 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2010;"},
    {"input": "Pokaż ile samochodów wyprodukowano w 2015 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2015;"},
    {"input": "Wymień wszystkie modele samochodów marki Ford.",
     "output": "SELECT model FROM cars WHERE make = 'Ford';"},
    {"input": "Ile samochodów wyprodukowano w 2020 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2020;"},
    {"input": "Pokaż wszystkie samochody o cenie mniejszej niż 20 000.",
     "output": "SELECT * FROM cars WHERE price < 20000;"},
    {"input": "Wyświetl wszystkie marki samochodów, które są czerwone.",
     "output": "SELECT make FROM cars WHERE color = 'red';"},
    {"input": "Pokaż wszystkie samochody z rokiem produkcji pomiędzy 2010 a 2015.",
     "output": "SELECT * FROM cars WHERE year BETWEEN 2010 AND 2015;"}
]

eval_data = [
    {"input": "Pokaż ile samochodów wyprodukowano w 2015 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2015;"},
    {"input": "Pokaż wszystkie samochody marki Toyota.",
     "output": "SELECT * FROM cars WHERE make = 'Toyota';"},
    {"input": "Pokaż wszystkie samochody wyprodukowane po 2015 roku.",
     "output": "SELECT * FROM cars WHERE year > 2015;"},
    {"input": "Pokaż wszystkie samochody wyprodukowane po 2016 roku.",
     "output": "SELECT * FROM cars WHERE year > 2016;"},
    {"input": "Pokaż wszystkie samochody wyprodukowane po 2018 roku.",
     "output": "SELECT * FROM cars WHERE year > 2018;"},
    {"input": "Pokaż samochody marki Ford.",
     "output": "SELECT * FROM cars WHERE make = 'Ford';"},
    {"input": "Pokaż samochody marki Opel.",
     "output": "SELECT * FROM cars WHERE make = 'Opel';"},
    {"input": "Pokaż samochody marki Audi.",
     "output": "SELECT * FROM cars WHERE make = 'Audi';"},
    {"input": "Pokaż ile samochodów wyprodukowano w 2020 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2020;"},
    {"input": "Pokaż ile samochodów wyprodukowano w 2010 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2010;"},
    {"input": "Pokaż ile samochodów wyprodukowano w 2015 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2015;"},
    {"input": "Wymień wszystkie modele samochodów marki Ford.",
     "output": "SELECT model FROM cars WHERE make = 'Ford';"},
    {"input": "Ile samochodów wyprodukowano w 2020 roku?",
     "output": "SELECT COUNT(*) FROM cars WHERE year = 2020;"},
    {"input": "Pokaż wszystkie samochody o cenie mniejszej niż 20 000.",
     "output": "SELECT * FROM cars WHERE price < 20000;"},
    {"input": "Wyświetl wszystkie marki samochodów, które są czerwone.",
     "output": "SELECT make FROM cars WHERE color = 'red';"},
    {"input": "Pokaż wszystkie samochody z rokiem produkcji pomiędzy 2010 a 2015.",
     "output": "SELECT * FROM cars WHERE year BETWEEN 2010 AND 2015;"}
]


# Utwórz Dataset
print("Tworzenie datasetu")
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)
print()

# Tokenizer i model
print("Tworzenie tokenizera i wzorca modelu")
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")
print()

# Funkcja przetwarzania danych
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        padding=True,
        truncation=True
    )
    labels = tokenizer(
        examples["output"],
        max_length=512,
        padding=True,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Przetwórz dane

columns_to_remove = ["input", "output"]   # usuwanie kolumn input i output z datasetu

print("Przetwarzanie train dataset")
train_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=columns_to_remove)
print(train_dataset[0])
print()

print("Przetwarzanie eval dataset")
eval_dataset = eval_dataset.map(preprocess_function, batched=False, remove_columns=columns_to_remove)
print(eval_dataset[0])
print()



# Argumenty treningowe
training_args = TrainingArguments(
    dataloader_num_workers = 0,
    output_dir="./results",
    eval_strategy="epoch",  # Poprawiony parametr
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
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

# Zapisanie modelu

trainer.save_model("./results")



    