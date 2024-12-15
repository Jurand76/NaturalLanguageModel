from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Test modelu

model = T5ForConditionalGeneration.from_pretrained("./results")  # Ścieżka do wytrenowanego modelu
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

def translate_nlp_to_sql(input_text):
    # Tokenizacja wejścia
    inputs = tokenizer(
        input_text,
        return_tensors="pt",  # Batch w formacie PyTorch
        max_length=128,
        truncation=True,
        padding=True
    )

    # Generowanie wyniku przez model
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        num_beams=4,  # Używamy beam search dla dokładniejszego wyniku
        early_stopping=True
    )

    # Dekodowanie wyniku
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

input_nlp = "Pokaż wszystkie samochody wyprodukowane po 2015 roku."

# Generowanie SQL
output_sql = translate_nlp_to_sql(input_nlp)

# Wynik
print(f"NLP: {input_nlp}")
print(f"SQL: {output_sql}")