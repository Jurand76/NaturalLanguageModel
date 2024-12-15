from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

# Test modelu

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

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
        input_ids=inputs["input_ids"],
        max_length=128,
        num_beams=5,
        do_sample=False  # Wyłączenie losowego próbkowania
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