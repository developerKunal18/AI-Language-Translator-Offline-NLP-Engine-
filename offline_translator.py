from transformers import MarianMTModel, MarianTokenizer

print("🌐 Offline AI Translator\n")
print("1. English → Hindi")
print("2. English → Marathi")

choice = input("Choose language: ")

if choice == "1":
    model_name = "Helsinki-NLP/opus-mt-en-hi"
elif choice == "2":
    model_name = "Helsinki-NLP/opus-mt-en-mr"
else:
    print("Invalid choice")
    exit()

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = input("\nEnter English text: ")

tokens = tokenizer([text], return_tensors="pt", padding=True)
translated = model.generate(**tokens)
result = tokenizer.decode(translated[0], skip_special_tokens=True)

print("\nTranslated Text:")
print(result)
