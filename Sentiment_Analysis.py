import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased')
model.load_state_dict(torch.load("Model/BERT_Model.pt", weights_only=True))
model.eval()

def predict(review):
    encoding = tokenizer.encode_plus(review,
                                     add_special_tokens=True,
                                     max_length=128,
                                     return_tensors='pt',
                                     padding='max_length',
                                     return_attention_mask=True,
                                     truncation=True
                                     )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, axis=1).item()

    return "Positive" if predictions == 1 else "Negative"


def main():
    print("┌─────────────────────────────────┐")
    print("│ Welcome to Sentiment Prediction │")
    print("└─────────────────────────────────┘")
    while True:
        option = input("""Enter "Yes" to Continue or "Exit" to Quit: """).lower()
        print()
        if option == "exit":
            print("Thank you for using Sentiment Prediction")
            break
        elif option == "yes":
            review = input("Enter the review for predicting the sentiment: ")
            print()
            sentiment = predict(review)
            print("───────────────────────────────────────────")
            print(f"The sentiment of the review is {sentiment}")
            print("───────────────────────────────────────────")
            print()

if __name__ == "__main__":
    main()
