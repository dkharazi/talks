from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
from flask import Flask, request, render_template


tokenizer = AutoTokenizer.from_pretrained("lvwerra/bert-imdb")
model = AutoModelForSequenceClassification.from_pretrained("lvwerra/bert-imdb")


app = Flask(__name__)


@app.route('/', methods=['GET'])
def form():
    return render_template('bertform.html')

@app.route('/', methods=['POST'])
def pred():
	review = request.form['text']
	test_tensor = tokenizer.encode(review, return_tensors="pt")
	outputs = model.forward(test_tensor)
	probs = tf.nn.softmax(outputs.logits[0].detach())
	prob_pos = probs[1].numpy()
	return render_template('bertform.html') + '<b>Review:</b> {} <br /> <b>Sentiment:</b> {}'.format(review, str(prob_pos))

if __name__ == '__main__':
    app.run()