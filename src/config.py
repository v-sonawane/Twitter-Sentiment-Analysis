
import tokenizers
import transformers
import os

DEVICE="cuda"
BERT_PATH="/home/yesshecodes/Documents/BERT/"
model = transformers.BertModel.from_pretrained("bert-base-uncased")
MAX_LEN=512
TRAIN_BATCH_SIZE=15
VALID_BATCH_SIZE=8
EPOCHS=10
TRAINING_FILE="/media/yesshecodes/DATA/Tweet Sentiment Extraction/data/train.csv"
TOKENIZER=tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH,"vocab.txt"),
    lowercase=True
)

