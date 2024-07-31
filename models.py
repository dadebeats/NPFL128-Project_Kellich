from torch import nn
from tensorflow.keras.layers import Layer
from transformers import TFBertModel, TFBertTokenizer
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from transformers import TFDistilBertModel, DistilBertTokenizer

model_name = 'distilbert-base-uncased'
class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.bert = TFDistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding


class BertLayer(Layer):
    def __init__(self, **kwargs):
        self.output_dim = 768
        self.bert = TFDistilBertModel.from_pretrained(model_name)
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

    def compute_output_shape(self, input_shape):
        return (input_shape[0][1], self.output_dim)
        #return (input_shape[0], self.output_dim)


def create_model2(numerical_input_dim, hidden_dim, output_dim):
    # BERT inputs
    input_ids = Input(shape=(512,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(512,), dtype='int32', name='attention_mask')

    # Numerical inputs
    numerical_input = Input(shape=(numerical_input_dim,), dtype='float32', name='numerical_input')

    # BERT embeddings
    bert_model = BertLayer()

    for layer in bert_model.bert.layers[:-1]:
        layer.trainable = False
    bert_output = bert_model([input_ids, attention_mask])

    # Numerical MLP
    numerical_mlp = Dense(hidden_dim, activation='relu')(numerical_input)
    numerical_mlp = Dropout(0.5)(numerical_mlp)
    numerical_mlp = Dense(hidden_dim, activation='relu')(numerical_mlp)

    # Concatenate BERT and numerical embeddings
    combined = Concatenate()([bert_output, numerical_mlp])

    # Final MLP layers
    x = Dense(hidden_dim, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(hidden_dim, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(output_dim, activation=None)(x)

    # Create the model
    model = Model(inputs=[input_ids, attention_mask, numerical_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    return model