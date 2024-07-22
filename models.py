from torch import nn
import torch
from tensorflow.keras.layers import Layer
from transformers import BertModel, BertTokenizer

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate

class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding


class Model2(nn.Module):
    def __init__(self, bert_encoder, numerical_input_dim, hidden_dim, output_dim):
        super(Model2, self).__init__()
        self.bert_encoder = bert_encoder
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim + bert_encoder.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_ids, attention_mask, numerical_data):
        text_embedding = self.bert_encoder(input_ids, attention_mask)
        numerical_embedding = self.numerical_mlp(numerical_data)
        combined_embedding = torch.cat((text_embedding, numerical_embedding), dim=1)
        output = self.final_mlp(combined_embedding)
        return output


class BertLayer(Layer):
    def __init__(self, **kwargs):
        self.output_dim = 768
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
def create_model2(bert_model, numerical_input_dim, hidden_dim, output_dim):
    # BERT inputs
    input_ids = Input(shape=(512,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(512,), dtype='int32', name='attention_mask')

    # Numerical inputs
    numerical_input = Input(shape=(numerical_input_dim,), dtype='float32', name='numerical_input')

    # BERT embeddings
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