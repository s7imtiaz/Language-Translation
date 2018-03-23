
# Language Translation
In this project, you’re going to take a peek into the realm of neural network machine translation.  You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
## Get the Data
Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## Explore the Data
Play around with view_sentence_range to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028
    
    English sentences 0 to 10:
    new jersey is sometimes quiet during autumn , and it is snowy in april .
    the united states is usually chilly during july , and it is usually freezing in november .
    california is usually quiet during march , and it is usually hot in june .
    the united states is sometimes mild during june , and it is cold in september .
    your least liked fruit is the grape , but my least liked is the apple .
    his favorite fruit is the orange , but my favorite is the grape .
    paris is relaxing during december , but it is usually chilly in july .
    new jersey is busy during spring , and it is never hot in march .
    our least liked fruit is the lemon , but my least liked is the grape .
    the united states is sometimes busy during january , and it is sometimes warm in november .
    
    French sentences 0 to 10:
    new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
    california est généralement calme en mars , et il est généralement chaud en juin .
    les états-unis est parfois légère en juin , et il fait froid en septembre .
    votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
    son fruit préféré est l'orange , mais mon préféré est le raisin .
    paris est relaxant en décembre , mais il est généralement froid en juillet .
    new jersey est occupé au printemps , et il est jamais chaude en mars .
    notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
    les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .


## Implement Preprocessing Function
### Text to Word Ids
As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of `target_text`.  This will help the neural network predict when the sentence should end.

You can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


```python
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_id_text = []
    target_id_text = []
    for sentence in source_text.split("\n"):
        out = [source_vocab_to_int[w] for w in sentence.split()]
        source_id_text.append(out)
    for sentence in target_text.split("\n"):
        out = [target_vocab_to_int[w] for w in sentence.split()]
        out.append(target_vocab_to_int['<EOS>'])
        target_id_text.append(out)
    return source_id_text, target_id_text

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)
```

    Tests Passed


### Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper
import problem_unittests as tests

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0.0'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.1.0
    Default GPU Device: /gpu:0


## Build the Neural Network
You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoder_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.
- Target sequence length placeholder named "target_sequence_length" with rank 1
- Max target sequence length tensor named "max_target_len" getting its value from applying tf.reduce_max on the target_sequence_length placeholder. Rank 0.
- Source sequence length placeholder named "source_sequence_length" with rank 1

Return the placeholders in the following the tuple (input, targets, learning rate, keep probability, target sequence length, max target sequence length, source sequence length)


```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    # TODO: Implement Function
    i = tf.placeholder(tf.int32, shape = (None,None), name = 'input')
    t = tf.placeholder(tf.int32, shape = (None, None), name = 'target')
    lr = tf.placeholder(tf.float32, shape = None, name = "learning_rate")
    keep_prob = tf.placeholder(tf.float32, shape = None, name = "keep_prob")
    target_sequence_length = tf.placeholder(tf.int32, shape = [None,], name = 'target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length, name = 'max_target_len')
    source_sequence_length = tf.placeholder(tf.float32, shape = [None,], name = 'source_sequence_length')
    return i, t, lr, keep_prob , target_sequence_length , max_target_len, source_sequence_length


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### Process Decoder Input
Implement `process_decoder_input` by removing the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.


```python
def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    target_data_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1,1])
    target_data = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), target_data_slice], 1)
    return target_data

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_encoding_input(process_decoder_input)
```

    Tests Passed


### Encoding
Implement `encoding_layer()` to create a Encoder RNN layer:
 * Embed the encoder input using [`tf.contrib.layers.embed_sequence`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence)
 * Construct a [stacked](https://github.com/tensorflow/tensorflow/blob/6947f65a374ebf29e74bb71e36fd82760056d82c/tensorflow/docs_src/tutorials/recurrent.md#stacking-multiple-lstms) [`tf.contrib.rnn.LSTMCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell) wrapped in a [`tf.contrib.rnn.DropoutWrapper`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper)
 * Pass cell and embedded input to [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)


```python
from imp import reload
reload(tests)

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    # TODO: Implement Function
    def lstm_cell():
        lstm = tf.contrib.rnn.LSTMCell(rnn_size)
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return lstm
    
    embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)
    
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
    output, state = tf.nn.dynamic_rnn(cell, embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    
    return output, state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed


### Decoding - Training
Create a training decoding layer:
* Create a [`tf.contrib.seq2seq.TrainingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper) 
* Create a [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
* Obtain the decoder outputs from [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)


```python

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    # TODO: Implement Function
    dropout_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,output_keep_prob = keep_prob)

    training_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,target_sequence_length)
        
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dropout_cell, training_helper, encoder_state, output_layer)
        
        
    training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder, maximum_iterations = max_summary_length)[0]


    return training_decoder_output


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```

    Tests Passed


### Decoding - Inference
Create inference decoder:
* Create a [`tf.contrib.seq2seq.GreedyEmbeddingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper)
* Create a [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
* Obtain the decoder outputs from [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    # TODO: Implement Function
    dropout_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,output_keep_prob = keep_prob)
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, tf.fill([batch_size], 
                                                                                        start_of_sequence_id), 
                                                                end_of_sequence_id)
    
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dropout_cell, inference_helper, encoder_state,
                                                       output_layer = output_layer)
    
    basic_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder, maximum_iterations = max_target_sequence_length)[0]

    return basic_decoder_output

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```

    Tests Passed


### Build the Decoding Layer
Implement `decoding_layer()` to create a Decoder RNN layer.

* Embed the target sequences
* Construct the decoder LSTM cell (just like you constructed the encoder cell above)
* Create an output layer to map the outputs of the decoder to the elements of our vocabulary
* Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob)` function to get the training logits.
* Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob)` function to get the inference logits.

Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
from tensorflow.python.layers.core import Dense
def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # TODO: Implement Function
    def lstm_cell():
        lstm = tf.contrib.rnn.LSTMCell(rnn_size)
        lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
        return lstm
    
    with tf.variable_scope('decoding_scope') as decoding_scope:
        dec_embeddings = tf.get_variable('dec_embeddings', [target_vocab_size, decoding_embedding_size])
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
        
        stack_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        output = Dense(target_vocab_size, use_bias = False)
        
        train_logits = decoding_layer_train(encoder_state, stack_cell, dec_embed_input, target_sequence_length,
                                            max_target_sequence_length, output, keep_prob)
        decoding_scope.reuse_variables()
        
        layer_infer = decoding_layer_infer(encoder_state, stack_cell, tf.get_variable('dec_embeddings'),
                                          target_vocab_to_int['<GO>'],
                                          target_vocab_to_int['<EOS>'],
                                          max_target_sequence_length,
                                          target_vocab_size,
                                          output,
                                          batch_size,
                                          keep_prob)
        
    return train_logits, layer_infer



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:

- Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,  source_sequence_length, source_vocab_size, encoding_embedding_size)`.
- Process target data using your `process_decoder_input(target_data, target_vocab_to_int, batch_size)` function.
- Decode the encoded input using your `decoding_layer(dec_input, enc_state, target_sequence_length, max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, dec_embedding_size)` function.


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    # TODO: Implement Function
    encoded_input, enc_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob, source_sequence_length,
                                             source_vocab_size, enc_embedding_size)
    
    process_target_data = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    
    decoded_output, dec_infer = decoding_layer(process_target_data, enc_state, target_sequence_length,
                                              max_target_sentence_length, rnn_size, num_layers, 
                                              target_vocab_to_int, target_vocab_size, batch_size, 
                                              keep_prob, dec_embedding_size)
    
    
    return decoded_output, dec_infer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `num_layers` to the number of layers.
- Set `encoding_embedding_size` to the size of the embedding for the encoder.
- Set `decoding_embedding_size` to the size of the embedding for the decoder.
- Set `learning_rate` to the learning rate.
- Set `keep_probability` to the Dropout keep probability
- Set `display_step` to state how many steps between each debug output statement


```python
# Number of Epochs
epochs = 3
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 512
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = 256
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.7
display_step = 10
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

```

Batch and pad the source and target sequences


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

```

### Train
Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))                                                                                                  
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})


                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch   10/538 - Train Accuracy: 0.2672, Validation Accuracy: 0.3569, Loss: 3.6803
    Epoch   0 Batch   20/538 - Train Accuracy: 0.3460, Validation Accuracy: 0.3929, Loss: 3.0737
    Epoch   0 Batch   30/538 - Train Accuracy: 0.3520, Validation Accuracy: 0.4231, Loss: 2.8637
    Epoch   0 Batch   40/538 - Train Accuracy: 0.4251, Validation Accuracy: 0.4368, Loss: 2.4170
    Epoch   0 Batch   50/538 - Train Accuracy: 0.3943, Validation Accuracy: 0.4524, Loss: 2.4005
    Epoch   0 Batch   60/538 - Train Accuracy: 0.3937, Validation Accuracy: 0.4551, Loss: 2.3134
    Epoch   0 Batch   70/538 - Train Accuracy: 0.4023, Validation Accuracy: 0.4391, Loss: 2.0700
    Epoch   0 Batch   80/538 - Train Accuracy: 0.3986, Validation Accuracy: 0.4673, Loss: 2.0119
    Epoch   0 Batch   90/538 - Train Accuracy: 0.4420, Validation Accuracy: 0.4702, Loss: 1.7752
    Epoch   0 Batch  100/538 - Train Accuracy: 0.4309, Validation Accuracy: 0.4778, Loss: 1.6526
    Epoch   0 Batch  110/538 - Train Accuracy: 0.4412, Validation Accuracy: 0.4995, Loss: 1.6233
    Epoch   0 Batch  120/538 - Train Accuracy: 0.4443, Validation Accuracy: 0.4858, Loss: 1.4679
    Epoch   0 Batch  130/538 - Train Accuracy: 0.4842, Validation Accuracy: 0.5083, Loss: 1.3656
    Epoch   0 Batch  140/538 - Train Accuracy: 0.4209, Validation Accuracy: 0.4931, Loss: 1.3914
    Epoch   0 Batch  150/538 - Train Accuracy: 0.4916, Validation Accuracy: 0.5211, Loss: 1.2651
    Epoch   0 Batch  160/538 - Train Accuracy: 0.4879, Validation Accuracy: 0.5231, Loss: 1.1502
    Epoch   0 Batch  170/538 - Train Accuracy: 0.5182, Validation Accuracy: 0.5378, Loss: 1.1189
    Epoch   0 Batch  180/538 - Train Accuracy: 0.4896, Validation Accuracy: 0.5066, Loss: 1.0881
    Epoch   0 Batch  190/538 - Train Accuracy: 0.4892, Validation Accuracy: 0.5245, Loss: 1.0732
    Epoch   0 Batch  200/538 - Train Accuracy: 0.5025, Validation Accuracy: 0.5316, Loss: 1.0123
    Epoch   0 Batch  210/538 - Train Accuracy: 0.5136, Validation Accuracy: 0.5538, Loss: 0.9550
    Epoch   0 Batch  220/538 - Train Accuracy: 0.5076, Validation Accuracy: 0.5415, Loss: 0.9073
    Epoch   0 Batch  230/538 - Train Accuracy: 0.5322, Validation Accuracy: 0.5588, Loss: 0.9142
    Epoch   0 Batch  240/538 - Train Accuracy: 0.5225, Validation Accuracy: 0.5559, Loss: 0.8903
    Epoch   0 Batch  250/538 - Train Accuracy: 0.5613, Validation Accuracy: 0.5811, Loss: 0.8407
    Epoch   0 Batch  260/538 - Train Accuracy: 0.5565, Validation Accuracy: 0.5751, Loss: 0.8211
    Epoch   0 Batch  270/538 - Train Accuracy: 0.5490, Validation Accuracy: 0.5787, Loss: 0.7928
    Epoch   0 Batch  280/538 - Train Accuracy: 0.6066, Validation Accuracy: 0.5964, Loss: 0.7423
    Epoch   0 Batch  290/538 - Train Accuracy: 0.5650, Validation Accuracy: 0.5904, Loss: 0.7505
    Epoch   0 Batch  300/538 - Train Accuracy: 0.5934, Validation Accuracy: 0.5890, Loss: 0.7188
    Epoch   0 Batch  310/538 - Train Accuracy: 0.5920, Validation Accuracy: 0.6033, Loss: 0.7020
    Epoch   0 Batch  320/538 - Train Accuracy: 0.6021, Validation Accuracy: 0.6108, Loss: 0.6841
    Epoch   0 Batch  330/538 - Train Accuracy: 0.6122, Validation Accuracy: 0.6028, Loss: 0.6600
    Epoch   0 Batch  340/538 - Train Accuracy: 0.5795, Validation Accuracy: 0.5980, Loss: 0.7024
    Epoch   0 Batch  350/538 - Train Accuracy: 0.5943, Validation Accuracy: 0.5920, Loss: 0.6713
    Epoch   0 Batch  360/538 - Train Accuracy: 0.5941, Validation Accuracy: 0.6087, Loss: 0.6658
    Epoch   0 Batch  370/538 - Train Accuracy: 0.5779, Validation Accuracy: 0.6119, Loss: 0.6519
    Epoch   0 Batch  380/538 - Train Accuracy: 0.6113, Validation Accuracy: 0.6293, Loss: 0.6041
    Epoch   0 Batch  390/538 - Train Accuracy: 0.6401, Validation Accuracy: 0.6149, Loss: 0.5742
    Epoch   0 Batch  400/538 - Train Accuracy: 0.6107, Validation Accuracy: 0.6332, Loss: 0.5697
    Epoch   0 Batch  410/538 - Train Accuracy: 0.6336, Validation Accuracy: 0.6451, Loss: 0.5875
    Epoch   0 Batch  420/538 - Train Accuracy: 0.6498, Validation Accuracy: 0.6536, Loss: 0.5501
    Epoch   0 Batch  430/538 - Train Accuracy: 0.6508, Validation Accuracy: 0.6424, Loss: 0.5395
    Epoch   0 Batch  440/538 - Train Accuracy: 0.6555, Validation Accuracy: 0.6705, Loss: 0.5462
    Epoch   0 Batch  450/538 - Train Accuracy: 0.6830, Validation Accuracy: 0.6843, Loss: 0.5243
    Epoch   0 Batch  460/538 - Train Accuracy: 0.6657, Validation Accuracy: 0.6832, Loss: 0.4996
    Epoch   0 Batch  470/538 - Train Accuracy: 0.7083, Validation Accuracy: 0.6790, Loss: 0.4742
    Epoch   0 Batch  480/538 - Train Accuracy: 0.7059, Validation Accuracy: 0.6829, Loss: 0.4619
    Epoch   0 Batch  490/538 - Train Accuracy: 0.7206, Validation Accuracy: 0.7102, Loss: 0.4522
    Epoch   0 Batch  500/538 - Train Accuracy: 0.7296, Validation Accuracy: 0.6937, Loss: 0.4142
    Epoch   0 Batch  510/538 - Train Accuracy: 0.7227, Validation Accuracy: 0.7156, Loss: 0.4298
    Epoch   0 Batch  520/538 - Train Accuracy: 0.7033, Validation Accuracy: 0.7045, Loss: 0.4714
    Epoch   0 Batch  530/538 - Train Accuracy: 0.7246, Validation Accuracy: 0.7168, Loss: 0.4328
    Epoch   1 Batch   10/538 - Train Accuracy: 0.7176, Validation Accuracy: 0.7283, Loss: 0.4150
    Epoch   1 Batch   20/538 - Train Accuracy: 0.7734, Validation Accuracy: 0.7429, Loss: 0.4079
    Epoch   1 Batch   30/538 - Train Accuracy: 0.7281, Validation Accuracy: 0.7340, Loss: 0.4007
    Epoch   1 Batch   40/538 - Train Accuracy: 0.7686, Validation Accuracy: 0.7401, Loss: 0.3339
    Epoch   1 Batch   50/538 - Train Accuracy: 0.7604, Validation Accuracy: 0.7603, Loss: 0.3710
    Epoch   1 Batch   60/538 - Train Accuracy: 0.7854, Validation Accuracy: 0.7401, Loss: 0.3470
    Epoch   1 Batch   70/538 - Train Accuracy: 0.7775, Validation Accuracy: 0.7699, Loss: 0.3211
    Epoch   1 Batch   80/538 - Train Accuracy: 0.7754, Validation Accuracy: 0.7720, Loss: 0.3384
    Epoch   1 Batch   90/538 - Train Accuracy: 0.7855, Validation Accuracy: 0.7939, Loss: 0.3199
    Epoch   1 Batch  100/538 - Train Accuracy: 0.8406, Validation Accuracy: 0.8063, Loss: 0.2808
    Epoch   1 Batch  110/538 - Train Accuracy: 0.8314, Validation Accuracy: 0.8056, Loss: 0.2890
    Epoch   1 Batch  120/538 - Train Accuracy: 0.8391, Validation Accuracy: 0.8176, Loss: 0.2598
    Epoch   1 Batch  130/538 - Train Accuracy: 0.8544, Validation Accuracy: 0.8249, Loss: 0.2555
    Epoch   1 Batch  140/538 - Train Accuracy: 0.8238, Validation Accuracy: 0.8224, Loss: 0.2888
    Epoch   1 Batch  150/538 - Train Accuracy: 0.8484, Validation Accuracy: 0.8317, Loss: 0.2516
    Epoch   1 Batch  160/538 - Train Accuracy: 0.8268, Validation Accuracy: 0.8233, Loss: 0.2295
    Epoch   1 Batch  170/538 - Train Accuracy: 0.8449, Validation Accuracy: 0.8306, Loss: 0.2389
    Epoch   1 Batch  180/538 - Train Accuracy: 0.8564, Validation Accuracy: 0.8377, Loss: 0.2225
    Epoch   1 Batch  190/538 - Train Accuracy: 0.8436, Validation Accuracy: 0.8548, Loss: 0.2438
    Epoch   1 Batch  200/538 - Train Accuracy: 0.8756, Validation Accuracy: 0.8469, Loss: 0.1989
    Epoch   1 Batch  210/538 - Train Accuracy: 0.8486, Validation Accuracy: 0.8510, Loss: 0.2110
    Epoch   1 Batch  220/538 - Train Accuracy: 0.8384, Validation Accuracy: 0.8485, Loss: 0.1999
    Epoch   1 Batch  230/538 - Train Accuracy: 0.8643, Validation Accuracy: 0.8530, Loss: 0.1986
    Epoch   1 Batch  240/538 - Train Accuracy: 0.8807, Validation Accuracy: 0.8770, Loss: 0.1938
    Epoch   1 Batch  250/538 - Train Accuracy: 0.8902, Validation Accuracy: 0.8475, Loss: 0.1810
    Epoch   1 Batch  260/538 - Train Accuracy: 0.8517, Validation Accuracy: 0.8612, Loss: 0.1865
    Epoch   1 Batch  270/538 - Train Accuracy: 0.8701, Validation Accuracy: 0.8766, Loss: 0.1663
    Epoch   1 Batch  280/538 - Train Accuracy: 0.9103, Validation Accuracy: 0.8532, Loss: 0.1536
    Epoch   1 Batch  290/538 - Train Accuracy: 0.8980, Validation Accuracy: 0.8752, Loss: 0.1470
    Epoch   1 Batch  300/538 - Train Accuracy: 0.8778, Validation Accuracy: 0.8714, Loss: 0.1597
    Epoch   1 Batch  310/538 - Train Accuracy: 0.9055, Validation Accuracy: 0.8697, Loss: 0.1440
    Epoch   1 Batch  320/538 - Train Accuracy: 0.8921, Validation Accuracy: 0.8826, Loss: 0.1423
    Epoch   1 Batch  330/538 - Train Accuracy: 0.9103, Validation Accuracy: 0.8754, Loss: 0.1277
    Epoch   1 Batch  340/538 - Train Accuracy: 0.8670, Validation Accuracy: 0.8796, Loss: 0.1416
    Epoch   1 Batch  350/538 - Train Accuracy: 0.9079, Validation Accuracy: 0.8885, Loss: 0.1443
    Epoch   1 Batch  360/538 - Train Accuracy: 0.8994, Validation Accuracy: 0.8766, Loss: 0.1345
    Epoch   1 Batch  370/538 - Train Accuracy: 0.9004, Validation Accuracy: 0.8874, Loss: 0.1274
    Epoch   1 Batch  380/538 - Train Accuracy: 0.8982, Validation Accuracy: 0.8869, Loss: 0.1151
    Epoch   1 Batch  390/538 - Train Accuracy: 0.9241, Validation Accuracy: 0.8853, Loss: 0.0991
    Epoch   1 Batch  400/538 - Train Accuracy: 0.9007, Validation Accuracy: 0.8858, Loss: 0.1139
    Epoch   1 Batch  410/538 - Train Accuracy: 0.9027, Validation Accuracy: 0.8825, Loss: 0.1245
    Epoch   1 Batch  420/538 - Train Accuracy: 0.9021, Validation Accuracy: 0.8864, Loss: 0.1088
    Epoch   1 Batch  430/538 - Train Accuracy: 0.8971, Validation Accuracy: 0.8945, Loss: 0.0993
    Epoch   1 Batch  440/538 - Train Accuracy: 0.8799, Validation Accuracy: 0.8933, Loss: 0.1202
    Epoch   1 Batch  450/538 - Train Accuracy: 0.8996, Validation Accuracy: 0.8903, Loss: 0.1244
    Epoch   1 Batch  460/538 - Train Accuracy: 0.8895, Validation Accuracy: 0.8924, Loss: 0.1166
    Epoch   1 Batch  470/538 - Train Accuracy: 0.9183, Validation Accuracy: 0.8871, Loss: 0.0980
    Epoch   1 Batch  480/538 - Train Accuracy: 0.9167, Validation Accuracy: 0.9020, Loss: 0.0978
    Epoch   1 Batch  490/538 - Train Accuracy: 0.8983, Validation Accuracy: 0.8892, Loss: 0.0938
    Epoch   1 Batch  500/538 - Train Accuracy: 0.9189, Validation Accuracy: 0.8977, Loss: 0.0768
    Epoch   1 Batch  510/538 - Train Accuracy: 0.9109, Validation Accuracy: 0.9080, Loss: 0.0881
    Epoch   1 Batch  520/538 - Train Accuracy: 0.9145, Validation Accuracy: 0.9027, Loss: 0.0931
    Epoch   1 Batch  530/538 - Train Accuracy: 0.8951, Validation Accuracy: 0.9041, Loss: 0.0978
    Epoch   2 Batch   10/538 - Train Accuracy: 0.9141, Validation Accuracy: 0.8965, Loss: 0.0884
    Epoch   2 Batch   20/538 - Train Accuracy: 0.9256, Validation Accuracy: 0.8974, Loss: 0.0872
    Epoch   2 Batch   30/538 - Train Accuracy: 0.8932, Validation Accuracy: 0.9094, Loss: 0.1004
    Epoch   2 Batch   40/538 - Train Accuracy: 0.9288, Validation Accuracy: 0.9190, Loss: 0.0701
    Epoch   2 Batch   50/538 - Train Accuracy: 0.9289, Validation Accuracy: 0.9125, Loss: 0.0794
    Epoch   2 Batch   60/538 - Train Accuracy: 0.9256, Validation Accuracy: 0.9041, Loss: 0.0784
    Epoch   2 Batch   70/538 - Train Accuracy: 0.9040, Validation Accuracy: 0.9006, Loss: 0.0819
    Epoch   2 Batch   80/538 - Train Accuracy: 0.9260, Validation Accuracy: 0.9290, Loss: 0.0807
    Epoch   2 Batch   90/538 - Train Accuracy: 0.9271, Validation Accuracy: 0.9286, Loss: 0.0837
    Epoch   2 Batch  100/538 - Train Accuracy: 0.9445, Validation Accuracy: 0.9153, Loss: 0.0691
    Epoch   2 Batch  110/538 - Train Accuracy: 0.9428, Validation Accuracy: 0.9061, Loss: 0.0741
    Epoch   2 Batch  120/538 - Train Accuracy: 0.9365, Validation Accuracy: 0.9121, Loss: 0.0584
    Epoch   2 Batch  130/538 - Train Accuracy: 0.9174, Validation Accuracy: 0.9212, Loss: 0.0691
    Epoch   2 Batch  140/538 - Train Accuracy: 0.9025, Validation Accuracy: 0.9222, Loss: 0.0883
    Epoch   2 Batch  150/538 - Train Accuracy: 0.9365, Validation Accuracy: 0.9164, Loss: 0.0661
    Epoch   2 Batch  160/538 - Train Accuracy: 0.8953, Validation Accuracy: 0.9091, Loss: 0.0656
    Epoch   2 Batch  170/538 - Train Accuracy: 0.9275, Validation Accuracy: 0.9094, Loss: 0.0742
    Epoch   2 Batch  180/538 - Train Accuracy: 0.9360, Validation Accuracy: 0.9119, Loss: 0.0674
    Epoch   2 Batch  190/538 - Train Accuracy: 0.9174, Validation Accuracy: 0.9205, Loss: 0.0918
    Epoch   2 Batch  200/538 - Train Accuracy: 0.9389, Validation Accuracy: 0.9141, Loss: 0.0587
    Epoch   2 Batch  210/538 - Train Accuracy: 0.9275, Validation Accuracy: 0.9286, Loss: 0.0709
    Epoch   2 Batch  220/538 - Train Accuracy: 0.9023, Validation Accuracy: 0.9302, Loss: 0.0678
    Epoch   2 Batch  230/538 - Train Accuracy: 0.9260, Validation Accuracy: 0.9288, Loss: 0.0627
    Epoch   2 Batch  240/538 - Train Accuracy: 0.9424, Validation Accuracy: 0.9228, Loss: 0.0655
    Epoch   2 Batch  250/538 - Train Accuracy: 0.9408, Validation Accuracy: 0.9252, Loss: 0.0607
    Epoch   2 Batch  260/538 - Train Accuracy: 0.8979, Validation Accuracy: 0.9206, Loss: 0.0684
    Epoch   2 Batch  270/538 - Train Accuracy: 0.9377, Validation Accuracy: 0.9405, Loss: 0.0559
    Epoch   2 Batch  280/538 - Train Accuracy: 0.9368, Validation Accuracy: 0.9187, Loss: 0.0555
    Epoch   2 Batch  290/538 - Train Accuracy: 0.9576, Validation Accuracy: 0.9185, Loss: 0.0515
    Epoch   2 Batch  300/538 - Train Accuracy: 0.9256, Validation Accuracy: 0.9331, Loss: 0.0641
    Epoch   2 Batch  310/538 - Train Accuracy: 0.9563, Validation Accuracy: 0.9332, Loss: 0.0651
    Epoch   2 Batch  320/538 - Train Accuracy: 0.9386, Validation Accuracy: 0.9297, Loss: 0.0603
    Epoch   2 Batch  330/538 - Train Accuracy: 0.9481, Validation Accuracy: 0.9352, Loss: 0.0515
    Epoch   2 Batch  340/538 - Train Accuracy: 0.9355, Validation Accuracy: 0.9339, Loss: 0.0581
    Epoch   2 Batch  350/538 - Train Accuracy: 0.9453, Validation Accuracy: 0.9292, Loss: 0.0641
    Epoch   2 Batch  360/538 - Train Accuracy: 0.9445, Validation Accuracy: 0.9396, Loss: 0.0557
    Epoch   2 Batch  370/538 - Train Accuracy: 0.9494, Validation Accuracy: 0.9341, Loss: 0.0549
    Epoch   2 Batch  380/538 - Train Accuracy: 0.9439, Validation Accuracy: 0.9377, Loss: 0.0478
    Epoch   2 Batch  390/538 - Train Accuracy: 0.9362, Validation Accuracy: 0.9263, Loss: 0.0500
    Epoch   2 Batch  400/538 - Train Accuracy: 0.9568, Validation Accuracy: 0.9366, Loss: 0.0538
    Epoch   2 Batch  410/538 - Train Accuracy: 0.9533, Validation Accuracy: 0.9377, Loss: 0.0535
    Epoch   2 Batch  420/538 - Train Accuracy: 0.9410, Validation Accuracy: 0.9499, Loss: 0.0536
    Epoch   2 Batch  430/538 - Train Accuracy: 0.9240, Validation Accuracy: 0.9270, Loss: 0.0587
    Epoch   2 Batch  440/538 - Train Accuracy: 0.9334, Validation Accuracy: 0.9531, Loss: 0.0553
    Epoch   2 Batch  450/538 - Train Accuracy: 0.9180, Validation Accuracy: 0.9366, Loss: 0.0708
    Epoch   2 Batch  460/538 - Train Accuracy: 0.9295, Validation Accuracy: 0.9529, Loss: 0.0593
    Epoch   2 Batch  470/538 - Train Accuracy: 0.9481, Validation Accuracy: 0.9343, Loss: 0.0532
    Epoch   2 Batch  480/538 - Train Accuracy: 0.9390, Validation Accuracy: 0.9357, Loss: 0.0555
    Epoch   2 Batch  490/538 - Train Accuracy: 0.9371, Validation Accuracy: 0.9215, Loss: 0.0540
    Epoch   2 Batch  500/538 - Train Accuracy: 0.9680, Validation Accuracy: 0.9293, Loss: 0.0349
    Epoch   2 Batch  510/538 - Train Accuracy: 0.9362, Validation Accuracy: 0.9384, Loss: 0.0487
    Epoch   2 Batch  520/538 - Train Accuracy: 0.9297, Validation Accuracy: 0.9142, Loss: 0.0630
    Epoch   2 Batch  530/538 - Train Accuracy: 0.9158, Validation Accuracy: 0.9260, Loss: 0.0765
    Model Trained and Saved


### Save Parameters
Save the `batch_size` and `save_path` parameters for inference.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## Sentence to Sequence
To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using `vocab_to_int`
 - Convert words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    sentence = sentence.lower()
    sentence = sentence.split(' ')
    default = "<UNK>"
    
    word_ids = [vocab_to_int.get(i, vocab_to_int[default]) for i in sentence]
    
    return word_ids


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

    Tests Passed


## Translate
This will translate `translate_sentence` from English to French.


```python
translate_sentence = 'he saw a old yellow truck .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         source_sequence_length: [len(translate_sentence)]*batch_size,
                                         keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in translate_logits]))
print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))

```

    INFO:tensorflow:Restoring parameters from checkpoints/dev
    Input
      Word Ids:      [201, 112, 141, 97, 116, 17, 19]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']
    
    Prediction
      Word Ids:      [357, 55, 330, 237, 232, 257, 168, 184, 1]
      French Words: il a vu le vieux camion jaune . <EOS>


## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
