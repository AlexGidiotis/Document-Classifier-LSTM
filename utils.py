import numpy as np

from keras import backend as K


def visualize_attention(test_seq,
    model,
    id2wrd,
    n):
    """
    Visualize the top n words that the model pays attention to. 
    We first do a forward pass and get the output of the LSTM layer.
    THen we apply the function of the Attention layer and get the weights.
    Finally we obtain and print the words of the input sequence 
    that have these weights.


    """

    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[4].output])
    out = get_layer_output([test_seq, ])[0]  # test mode

    att_w = model.layers[5].get_weights()

    eij = np.tanh(np.dot(out[0], att_w[0]))
    ai = np.exp(eij)
    weights = ai/np.sum(ai)
    weights = np.sum(weights,axis=1)

    topKeys = np.argpartition(weights,-n)[-n:]

    print ' '.join([id2wrd[wrd_id] for wrd_id in test_seq[0] if wrd_id != 0.]) 
    
    for k in test_seq[0][topKeys]:
        if k != 0.:
            print id2wrd[k]
    
    return