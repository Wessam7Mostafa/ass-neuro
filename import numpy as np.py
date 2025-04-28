import numpy as np
text = "I love deep learning"
words = text.split()

word_to_idx = {word: idx for idx, word in enumerate(words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
X = [word_to_idx[w] for w in words[:-1]]  
Y = word_to_idx[words[-1]]  
vocab_size = len(words)
def one_hot(idx, vocab_size):
    vec = np.zeros(vocab_size)
    vec[idx] = 1
    return vec

X_oh = np.array([one_hot(idx, vocab_size) for idx in X]) 


hidden_size = 8  

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 
Why = np.random.randn(vocab_size, hidden_size) * 0.01  

bh = np.zeros((hidden_size, 1))  
by = np.zeros((vocab_size, 1)) 


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


lr = 0.1
epochs = 500

for epoch in range(epochs):
    hs = np.zeros((hidden_size, 1)) 
    loss = 0

    
    for t in range(3):
        x = X_oh[t].reshape(-1, 1)  
        hs = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hs) + bh)  
    y_pred = np.dot(Why, hs) + by  
    probs = softmax(y_pred)  

    loss = -np.log(probs[Y][0])  

    
    dy = probs
    dy[Y] -= 1 

    dWhy = np.dot(dy, hs.T)
    dby = dy

    dh = np.dot(Why.T, dy) * (1 - hs * hs)  

    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros_like(bh)

    for t in range(2, -1, -1):
        x = X_oh[t].reshape(-1, 1)
        dWxh += np.dot(dh, x.T)
        dWhh += np.dot(dh, hs.T)
        dbh += dh

    
    for param, dparam in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby]):
        param -= lr * dparam

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


print("\n=== Testing ===")
hs = np.zeros((hidden_size, 1))
for t in range(3):
    x = X_oh[t].reshape(-1, 1)
    hs = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hs) + bh)

y_pred = np.dot(Why, hs) + by
probs = softmax(y_pred)
pred_idx = np.argmax(probs)
predicted_word = idx_to_word[pred_idx]

print(f"Predicted 4th word: {predicted_word}")
print(f"Actual 4th word: {words[-1]}")
