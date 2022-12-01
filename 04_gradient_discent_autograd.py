import torch
import time
### f = w * x

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x
# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 100

start = time.perf_counter()
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradients = backward
    l.backward()    # dl/dw
    
    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients to avoid accumolation in w.grad
    w.grad.zero_()

    if epoch % int(n_iters/10) == 0:
        print(f"Epoch: {epoch + 1}\tWeights: {w:.3f}\tLoss:{l:.8f}\tEpoch time:{(time.perf_counter() - start):.8f}")


print(f"Prediction before training: f(5) = {forward(5):.3f}\tTraining time:{(time.perf_counter() - start):.8f}")