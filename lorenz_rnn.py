import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
INPUT_SIZE = 3
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3
LR = 0.01

# Create dataset
para = (10.0, 28, 8.0/3.0)

#define lorenz equations
def f(state, t):
    x, y, z = state
    sigma, rho, beta = para
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z
# time points
t_seq = np.arange(0, 40, 0.01)

# initial values of x, y, z
state0 = [1.0, 1.0, 1.0]
# solve ODE
states = odeint(f, state0, t_seq)
inputs = torch.FloatTensor(states[:-1]).view(1, states.shape[0] - 1, 3)
targets = torch.FloatTensor(states[1:]).view(1, states.shape[0] - 1, 3)

print(states[:10])

# Define the model
class RnnNet(nn.Module):
    def __init__(self,):
        super(RnnNet, self).__init__()

        self.rnn = nn.RNN(input_size = INPUT_SIZE,
                          hidden_size = HIDDEN_SIZE,
                          num_layers = 1,
                          batch_first = True)
        """
        self.rnn2 = nn.RNN(input_size = HIDDEN_SIZE,
                           hidden_size = HIDDEN_SIZE,
                           num_layers = 1,
                           batch_first = True)
        """
        self.linear = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
    def forward(self, x, hidden):
        out, hidden = self.rnn(x,hidden)
        # [1,seq_len, h] => [seq_len, h]
        # out, hidden = self.rnn2(out1, hidden1)
        out = out.view(-1, HIDDEN_SIZE)
        out = self.linear(out) # [seq_len, h] => [seq_len, 1]
        out = out.unsqueeze(dim=0) #[seq_len, 1] => [1, seq_len, 1]
        return out, hidden

# Train the model
model = RnnNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), LR)

hidden = Variable(torch.zeros(1,1,HIDDEN_SIZE))
inputs = Variable(inputs)
targets = Variable(targets)

for iter in range(1001):
    
    output, hidden = model(inputs, hidden)
    hidden = hidden.data
    loss = criterion(output, targets)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print("Iteration{} : loss {}".format(iter, loss.item()))
    
# Give any initial point predict the following points and Visualize the result
predictions = []
input_point = inputs[:, 30, :]
for _ in range(inputs.shape[1]):
    input_point = input_point.view(1, 1, 3)
    (pred, hidden) = model(input_point, hidden)
    input_point = pred
    predictions.append(pred.detach().numpy().ravel())
inputs = inputs.data.numpy().ravel()
predictions = np.asarray(predictions)
print(predictions)
print(predictions[:,0])
fig = plt.figure()
ax = fig.gca(projection='3d')

print(predictions.shape)
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2])
plt.show()
plt.savefig("64_rnn.png")

print(states[:10])
fig1 = plt.figure()
ax =fig1.gca(projection = '3d')
ax.plot(states[:,0], states[:,1], states[:,2])

plt.savefig("out_exact.png")
