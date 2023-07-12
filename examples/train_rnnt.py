from datasets.librispeech import iterate
from tinygrad.tensor import Tensor, Function
from tinygrad.nn.optim import LAMB, get_parameters
import numpy as np
import time

from models.rnnt import RNNT

def rnnt_loss_forward(x, y, blank=28):
  T, U, _ = x.shape

  # TO DO: update this to tensors. Need to work around item assignment.
  alphas = np.zeros((T,U), dtype=np.int32)
  alphas[1,0] = 1

  for t in range(1, T):
    alphas[t, 0] = alphas[t - 1, 0] + x[t - 1, 0, blank].numpy()

  for u in range(1, U):
    alphas[0, u] = alphas[0, u - 1] + x[0, u - 1, int(y[u - 1].numpy())].numpy()

  for t in range(1, T):
    for u in range(1, U):
      start = time.time()
      print(f"{t*U + u}/{T*U}")
      no_emit = alphas[t - 1, u] + x[t - 1, u, blank].numpy()
      emit = alphas[t, u - 1] + x[t, u - 1, int(y[u - 1].numpy())].numpy()
      alphas[t, u] = np.log(np.exp(emit) + np.exp(no_emit))
      print(f"Time: {time.time() - start}")

  log_likelihood = alphas[T - 1, U - 1] + x[T - 1, U - 1, blank].numpy()
  return alphas, -log_likelihood

def rnnt_loss_backward(x, y, blank=28):
  T, U, _ = x.shape

  # TO DO: update this to tensors.
  betas = np.zeros((T, U))
  betas[T - 1, U - 1] = x[T - 1, U - 1, blank].numpy()

  for t in reversed(range(T - 1)):
    betas[t, U - 1] = betas[t + 1, U - 1] + x[t, U - 1, blank].numpy()

  for u in reversed(range(U - 1)):
    betas[T - 1, u] = betas[T - 1, u + 1] + x[T - 1, u, y[u].numpy()].numpy()

  for t in reversed(range(T - 1)):
    for u in reversed(range(U - 1)):
      no_emit = betas[t + 1, u] + x[t, u, blank].numpy()
      emit = betas[t, u + 1] + x[t, u, y[u].numpy()].numpy()
      betas[t, u] = np.log(np.exp(emit) + np.exp(no_emit))

  return betas

def rnnt_loss_grad(x, alphas, betas, y, blank=28):
  T, U, _ = x.shape

  # TO DO: update this to tensors.
  grads = np.full(x.shape, -np.inf)
  log_likelihood = betas[0, 0]

  grads[T - 1, U - 1, blank] = alphas[T - 1, U - 1]
  grads[:T - 1, :, blank] = alphas[:T - 1, : ] + betas[1:, :]

  for u, l in enumerate(y):
    grads[:, u, l] = alphas[:, u] + betas[:, u + 1]

  grads = -(np.exp(grads + x - log_likelihood))

  return grads

def rnnt_loss(x, y, blank=28):
  alphas, log_likelihood = rnnt_loss_forward(x, y, blank)
  betas = rnnt_loss_backward(x, y, blank)
  grads = rnnt_loss_grad(x, alphas, betas, y, blank)
  return log_likelihood, grads

def rnnt_loss_batch(x, x_lens, y, y_lens, blank=28):
  grads = np.zeros_like(x) # Need?
  losses = []
  for b in range(x.shape[0]):
    # TO DO: check if realize is required
    t = int(Tensor(x_lens)[b].numpy()/2)
    u = int(Tensor(y_lens)[b].numpy())
    loss, grad = rnnt_loss(Tensor(x)[b, :t, :u, :], Tensor(y)[b, :u], blank)
    losses.append(loss)
    grads[b, :t, :u, :] = grad
  return Tensor(losses, dtype=np.float32), grads

class RNNTLoss(Function):
  def forward(self, x, x_lens, y, y_lens):
    self.x, self.x_lens, self.y, self.y_lens = x, x_lens, y, y_lens

    loss, grads = rnnt_loss_batch(x, x_lens, y, y_lens) 
    self.grads = grads

    return Tensor(loss, x.device, loss.dtype)

  def backward(self, grad_output):
    return Tensor(self.grads, grad_output.device, self.grads.dtype), None, None, None

if __name__ == "__main__":
  # Tinygrad set flags
  Tensor.training = True
  np.set_printoptions(linewidth=200) # Do we need this

  # Load Model
  mdl = RNNT()
  #mdl.load_from_pretrained()
  optim = LAMB(get_parameters(mdl), lr=4e-3, wd=1e-3)

  LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

  optim.zero_grad()
  for _ in range(100):
    # TO DO: update to training data set.
    for X, Y, y_raw in iterate(dataset='dev-clean', val=False, bs=1):
      x, x_lens, y = Tensor(X[0]), Tensor(X[1]), Tensor(Y)
      out = mdl(x, y)
      tt = mdl.decode(x, x_lens)
      for n, t in enumerate(tt):
        tnp = np.array(t)
        print(["".join([LABELS[int(tnp[i])] for i in range(tnp.shape[0])])])
        print(y_raw[n])

      # print(out.shape)
      print("forward done")
      # TO DO: update y_lens + set blank
      loss = RNNTLoss.apply(out.log_softmax(), x_lens, y, Tensor([y.shape[1]])).mean() # Why pass tensor [10,10,10,10]?
      print("loss done")
      loss.backward()
      print("backward done")
      optim.step()
      print("step done")
      optim.zero_grad()
      print("zero grad done")

      print("loss %.4f" % loss.numpy()[0])