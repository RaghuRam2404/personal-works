# Week 1 — Glossary

**Tensor**: Multi-dimensional array with a fixed dtype and device; the fundamental data unit in PyTorch.

**Broadcasting**: PyTorch rule for operating on tensors with mismatched shapes by virtually expanding dimensions of size 1.

**requires_grad**: Flag on a tensor that tells PyTorch to track operations on it in the computational graph.

**Computational graph**: Directed acyclic graph recording operations on tensors; used to compute gradients via backpropagation.

**Autograd**: PyTorch's automatic differentiation engine; builds the computational graph dynamically during the forward pass.

**backward()**: Method that traverses the computational graph in reverse and accumulates gradients in `.grad` on leaf tensors.

**Leaf tensor**: A tensor created directly by the user (not as the output of an operation); model parameters are leaf tensors.

**optimizer.zero_grad()**: Resets accumulated `.grad` on all tracked parameters to zero; must be called before every backward pass.

**optimizer.step()**: Updates model parameters using the currently stored `.grad` values according to the optimizer's update rule.

**Loss function**: A scalar-valued function measuring the discrepancy between model predictions and ground truth.

**torch.no_grad()**: Context manager that disables gradient tracking; use during inference or validation to save memory and compute.

**model.eval()**: Switches the model to evaluation mode — disables dropout and uses running statistics for batch norm.

**model.train()**: Switches the model back to training mode; always call after a validation loop.

**Stride**: The step size (in memory) between consecutive elements along each dimension of a tensor; governs how `view` and `transpose` work.

**micrograd**: Karpathy's minimal scalar autograd engine; implementing it from scratch is the best way to internalize backprop.
