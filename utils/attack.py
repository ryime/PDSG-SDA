import torchattacks
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
import numpy as np

def set_PDSG(model: nn.Module):
    for name, m in model.named_children():
        if isinstance(m, neuron.LIFNode):
            setattr(model, name, PDSG_LIFNode(m.tau, m.decay_input, m.v_threshold, m.v_reset, m.detach_reset))
        else:
            set_PDSG(m)
    return model

class FGSM(torchattacks.FGSM):
    def __init__(self, model, encoder, eps=8/255, min_val=0, max_val=1):
        super().__init__(model, eps)
        self.model = SNNContainer(model, encoder)
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, inputs, labels=None, *args, **kwargs):
        self.model.training = self.model.model.training
        return super().__call__(inputs, labels, *args, **kwargs)
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=self.min_val, max=self.max_val).detach()

        return adv_images
        
class PGD(torchattacks.PGD):
    def __init__(self, model, encoder, eps=8/255, steps=10, random_start=True, min_val=0, max_val=1):
        super().__init__(model, eps, eps/4, steps, random_start)
                
        self.model = SNNContainer(model, encoder)
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, inputs, labels=None, *args, **kwargs):
        self.model.training = self.model.model.training
        return super().__call__(inputs, labels, *args, **kwargs)
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.min_val, max=self.max_val).detach()

        return adv_images

class CWLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, labels):
        num_classes = inputs.shape[1]
        loss = 0
        labels = nn.functional.one_hot(labels, num_classes=num_classes).float()
        for b in range(inputs.shape[0]):
            t = inputs[b][labels[b] == 1] - torch.max(inputs[b][labels[b] != 1])
            loss += torch.max(t, torch.zeros_like(t))
        return loss / inputs.shape[0]

@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

# PDSG backward function
def piecewise_quadratic_backward(grad_output: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor):
    sigma = sigma.expand_as(x)
    grad = torch.exp(- ((x - 0.5 * sigma)/(np.sqrt(2)*sigma))**2) / (np.sqrt(2*np.pi)*sigma)
    grad_input = grad_output * grad
    return grad_input, None, None, None


class surrogate_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, sigma: torch.Tensor):
        if x.requires_grad:
            ctx.save_for_backward(x, sigma)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.saved_tensors[1])

class PDSG(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, v: torch.Tensor):
        with torch.no_grad():
            if len(x.shape) == 4: # for conv layer
                if x.shape[2] == x.shape[3]: # conventional conv layer
                    dim = (0,1,3,4)
                else: # transformer layer
                    dim = (0,1,2,4)
            else: # for linear layer
                dim = tuple(range(len(x.shape)))
            sigma = v.std(dim=dim, keepdim=True).squeeze(0)
            sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
        return surrogate_func.apply(x, sigma)

class SDA(nn.Module):
    def __init__(self, model, k_init=10, N=500, batch_limit=64):
        super().__init__()
        self.k_init = k_init
        self.N = N
        self.model = model
        self.batch_limit = batch_limit # prevent GPU out of memory when k is large

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        for l in self.model.modules():
            if isinstance(l, neuron.BaseNode):
                l.train() # enable back-propagation
            else:
                l.eval()   
        assert inputs.shape[1] == 1
        criterion = CWLoss()

        selected_mask = torch.zeros_like(inputs)
        FDs = torch.zeros_like(inputs).fill_(torch.inf)
        adv_inputs = inputs.clone().detach().to(inputs)
        success_flag = False
        
        # Generation process
        for n in range(self.N):
            # calculate gradients
            adv_inputs = adv_inputs.detach()
            adv_inputs.requires_grad_(True)
            outputs = self.model(adv_inputs).mean(0)
            loss = criterion(outputs, labels)
            
            grad = torch.autograd.grad(loss, adv_inputs,
                                retain_graph=False, create_graph=False)[0]
            
            # ---------------------------------------------------------------------------
            # step 1: select contributing gradients. (1-2x)*g<=0 equals to 0<=x-sgn(g)<=1
            grad_mask = adv_inputs - grad.sign()
            grad_mask = torch.bitwise_and(grad_mask >= 0.0, grad_mask <= 1.0)
            grad = grad * grad_mask
            
            # prevent gradient vanishing (seldom occurs)
            if (grad != 0).any():
                random_value = torch.rand_like(grad) * grad[grad != 0].abs().min()
            else:
                random_value = torch.rand_like(grad)
            grad[grad == 0] = random_value[grad == 0]
            
            # exclude selected pixels
            grad = grad * (1 - selected_mask)
            
            # ---------------------------------------------------------------------------
            # step 2: select topk gradients
            
            # update k
            k = (n+1) * self.k_init
            
            # indices of topk gradients
            indices = list(np.unravel_index(torch.topk(grad.abs().flatten(), k=k)[1].cpu(), shape=inputs.shape))
            
            # parallel perturb and forward
            indices = [list(i) for i in indices]
            indices[1] = list(np.linspace(0, k-1, k, dtype=int) % self.batch_limit)
            with torch.no_grad():
                if k <= self.batch_limit:
                    parallel_adv_inputs = adv_inputs.repeat(1,k,1,1,1)
                    parallel_adv_inputs[indices] = 1 - parallel_adv_inputs[indices]
                    outputs = self.model(parallel_adv_inputs).mean(0)
                    loss_each_batch = []
                    for b in range(k):
                        loss_each_batch.append(criterion(outputs[b].unsqueeze(0), labels))
                    loss_each_batch = torch.stack(loss_each_batch).squeeze(1)
                else: # When k is large, GPU out of memory may occur. Hence divide k by groups.
                    loss_each_batch = []
                    for kb in range(int(np.ceil(k / self.batch_limit))):
                        start = kb * self.batch_limit
                        end = np.min([(kb+1)*self.batch_limit, k])
                        parallel_adv_inputs = adv_inputs.repeat(1,end-start,1,1,1)
                        index = list(np.array(indices)[:,start:end])
                        parallel_adv_inputs[index] = 1 - parallel_adv_inputs[index]
                        outputs = self.model(parallel_adv_inputs).mean(0)
                        for b in range(outputs.shape[0]):
                            loss_each_batch.append(criterion(outputs[b].unsqueeze(0), labels))
                    loss_each_batch = torch.stack(loss_each_batch).squeeze(1)           
            indices[1] = np.zeros(k, dtype=int)
            
            # ---------------------------------------------------------------------------
            # step 3: calculate FDs, FDs=[loss(adv)-loss(ori)]/delta. We expect loss(adv) < loss(ori)
            FDs[indices] = loss_each_batch - loss.repeat(k)
            selected_mask[indices] = 1
            adv_inputs.requires_grad_(False)
            selected_mask[FDs > 0] = 0
            FDs[FDs > 0] = torch.inf
            
            # update adversarial examples
            adv_inputs = inputs * (1 - selected_mask) + (1 - inputs) * selected_mask
            
            # test if attack is successful
            with torch.no_grad():
                outputs = self.model(adv_inputs).mean(0)
                _, predicted = outputs.max(1)
                if predicted.eq(labels).cpu().sum().item() == 0:
                    success_flag = True
                    break
        
        if not success_flag:
            return inputs
        
        # Reduction process
        final_adv_inputs = adv_inputs
        n_count = (selected_mask == 1).sum().item()
        FDs_sort = torch.topk(FDs.abs().flatten(), k=n_count, largest=False)[1].cpu()
        left_ptr = 0
        right_ptr = n_count - 1
        while left_ptr <= right_ptr: # binary search
            ptr = (left_ptr + right_ptr) // 2
            FDs_indices = np.unravel_index(FDs_sort[0:ptr+1], shape=inputs.shape)
            temp_adv_inputs = adv_inputs.clone().detach()
            temp_adv_inputs[FDs_indices] = 1 - adv_inputs[FDs_indices]
            with torch.no_grad():
                outputs = self.model(temp_adv_inputs).mean(0)
                _, predicted = outputs.max(1)
                if predicted.eq(labels).cpu().sum().item() == 0: # still adversarial
                    left_ptr = ptr + 1
                    final_adv_inputs = temp_adv_inputs
                else: # not adversarial
                    right_ptr = ptr - 1
 
        return final_adv_inputs 

class SNNContainer(nn.Module):
    def __init__(self, model, encoder):
        super().__init__()
        self.model = model
        self.encoder = encoder
    
    def forward(self, x):
        for l in self.model.modules():
            if isinstance(l, neuron.BaseNode):
                l.train() # enable back-propagation
            else:
                l.eval()
        return torch.mean(self.model(self.encoder(x)), dim=0)

class PDSG_LIFNode(nn.Module): # modified spikingjelly's LIFNode
    def __init__(self, tau: float=2.0, decay_input=True, v_threshold: float=1.0, v_reset: float=0.0, detach_reset: bool=False):
        super().__init__()
        self.tau = tau
        self.decay_input = decay_input
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = PDSG()
        
    def forward(self, x: torch.Tensor):
        mem = 0.
        mem_pot = []
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            if t == 0:
                if self.decay_input:
                    mem = (x[t, ...] + self.v_reset) / self.tau
                else:
                    mem = x[t, ...] + self.v_reset / self.tau
            else:
                if self.decay_input:
                    mem = mem + (x[t, ...] - (mem - self.v_reset)) / self.tau
                else:
                    mem = mem + (-(mem - self.v_reset)) / self.tau + x[t, ...]
            # record membrane potential[1:t] to calculate standard deviation (following Time-Accumulated Batch Normalization (TAB))
            mem_pot.append(mem.clone().detach())
            spike = self.surrogate_function(mem - self.v_threshold, torch.stack(mem_pot, dim=0))
            if self.detach_reset:
                spike_d = spike.detach()
            else:
                spike_d = spike
            if self.v_reset is not None:
                mem = mem * (1 - spike_d) + self.v_reset * spike_d
            else:
                mem -= spike_d * self.v_threshold
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        return out
    