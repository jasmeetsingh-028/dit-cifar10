import torch
import torch.nn as nn
import copy

class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the model that updates slowly toward the
    training weights. 
    - Used for sampling, 
    - EMA weights produce better, more stable generation quality than raw training weights.
    """

    def __init__(self, model: nn.Module, decay: float, warmup_steps: int = 0):

        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0

        self.ema_model = copy.deepcopy(model)

         #deep and shallow copy:
         # Shallow copy copies the outer object but the inner elements still point to the same objects in memory : 
         # change a nested value and it changes in both copies.
         # Deep copy recursively copies everything, including nested objects, so the two copies are fully independent:
         # change one and the other is unaffected.
        self.ema_model.eval()

        for p in self.ema_model.parameters():
            p.requires_grad_(False) 
    
    @torch.no_grad()
    def update(self, model:nn.Module):
        self.step += 1

        # 1. during warup steps (till n number of steps): copy weights directly, as for first few steps the model weight optimizations are extremly random
        if self.step <= self.warmup_steps:
            for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.copy_(p.data)
            return
        
        # 2. ema: from step n+1 till ith step: 
        # EMA is blending in a little bit of the model's weights at every step, 
        # with older contributions decaying away geometrically.
        decay = self.decay

        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha = 1 - decay)
            # 1st operation: ema_p.data * decay =  ema_p.data
            # 2nd operation: add ema_p.data with p.data * (1 - decay)
            # entire operation: ema_p.data = (ema_p.data * decay) + (p.data * (1-decay))
            # In-place operations: In-place operations update a variable’s value directly in the same memory location, instead of creating a new object.


        # Buffers

        # model.parameters() only gives you the learnable weights 
        # it skips buffers entirely. So when copying state into ema_model, 
        # you need a separate loop over model.buffers() to make sure those 
        # non-trainable-but-necessary values also get copied across.
        # {no ema needed}
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)
    
    def state_dict(self):
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dit.models.dit import  DIT

    model = DIT(n_blocks=2)
    ema = EMA(model, decay = 0.99, warmup_steps=5)

    # snapshot ema weights before any updates
    initial_ema_weight = ema.ema_model.dit_blocks[0].attn.qkv.weight.clone()

    for step in range(20):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01) # simulating an optimizer step

        ema.update(model)

    final_ema_weight = ema.ema_model.dit_blocks[0].attn.qkv.weight.clone()

    print("ema weight changed from initial:", not torch.allclose(initial_ema_weight, final_ema_weight))
