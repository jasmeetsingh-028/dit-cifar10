import torch
import torch.nn.functional as F

def training_loss(model, schedular, x0, y):

    """

    Single training step: sample t and noise, corrupt x0, predict noise, compute MSE.
    x0: (B, C = 3, H = 32, W= 32)
    y: (B, ) class labels

    image 1: t=410, noise_1 (random)  →  x_t1 = √ᾱ₄₁₀·x0_1 + √(1-ᾱ₄₁₀)·noise_1
    image 2: t=7,   noise_2 (random)  →  x_t2 = √ᾱ₇  ·x0_2 + √(1-ᾱ₇)  ·noise_2
    image 3: t=999, noise_3 (random)  →  x_t3 = √ᾱ₉₉₉·x0_3 + √(1-ᾱ₉₉₉)·noise_3
    image 4: t=2,   noise_4 (random)  →  x_t4 = √ᾱ₂  ·x0_4 + √(1-ᾱ₂) ·noise_4

    """

    B, _, _, _ = x0.shape
    device = x0.device

    # sample a random timestep per image in the batch
    t = torch.randint(0, schedular.timesteps, (B,), device = device, dtype = torch.long)


    # sample noise, same shape as the image x0
    # batch of noises
    noise = torch.rand_like(x0)  # every image gets its own independent noise sample 

    # this noise will be added to x0's to turn it into xt's
    
    # corrupt x0 to noise level at time steps t using the closed form noisification
    xt = schedular.q_sample(x0, t, noise)

    # model predicts the noise that was accumulated till time step 't'
    eps_pred = model(xt, t, y)
    

    ## My understanding: its like telling model that noise _1 is the noise is that was added to x0 to turn it into xt now your task it to predict noise_1 to remove it from the xt

    #MSE between the predicted and the true noise
    loss = F.mse_loss(eps_pred, noise)

    return loss

if __name__ == "__main__":

    import sys
    sys.path.insert(0, ".")
    from dit.models.dit import DIT
    from dit.diffusion.schedular import NoiseSchedular


    model = DIT()
    schedule = NoiseSchedular(timesteps=1000)

    x0 = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4, ))

    loss = training_loss(model, schedule, x0, y)

    print("loss:", loss.item())


    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"params with gradients: {len(grad_norms)}")