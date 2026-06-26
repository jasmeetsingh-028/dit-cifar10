import torch
import torch.nn.functional as F

def training_loss(model, schedular, x0, y):

    """

    Single training step: sample t and noise, corrupt x0, predict noise, compute MSE.
    x0: (B, C = 3, H = 32, W= 32)
    y: (B, ) class labels

    """

    B, _, _, _ = x0.shape
    device = x0.device

    # sample a random timestep per image in the batch
    t = torch.randint(0, schedular.timesteps, (B,), device = device, dtype = torch.long)


    # sample noise, same shape as the image x0
    noise = torch.rand_like(x0)

    #corrupt x0 to noise level at time steps t using the closed for noisificatiob
    xt = schedular.q_sample(x0, t, noise)

    # model predicts the noise that was accumulated till time step 't'
    eps_pred = model(xt, t, y)

    #MSE between the predicted and the true noise
    loss = F.mse_loss(eps_pred, noise)

    return loss
