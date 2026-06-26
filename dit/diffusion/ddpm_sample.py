import torch
from tqdm import tqdm



@torch.no_grad()
def ddpm_smaple(model, schedular, shape, y, device, cfg_scale = 1.0, show_progress = True):

    """
    Full reverse diffusion loop: pure noise -> generated image.

    model:     DiT instance (should be in eval mode, ideally EMA weights)
    schedule:  NoiseSchedule instance
    shape:     (B, C, H, W) shape of images to generate
    y:         (B,) class labels to condition on
    cfg_scale: classifier-free guidance scale. 1.0 = no guidance (pure conditional)
    """

    B = shape[0]
    x = torch.randn(shape, device = device)  # batch of pure noise

    timesteps = list(reversed(range(schedular.timesteps))) #shape: (1000, ) : [999, 998, ...0]
    print(timesteps[:10])

    if show_progress:
        timesteps = tqdm(timesteps, desc = ' sampling')
    
    for t in timesteps:
        #since there is a batch of images to be generated from a batch of noise, we need time step 
        t_batch = torch.full((B, ), t, device = device,  dtype = torch.long)

        if cfg_scale == 1.0:  # no guidance at all, use class label as in




            # eps_cond: what the model predicts when it knows "this should be a dog" via label

            # eps_uncond: what the model predicts with the null token, i.e. no class info at all (model trained for such cases via 10% class dropout training)

            # for eps_pred = eps_uncond + cfg_scale · (eps_cond - eps_uncond)

            # 1. if cfg scale = 1.0 then eps_pred = eps_cond {CLASSIFIER BASED GUIDANCE: via the label}

            # 2. if cfg scale = 0.0 then eps_pred = eps_uncond {FULLY CLASSFIER FREE GUIDANCE: model ignores label entirely}

            # 3. if cfg scale > 1.0 then push 3x harder on class based conditioning, model follws class label strcitly

            


