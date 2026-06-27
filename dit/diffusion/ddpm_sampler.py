import torch
from tqdm import tqdm



@torch.no_grad()
def ddpm_sample(model, schedular, shape, y, device, cfg_scale = 1.0, show_progress = True, return_intermediates = False):

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

    intermediates = [] if return_intermediates else None


    timesteps = list(reversed(range(schedular.timesteps))) #shape: (1000, ) : [999, 998, ...0]
    #print(timesteps[:10])

    if show_progress:
        timesteps = tqdm(timesteps, desc = ' sampling')
    
    for t in timesteps:
        #since there is a batch of images to be generated from a batch of noise, we need time step 
        t_batch = torch.full((B, ), t, device = device,  dtype = torch.long)

        if cfg_scale == 1.0:  # no guidance at all, use class label as in
            eps_pred = model(x, t_batch, y)
        


          # eps_cond: what the model predicts when it knows "this should be a dog" via label

        # eps_uncond: what the model predicts with the null token, i.e. no class info at all (model trained for such cases via 10% class dropout training)

        else:
            # for eps_pred = eps_uncond + cfg_scale · (eps_cond - eps_uncond)

            # 1. if cfg scale = 1.0 then eps_pred = eps_cond {CLASSIFIER BASED GUIDANCE: via the label}

            # 2. if cfg scale = 0.0 then eps_pred = eps_uncond {FULLY CLASSFIER FREE GUIDANCE: model ignores label entirely}

            # 3. if cfg scale > 1.0 then push 3x harder on class based conditioning, model follws class label strcitly

            null_y = torch.full_like(y, model.condition_embed.y_embed.null_token)
            eps_cond = model(x, t_batch, y)
            eps_uncond = model(x, t_batch, null_y)
            eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        
        # estimate clean image from current xt: pred x0 from xt using inverted q_sample

        x0_hat = schedular.predict_x0_from_eps(x, t_batch, eps_pred)

        x0_hat = x0_hat.clamp(-1, 1)  # keep estimate in valid pixel range


        #compute posterior mean
        mean = schedular.q_posterior_mean(x0_hat, x, t_batch)

        if t > 0:
            std = schedular.posterior_std(t_batch, x.shape)
            z = torch.randn_like(x)
            x = mean + std * z
        
        else:
            x = mean # final step, no noise added, sigma = 0
        
        if return_intermediates:
            # store x0_hat (the cleaner estimate), not x — makes a much better looking GIF
            intermediates.append(x0_hat.cpu())
            return x, intermediates

        
    return x


if __name__ == "__main__":
    import sys
    sys.path.insert(0,".")
    from dit.models.dit import DIT
    from dit.diffusion.schedular import NoiseSchedular

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    model = DIT(n_blocks=2).to(device).eval()
    schedular = NoiseSchedular(timesteps = 20, device = device)

    y = torch.randint(0, 10, (4,), device=device)
    samples = ddpm_sample(model = model, schedular = schedular, shape = (4, 3, 32, 32), y = y, device = device)

    print("samples shape:", samples.shape)
    assert not torch.isnan(samples).any(), "NaNs in output!"



