import torch
import torch.profiler
from tqdm import tqdm

def inference_step(model, processor, device, inputs, preprocessing_fn, postprocessing_fn):
  with torch.no_grad():
    if preprocessing_fn is not None:
        inputs = preprocessing_fn(inputs, processor)
    inputs = inputs.to(device)
    outputs = model.generate(inputs)
    if postprocessing_fn is not None:
        outputs = postprocessing_fn(outputs, processor)
    return outputs

def profile_inference(model, 
                      processor, 
                      X, 
                      log_out_dir, 
                      device, 
                      steps=10,
                      preprocessing_fn=None,
                      postprocessing_fn=None):
    model.to(device)
    model.eval()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available() and device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_out_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in tqdm(range(steps)):
            inference_step(model, processor, device, X, preprocessing_fn, postprocessing_fn)
            prof.step()