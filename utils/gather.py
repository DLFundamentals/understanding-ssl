import torch
import torch.distributed as dist

# save first and then gather for backward
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

# # gather first and then save for backward
# class GatherLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, tensor):
#         world_size = dist.get_world_size()
#         gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
#         dist.all_gather(gathered_tensor, tensor)  # Collects tensors across GPUs
#         ctx.save_for_backward(torch.cat(gathered_tensor, dim=0))
#         ctx.world_size = world_size
#         return tuple(gathered_tensor)  # Concatenates all gathered tensors

#     @staticmethod
#     def backward(ctx, *grad_output):
#         input_tensor, = ctx.saved_tensors
#         world_size = ctx.world_size
#         grad_input = grad_output[dist.get_rank()]  # Splits gradients properly
#         return grad_input