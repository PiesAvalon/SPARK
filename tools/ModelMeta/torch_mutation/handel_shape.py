import torch
from torch.nn import functional as F
from torch_mutation.config import device

def handle_shape_final(dead, origin):
    if dead.shape == origin.shape:
        return dead, origin
    if len(origin.shape) == 1:
        if len(dead.shape) == 2:
            dead = dead[:, :1]
            dead = dead.squeeze(-1)
            return dead, origin
        if len(dead.shape) == 3:
            dead = dead[:, :1, :1]
            dead = dead.squeeze(-1)
            dead = dead.squeeze(-1)
            return dead, origin
        if len(dead.shape) == 4:
            dead = dead[:, :1, :1, :1]
            dead = dead.squeeze(-1)
            dead = dead.squeeze(-1)
            dead = dead.squeeze(-1)
            return dead, origin
    if len(origin.shape) == 3 and len(dead.shape) == 4:
        dead = dead[:, :, :, :1]
        dead = dead.squeeze(-1)
        if dead.shape[1] > origin.shape[1]:
            dead = dead[:, :origin.shape[1], :]
        elif dead.shape[1] < origin.shape[1]:
            pad = (0, 0, origin.shape[1] - dead.shape[1], 0)
            dead = F.pad(dead, pad, "constant", 0)
        if dead.shape[2] > origin.shape[2]:
            dead = dead[:, :, :origin.shape[2]]
        elif dead.shape[2] < origin.shape[2]:
            pad = (0, origin.shape[2] - dead.shape[2], 0, 0)
            dead = F.pad(dead, pad, "constant", 0)
        return dead, origin

    if len(origin.shape) == 3 and len(dead.shape) == 2:
        dead = dead.unsqueeze(-1)
        if dead.shape[1] > origin.shape[1]:
            dead = dead[:, :origin.shape[1], :]
        elif dead.shape[1] < origin.shape[1]:
            pad = (0, 0, origin.shape[1] - dead.shape[1], 0)
            dead = F.pad(dead, pad, "constant", 0)
        if dead.shape[2] > origin.shape[2]:
            dead = dead[:, :, :origin.shape[2]]
        elif dead.shape[2] < origin.shape[2]:
            pad = (0, origin.shape[2] - dead.shape[2], 0, 0)
            dead = F.pad(dead, pad, "constant", 0)
        return dead, origin

    if len(dead.shape) == 4 and len(origin.shape) == 2:
        dtype = dead.dtype
        dead = dead.to(torch.float32)
        dead = torch.mean(dead, (-2, -1))
        dead = dead.to(dtype)
        if dead.shape[1] == origin.shape[1]:
            return dead, origin
        if dead.shape[1] < origin.shape[1]:
            pad = (0, origin.shape[1] - dead.shape[1])
            dead = F.pad(dead, pad, "constant", 0)
        elif dead.shape[1] > origin.shape[1]:
            dead = dead[:, :origin.shape[1]]
        return dead, origin
    if len(dead.shape) == 4 and len(origin.shape) == 4:
        # print("dead.shape", dead.shape)
        # print("origin.shape", origin.shape)
        if dead.shape[2] > origin.shape[2]:
            dead = dead[:, :, :origin.shape[2], :]
            # print("aaaadead.shape", dead.shape)
            # print("aaaaorigin.shape", origin.shape)
        elif dead.shape[2] < origin.shape[2]:
            pad = (0, 0, 0, origin.shape[2] - dead.shape[2])
            dead = F.pad(dead, pad, "constant", 0)
        # print("dead.shape", dead.shape)
        # print("origin.shape", origin.shape)
        if dead.shape[3] > origin.shape[3]:
            dead = dead[:, :, :, :origin.shape[3]]
        elif dead.shape[3] < origin.shape[3]:
            pad = (origin.shape[3] - dead.shape[3], 0)
            dead = F.pad(dead, pad, "constant", 0)
        if dead.shape[1] < origin.shape[1]:
            pad = (0, 0, 0, 0, 0, origin.shape[1] - dead.shape[1])
            dead = F.pad(dead, pad, "constant", 0)
        elif dead.shape[1] > origin.shape[1]:
            dead = dead[:, :origin.shape[1], :, :]
        return dead, origin
    if len(dead.shape) == 2 and len(origin.shape) == 2:
        if dead.shape[1] > origin.shape[1]:
            dead = dead[:, :origin.shape[1]]
        else:
            pad = (0, origin.shape[1] - dead.shape[1])
            dead = F.pad(dead, pad, "constant", 0)
        return dead, origin
    if len(dead.shape) == 2 and len(origin.shape) == 4:
        dead = dead.unsqueeze(-1)
        dead = dead.unsqueeze(-1)
        if dead.shape[1] > origin.shape[1]:
            dead = dead[:, :origin.shape[1], :, :]
        elif dead.shape[1] < origin.shape[1]:
            pad = (0, 0, 0, 0, origin.shape[1] - dead.shape[1], 0)
            dead = F.pad(dead, pad, "constant", 0)
        if dead.shape[2] > origin.shape[2]:
            dead = dead[:, :, :origin.shape[2], :]
        elif dead.shape[2] < origin.shape[2]:
            pad = (0, 0, 0, origin.shape[2] - dead.shape[2])
            dead = F.pad(dead, pad, "constant", 0)
        if dead.shape[3] > origin.shape[3]:
            dead = dead[:, :, :, :origin.shape[3]]
        return dead, origin
    else:
        raise ValueError("handle_shape_PIOC_final error", dead.shape, origin.shape)
  
def handle_shape_strict(a, b):
    if len(a.shape) == 1:
        a = a.unsqueeze(-1)
    if len(b.shape) == 1:
        b = b.unsqueeze(-1)
    if len(a.shape) == 3:
        a = a[:, :, :1]
        a = a.squeeze(dim=-1)
    if len(b.shape) == 3:
        b = b[:, :, :1]
        # print("b.shape", b.shape)
        b = b.squeeze(dim=-1)
    if a.shape == b.shape:
        return a.to(device), b.to(device)
    if len(a.shape) == 2 and len(b.shape) == 2:
        if a.shape[1] < b.shape[1]:
            pad = (0, b.shape[1] - a.shape[1])
            a = F.pad(a, pad, "constant", 0)
            return a.to(device), b.to(device)
        else:
            pad = (0, a.shape[1] - b.shape[1])
            b = F.pad(b, pad, "constant", 0)
            return a.to(device), b.to(device)
    elif len(a.shape) == 4 and len(b.shape) == 4:
        if a.shape[1] < b.shape[1]:
            pad = (0, 0, 0, 0, 0, b.shape[1] - a.shape[1])
            a = F.pad(a, pad, "constant", 0)
        elif a.shape[1] > b.shape[1]:
            pad = (0, 0, 0, 0, 0, a.shape[1] - b.shape[1])
            b = F.pad(b, pad, "constant", 0)
        if a.shape[2] < b.shape[2]:
            pad = (0, 0, 0, b.shape[2] - a.shape[2])
            a = F.pad(a, pad, "constant", 0)
        elif a.shape[2] > b.shape[2]:
            pad = (0, 0, 0, a.shape[2] - b.shape[2])
            b = F.pad(b, pad, "constant", 0)
        if a.shape[3] < b.shape[3]:
            pad = (b.shape[3] - a.shape[3], 0, 0, 0)
            a = F.pad(a, pad, "constant", 0)
        elif a.shape[3] > b.shape[3]:
            pad = (a.shape[3] - b.shape[3], 0, 0, 0)
            b = F.pad(b, pad, "constant", 0)
        return a.to(device), b.to(device)
    elif len(a.shape) == 2:
        a = a.unsqueeze(-1)
        a = a.unsqueeze(-1)
        if a.shape[1] > b.shape[1]:
            a = a[:, :b.shape[1], :, :]
        elif a.shape[1] < b.shape[1]:
            b = b[:, :a.shape[1], :, :]
        if a.shape[2] < b.shape[2]:
            pad = (0, 0, 0, b.shape[2] - a.shape[2])
            a = F.pad(a, pad, "constant", 0)
        elif a.shape[2] > b.shape[2]:
            pad = (0, 0, 0, a.shape[2] - b.shape[2])
            b = F.pad(b, pad, "constant", 0)
        if a.shape[3] > b.shape[3]:
            pad = (a.shape[3] - b.shape[3], 0)
            b = F.pad(b, pad, "constant", 0)
        elif a.shape[3] < b.shape[3]:
            b = b[:, :, :, :a.shape[3]]
        return a.to(device), b.to(device)
    elif len(b.shape) == 2:
        b = b.unsqueeze(-1)
        b = b.unsqueeze(-1)
        if a.shape[1] > b.shape[1]:
            pad = (0, 0, 0, 0, 0, a.shape[1] - b.shape[1])
            b = F.pad(b, pad, "constant", 0)
        elif a.shape[1] < b.shape[1]:
            b = b[:, :a.shape[1], :, :]
        if a.shape[2] < b.shape[2]:
            b = b[:, :, :a.shape[2], :]
        elif a.shape[2] > b.shape[2]:
            pad = (0, 0, 0, a.shape[2] - b.shape[2])
            b = F.pad(b, pad, "constant", 0)
        if a.shape[3] > b.shape[3]:
            pad = (a.shape[3] - b.shape[3], 0)
            b = F.pad(b, pad, "constant", 0)
        elif a.shape[3] < b.shape[3]:
            b = b[:, :, :, :a.shape[3]]
        return a.to(device), b.to(device)
    elif len(a.shape) == 5 and len(b.shape) == 5:
        if a.shape[-1] > b.shape[-1]:
            pad = (0, a.shape[-1] - b.shape[-1])
            b = F.pad(b, pad, "constant", 0)
        elif a.shape[-1] < b.shape[-1]:
            b = b[:, :, :, :, :a.shape[-1]]
        if a.shape[-2] > b.shape[-2]:
            pad = (0, 0, 0, a.shape[-2] - b.shape[-2])
            b = F.pad(b, pad, "constant", 0)
        elif a.shape[-2] < b.shape[-2]:
            b = b[:, :, :, :a.shape[-2], :]
        if a.shape[2] > b.shape[2]:
            pad = (0, 0, 0, 0, 0, a.shape[2] - b.shape[2])
            b = F.pad(b, pad, "constant", 0)
        elif a.shape[2] < b.shape[2]:
            b = b[:, :, :a.shape[2], :, :]
        if a.shape[1] > b.shape[1]:
            pad = (0, 0, 0, 0, 0, 0, 0, a.shape[1] - b.shape[1])
            b = F.pad(b, pad, "constant", 0)
        elif a.shape[1] < b.shape[1]:
            b = b[:, :a.shape[1], :, :, :]
        return a.to(device), b.to(device)
    else:
        # print("handle_shape error", a.shape, b.shape)
        raise ValueError("handle_shape error", a.shape, b.shape)


def make_unsqueeze(in_tensor):
    if len(in_tensor.shape) == 2:
        in_tensor = in_tensor.unsqueeze(-1)
        in_tensor = in_tensor.unsqueeze(-1)
        return in_tensor
    if len(in_tensor.shape) == 4:
        return in_tensor
    if len(in_tensor.shape) == 3:
        in_tensor = in_tensor.unsqueeze(-1)
        return in_tensor
    if len(in_tensor.shape) == 1:
        in_tensor = in_tensor.unsqueeze(-1)
        in_tensor = in_tensor.unsqueeze(-1)
        in_tensor = in_tensor.unsqueeze(-1)
        return in_tensor
    print("len error length:", len(in_tensor.shape))
    raise ValueError("make_unsqueeze len error")



def make_reduce(input):
    if len(input.shape) > 2:
        dtype = input.dtype
        input = input.float()
        ans = torch.mean(input, (-2, -1), keepdim=True)
        ans = ans.to(dtype)
        return ans
    return input

def handle_format(outputs):
    if isinstance(outputs, (torch.Tensor,)):#情况1
        return outputs
    elif isinstance(outputs[0], (torch.Tensor,)): #情况4
        length = len(outputs)
        ans = torch.zeros(outputs[0].shape).to(device)
        for out in outputs:
            ans, tmp = handle_shape_strict(ans, out)
            ans = torch.add(ans, tmp)
        ans = torch.div(ans, length)
        return ans
    elif isinstance(outputs[0], tuple): #情况2
        big_length = len(outputs)
        ans = torch.zeros(outputs[0][0].shape).to(device)
        for out in outputs:
            temp = handle_format(out).to(device)
            ans, tmp = handle_shape_strict(ans, temp)
            ans = torch.add(ans, tmp)
        ans = torch.div(ans, big_length)
        return ans
    elif isinstance(outputs, tuple):#情况3
        big_length = len(outputs)
        ans = torch.zeros(outputs[0][0].shape).to(device)
        for out in outputs[0]:
            temp = handle_format(out).to(device)
            ans, tmp = handle_shape_strict(ans, temp)
            ans = torch.add(ans, tmp)
        ans = torch.div(ans, big_length)
        return ans
    else:
        raise NotImplementedError("handle_tuple not implemented for type: ", type(outputs[0]))


# def handle_format(outputs): #gpt改，不会创建副本
#     if isinstance(outputs, torch.Tensor):  # 情况1
#         return outputs
#     elif isinstance(outputs[0], torch.Tensor):  # 情况4
#         length = len(outputs)
#         ans = torch.zeros_like(outputs[0], device=device)  # 使用 zeros_like 来初始化，不创建新张量
#         for out in outputs:
#             ans, tmp = handle_shape_strict(ans, out)
#             ans.add_(tmp)  # 使用原位操作加法
#         ans.div_(length)  # 使用原位操作除法
#         return ans
#     elif isinstance(outputs[0], tuple):  # 情况2
#         big_length = len(outputs)
#         ans = torch.zeros_like(outputs[0][0], device=device)  # 使用 zeros_like 来初始化
#         for out in outputs:
#             temp = handle_format(out).to(device)
#             ans, tmp = handle_shape_strict(ans, temp)
#             ans.add_(tmp)  # 使用原位操作加法
#         ans.div_(big_length)  # 使用原位操作除法
#         return ans
#     elif isinstance(outputs, tuple):  # 情况3
#         big_length = len(outputs)
#         ans = torch.zeros_like(outputs[0][0], device=device)  # 使用 zeros_like 来初始化
#         for out in outputs[0]:
#             temp = handle_format(out).to(device)
#             ans, tmp = handle_shape_strict(ans, temp)
#             ans.add_(tmp)  # 使用原位操作加法
#         ans.div_(big_length)  # 使用原位操作除法
#         return ans
#     else:
#         raise NotImplementedError("handle_tuple not implemented for type: ", type(outputs[0]))
