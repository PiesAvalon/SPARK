import mindspore
import mindspore.ops as ops
from mindspore.common import dtype as mstype


def handle_shape_final(dead, origin):
    # print("dead.shape", dead.shape)
    # print("origin.shape", origin.shape)
    if dead.shape == origin.shape:
        return dead, origin
    if len(origin.shape) == 1:
        if len(dead.shape) == 2:
            dead = ops.slice(dead, (0, 0), (dead.shape[0], 1))
            dead = dead.squeeze(-1)
            if dead.shape[0] > origin.shape[0]:
                dead = ops.slice(dead, (0,), (origin.shape[0],))
            elif dead.shape[0] < origin.shape[0]:
                pad = (0, origin.shape[0] - dead.shape[0])
                dead = ops.pad(dead, pad, "constant", 0)
            return dead, origin
        if len(dead.shape) == 3:
            dead = ops.slice(dead, (0, 0, 0), (dead.shape[0], 1, 1))
            dead = dead.squeeze(-1)
            dead = dead.squeeze(-1)
            if dead.shape[0] > origin.shape[0]:
                dead = ops.slice(dead, (0,), (origin.shape[0],))
            elif dead.shape[0] < origin.shape[0]:
                pad = (0, origin.shape[0] - dead.shape[0])
                dead = ops.pad(dead, pad, "constant", 0)
            return dead, origin
        if len(dead.shape) == 4:
            dead = ops.slice(dead, (0, 0, 0, 0), (dead.shape[0], 1, 1, 1))
            dead = dead.squeeze(-1)
            dead = dead.squeeze(-1)
            dead = dead.squeeze(-1)
            if dead.shape[0] > origin.shape[0]:
                dead = ops.slice(dead, (0,), (origin.shape[0],))
            elif dead.shape[0] < origin.shape[0]:
                pad = (0, origin.shape[0] - dead.shape[0])
                dead = ops.pad(dead, pad, "constant", 0)
            return dead, origin
    if len(origin.shape) == 3 and len(dead.shape) == 4:
        dead = ops.slice(dead, (0, 0, 0, 0), (dead.shape[0], dead.shape[1], dead.shape[2], 1))
        dead = dead.squeeze(-1)
        if dead.shape[1] > origin.shape[1]:
            dead = ops.slice(dead, (0, 0, 0), (dead.shape[0], origin.shape[1], dead.shape[2]))
        elif dead.shape[1] < origin.shape[1]:
            pad = (0, 0, origin.shape[1] - dead.shape[1], 0)
            dead = ops.pad(dead, pad, "constant", 0)
        if dead.shape[2] > origin.shape[2]:
            dead = ops.slice(dead, (0, 0, 0), (dead.shape[0], dead.shape[1], origin.shape[2]))
        elif dead.shape[2] < origin.shape[2]:
            pad = (0, origin.shape[2] - dead.shape[2], 0, 0)
            dead = ops.pad(dead, pad, "constant", 0)
        return dead, origin

    if len(origin.shape) == 3 and len(dead.shape) == 2:
        dead = dead.unsqueeze(-1)
        if dead.shape[1] > origin.shape[1]:
            dead = ops.slice(dead, (0, 0, 0), (dead.shape[0], origin.shape[1], dead.shape[2]))
        elif dead.shape[1] < origin.shape[1]:
            pad = (0, 0, origin.shape[1] - dead.shape[1], 0)
            dead = ops.pad(dead, pad, "constant", 0)
        if dead.shape[2] > origin.shape[2]:
            dead = ops.slice(dead, (0, 0, 0), (dead.shape[0], dead.shape[1], origin.shape[2]))
        elif dead.shape[2] < origin.shape[2]:
            pad = (0, origin.shape[2] - dead.shape[2], 0, 0)
            dead = ops.pad(dead, pad, "constant", 0)
        return dead, origin

    if len(dead.shape) == 4 and len(origin.shape) == 2:
        dead = ops.ReduceMax(keep_dims=False)(dead, (-2, -1))
        # print("reduceddead.shape", dead.shape)
        # if dead.shape[1] == origin.shape[1]:
        #     return dead, origin
        if dead.shape[1] < origin.shape[1]:
            pad = (0, origin.shape[1] - dead.shape[1])
            dead = ops.pad(dead, pad, "constant", 0)
        elif dead.shape[1] > origin.shape[1]:
            dead = ops.slice(dead, (0, 0), (dead.shape[0], origin.shape[1]))
        if dead.shape[0] < origin.shape[0]:
            pad = (0, 0, 0, origin.shape[0] - dead.shape[0])
            dead = ops.pad(dead, pad, "constant", 0)
        elif dead.shape[0] > origin.shape[0]:
            dead = ops.slice(dead, (0, 0), (origin.shape[0], dead.shape[1]))
        # print("processed dead.shape", dead.shape)
        # print("processed origin.shape", origin.shape)
        return dead, origin
    if len(dead.shape) == 4 and len(origin.shape) == 4:
        if dead.shape[2] > origin.shape[2]:
            dead = ops.slice(dead, (0, 0, 0, 0), (dead.shape[0], dead.shape[1], origin.shape[2], dead.shape[3]))
        elif dead.shape[2] < origin.shape[2]:
            pad = (0, 0, 0, origin.shape[2] - dead.shape[2])
            dead = ops.pad(dead, pad, "constant", 0)
        if dead.shape[3] > origin.shape[3]:
            dead = ops.slice(dead, (0, 0, 0, 0), (dead.shape[0], dead.shape[1], dead.shape[2], origin.shape[3]))
        elif dead.shape[3] < origin.shape[3]:
            pad = (origin.shape[3] - dead.shape[3], 0)
            dead = ops.pad(dead, pad, "constant", 0)
        if dead.shape[1] < origin.shape[1]:
            pad = (0, 0, 0, 0, 0, origin.shape[1] - dead.shape[1])
            dead = ops.pad(dead, pad, "constant", 0)
        elif dead.shape[1] > origin.shape[1]:
            dead = ops.slice(dead, (0, 0, 0, 0), (dead.shape[0], origin.shape[1], dead.shape[2], dead.shape[3]))
        return dead, origin
    if len(dead.shape) == 2 and len(origin.shape) == 2:
        if dead.shape[1] > origin.shape[1]:
            dead = ops.slice(dead, (0, 0), (dead.shape[0], origin.shape[1]))
        else:
            pad = (0, origin.shape[1] - dead.shape[1])
            dead = ops.pad(dead, pad, "constant", 0)
        if dead.shape[0] > origin.shape[0]:
            dead = ops.slice(dead, (0, 0), (origin.shape[0], dead.shape[1]))
        else:
            pad = (0, 0, 0, origin.shape[0] - dead.shape[0])
            dead = ops.pad(dead, pad, "constant", 0)
        return dead, origin
    if len(dead.shape) == 2 and len(origin.shape) == 4:
        dead = dead.unsqueeze(-1)
        dead = dead.unsqueeze(-1)
        if dead.shape[1] > origin.shape[1]:
            dead = ops.slice(dead, (0, 0, 0, 0), (dead.shape[0], origin.shape[1], dead.shape[2], dead.shape[3]))
        elif dead.shape[1] < origin.shape[1]:
            pad = (0, 0, 0, 0, origin.shape[1] - dead.shape[1], 0)
            dead = ops.pad(dead, pad, "constant", 0)
        if dead.shape[2] > origin.shape[2]:
            dead = ops.slice(dead, (0, 0, 0, 0), (dead.shape[0], dead.shape[1], origin.shape[2], dead.shape[3]))
        elif dead.shape[2] < origin.shape[2]:
            pad = (0, 0, 0, origin.shape[2] - dead.shape[2])
            dead = ops.pad(dead, pad, "constant", 0)
        if dead.shape[3] > origin.shape[3]:
            dead = ops.slice(dead, (0, 0, 0, 0), (dead.shape[0], dead.shape[1], dead.shape[2], origin.shape[3]))
        if dead.shape[0] > origin.shape[0]:
            dead = ops.slice(dead, (0, 0, 0, 0), (origin.shape[0], dead.shape[1], dead.shape[2], dead.shape[3]))
        elif dead.shape[0] < origin.shape[0]:
            pad = (0, origin.shape[0] - dead.shape[0], 0, 0)
            dead = ops.pad(dead, pad, "constant", 0)
        return dead, origin
    else:
        raise ValueError("handle_shape_PIOC_final error", dead.shape, origin.shape)


def handle_shape_loose(a, b):
    if len(a.shape) == 4 and len(b.shape) == 4:
        if a.shape[1] == b.shape[1]:
            return a, b
        if a.shape[1] < b.shape[1]:
            b = ops.slice(b, (0, 0, 0, 0), (b.shape[0], a.shape[1], b.shape[2], b.shape[3]))
        elif a.shape[1] > b.shape[1]:
            a = ops.slice(a, (0, 0, 0, 0), (a.shape[0], b.shape[1], a.shape[2], a.shape[3]))
        return a, b
    elif len(a.shape) == 4 and len(b.shape) == 2:
        b = b.unsqueeze(-1)
        b = b.unsqueeze(-1)
        if a.shape[1] == b.shape[1]:
            return a, b
        if a.shape[1] < b.shape[1]:
            b = ops.slice(b, (0, 0, 0, 0), (b.shape[0], a.shape[1], b.shape[2], b.shape[3]))
            return a, b
        if a.shape[1] > b.shape[1]:
            pad = (0, 0, 0, 0, 0, a.shape[1] - b.shape[1])
            b = ops.pad(b, pad, "constant", 0)
            return a, b
    else:
        raise ValueError("handle_shape_loose error")


def handle_shape_strict(a, b):
    # print("a.shape", a.shape)
    # print("b.shape", b.shape)
    if len(a.shape) == 1:
        a = a.unsqueeze(-1)
    if len(b.shape) == 1:
        b = b.unsqueeze(-1)
    if len(a.shape) == 3:
        a = ops.slice(a, (0, 0, 0), (a.shape[0], a.shape[1], 1))
        a = ops.squeeze(a, axis=-1)
    if len(b.shape) == 3:
        b = ops.slice(b, (0, 0, 0), (b.shape[0], b.shape[1], 1))
        b = ops.squeeze(b, axis=-1)
    # print("here")
    if a.shape == b.shape:
        # print("here1")
        return a, b
    # print("here2")
    if len(a.shape) == 2 and len(b.shape) == 2:
        if a.shape[1] < b.shape[1]:
            pad = (0, b.shape[1] - a.shape[1])
            a = ops.pad(a, pad, "constant", 0)
        else:
            pad = (0, a.shape[1] - b.shape[1])
            b = ops.pad(b, pad, "constant", 0)
        if a.shape[0] < b.shape[0]:
            pad = (0, 0, 0, b.shape[0] - a.shape[0])
            a = ops.pad(a, pad, "constant", 0)
        else:
            pad = (0, 0, 0, a.shape[0] - b.shape[0])
            b = ops.pad(b, pad, "constant", 0)
        return a, b
    elif len(a.shape) == 4 and len(b.shape) == 4:
        if a.shape[0] > b.shape[0]:
            a = ops.slice(a, (0, 0, 0, 0), (b.shape[0], a.shape[1], a.shape[2], a.shape[3]))
        elif a.shape[0] < b.shape[0]:
            b = ops.slice(b, (0, 0, 0, 0), (a.shape[0], b.shape[1], b.shape[2], b.shape[3]))
        if a.shape[1] < b.shape[1]:
            pad = (0, 0, 0, 0, 0, b.shape[1] - a.shape[1])
            a = ops.pad(a, pad, "constant", 0)
        elif a.shape[1] > b.shape[1]:
            pad = (0, 0, 0, 0, 0, a.shape[1] - b.shape[1])
            b = ops.pad(b, pad, "constant", 0)
        if a.shape[2] < b.shape[2]:
            pad = (0, 0, 0, b.shape[2] - a.shape[2])
            a = ops.pad(a, pad, "constant", 0)
        elif a.shape[2] > b.shape[2]:
            pad = (0, 0, 0, a.shape[2] - b.shape[2])
            b = ops.pad(b, pad, "constant", 0)
        if a.shape[3] < b.shape[3]:
            pad = (b.shape[3] - a.shape[3], 0, 0, 0)
            a = ops.pad(a, pad, "constant", 0)
        elif a.shape[3] > b.shape[3]:
            pad = (a.shape[3] - b.shape[3], 0, 0, 0)
            b = ops.pad(b, pad, "constant", 0)
        return a, b
    elif len(a.shape) == 2:
        a = a.unsqueeze(-1)
        a = a.unsqueeze(-1)
        if a.shape[0] > b.shape[0]:
            a = ops.slice(a, (0, 0, 0, 0), (b.shape[0], a.shape[1], a.shape[2], a.shape[3]))
        elif a.shape[0] < b.shape[0]:
            b = ops.slice(b, (0, 0, 0, 0), (a.shape[0], b.shape[1], b.shape[2], b.shape[3]))
        if a.shape[1] > b.shape[1]:
            a = ops.slice(a, (0, 0, 0, 0), (a.shape[0], b.shape[1], a.shape[2], a.shape[3]))
        elif a.shape[1] < b.shape[1]:
            b = ops.slice(b, (0, 0, 0, 0), (b.shape[0], a.shape[1], b.shape[2], b.shape[3]))
        if a.shape[2] < b.shape[2]:
            pad = (0, 0, 0, b.shape[2] - a.shape[2])
            a = ops.pad(a, pad, "constant", 0)
        elif a.shape[2] > b.shape[2]:
            pad = (0, 0, 0, a.shape[2] - b.shape[2])
            b = ops.pad(b, pad, "constant", 0)
        if a.shape[3] > b.shape[3]:
            pad = (a.shape[3] - b.shape[3], 0)
            b = ops.pad(b, pad, "constant", 0)
        elif a.shape[3] < b.shape[3]:
            b = ops.slice(b, (0, 0, 0, 0), (b.shape[0], b.shape[1], b.shape[2], a.shape[3]))
        return a, b
    elif len(b.shape) == 2:
        b = b.unsqueeze(-1)
        b = b.unsqueeze(-1)
        if a.shape[0] > b.shape[0]:
            a = ops.slice(a, (0, 0, 0, 0), (b.shape[0], a.shape[1], a.shape[2], a.shape[3]))
        elif a.shape[0] < b.shape[0]:
            b = ops.slice(b, (0, 0, 0, 0), (a.shape[0], b.shape[1], b.shape[2], b.shape[3]))
        if a.shape[1] > b.shape[1]:
            pad = (0, 0, 0, 0, 0, a.shape[1] - b.shape[1])
            b = ops.pad(b, pad, "constant", 0)
        elif a.shape[1] < b.shape[1]:
            b = ops.slice(b, (0, 0, 0, 0), (b.shape[0], a.shape[1], b.shape[2], b.shape[3]))
        if a.shape[2] < b.shape[2]:
            b = ops.slice(b, (0, 0, 0, 0), (b.shape[0], b.shape[1], a.shape[2], a.shape[3]))
        elif a.shape[2] > b.shape[2]:
            pad = (0, 0, 0, a.shape[2] - b.shape[2])
            b = ops.pad(b, pad, "constant", 0)
        if a.shape[3] > b.shape[3]:
            pad = (a.shape[3] - b.shape[3], 0)
            b = ops.pad(b, pad, "constant", 0)
        elif a.shape[3] < b.shape[3]:
            b = ops.slice(b, (0, 0, 0, 0), (b.shape[0], b.shape[1], b.shape[2], a.shape[3]))
        return a, b
    elif len(a.shape) == 5 and len(b.shape) == 5:
        if a.shape[-1] > b.shape[-1]:
            pad = (0, a.shape[-1] - b.shape[-1])
            b = ops.pad(b, pad, "constant", 0)
        elif a.shape[-1] < b.shape[-1]:
            b = ops.slice(b, (0, 0, 0, 0, 0), (b.shape[0], b.shape[1], b.shape[2], b.shape[3], a.shape[-1]))
        if a.shape[-2] > b.shape[-2]:
            pad = (0, 0, 0, a.shape[-2] - b.shape[-2])
            b = ops.pad(b, pad, "constant", 0)
        elif a.shape[-2] < b.shape[-2]:
            b = ops.slice(b, (0, 0, 0, 0, 0), (b.shape[0], b.shape[1], b.shape[2], a.shape[-2], b.shape[-1]))
        if a.shape[2] > b.shape[2]:
            pad = (0, 0, 0, 0, 0, a.shape[2] - b.shape[2])
            b = ops.pad(b, pad, "constant", 0)
        elif a.shape[2] < b.shape[2]:
            b = ops.slice(b, (0, 0, 0, 0, 0), (b.shape[0], b.shape[1], a.shape[2], b.shape[3], b.shape[4]))
        if a.shape[1] > b.shape[1]:
            pad = (0, 0, 0, 0, 0, 0, 0, a.shape[1] - b.shape[1])
            b = ops.pad(b, pad, "constant", 0)
        elif a.shape[1] < b.shape[1]:
            b = ops.slice(b, (0, 0, 0, 0, 0), (b.shape[0], a.shape[1], b.shape[2], b.shape[3], b.shape[4]))
        return a, b
    else:
        print("handle_shape error", a.shape, b.shape)
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
        input = ops.cast(input, mstype.float32)
        ans = ops.ReduceMean(keep_dims=True)(input, (-2, -1))
        ans = ops.cast(ans, dtype)
        return ans
    return input



def handle_format(outputs):
    if isinstance(outputs, (mindspore.Tensor, mindspore.common._stub_tensor.StubTensor)):
        return outputs
    elif isinstance(outputs[0], (mindspore.Tensor, mindspore.common._stub_tensor.StubTensor)):
        length = len(outputs)
        ans = mindspore.numpy.zeros(outputs[0].shape, mindspore.float32)
        for out in outputs:
            # print("adding")
            # print("ans.shape: ", ans.shape)
            # print("out.shape: ", out.shape)
            ans, tmp = handle_shape_strict(ans, out)
            ans = ops.add(ans, tmp)
            # print("ans.shape: ", ans.shape)
            # print("added")
        ans = ops.div(ans, length)
        return ans
    elif isinstance(outputs[0], tuple):
        big_length = len(outputs)
        print(type(outputs[0][0]))
        ans = mindspore.numpy.zeros(outputs[0][0].shape, mindspore.float32)
        for out in outputs:
            # print("ans.shape: ", ans.shape)
            # print("out.shape: ", out[0].shape)
            temp = handle_format(out)
            # print("temp.shape: ", temp.shape)
            ans, tmp = handle_shape_strict(ans, temp)
            # print("ans.shape: ", ans.shape, "tmp.shape: ", tmp.shape)
            ans = ops.add(ans, tmp)
            # print("fucker", ans.shape)
        ans = ops.div(ans, big_length)
        return ans
    elif isinstance(outputs, tuple):
        big_length = len(outputs)
        ans = mindspore.numpy.zeros(outputs[0][0].shape, mindspore.float32)
        for out in outputs[0]:
            # print("ans.shape: ", ans.shape)
            # print("out.shape: ", out[0].shape)
            temp = handle_format(out)
            # print("temp.shape: ", temp.shape)
            ans, tmp = handle_shape_strict(ans, temp)
            # print("ans.shape: ", ans.shape, "tmp.shape: ", tmp.shape)
            ans = ops.add(ans, tmp)
            # print("fucker", ans.shape)
        ans = ops.div(ans, big_length)
        return ans
    else:
        raise NotImplementedError("handle_format not implemented for type: ", type(outputs[0]))
