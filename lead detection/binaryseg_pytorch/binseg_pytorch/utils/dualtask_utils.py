import torch
import torch.nn.functional as F
import numpy as np


def calc_pad_same(in_siz, out_siz, stride, ksize):
    """Calculate same padding width.
    Args:
    ksize: kernel size [I, J].
    Returns:
    pad_: Actual padding width.
    """
    return (out_siz - 1) * stride + ksize - in_siz


def conv2d_same(input, kernel, groups,bias=None,stride=1,padding=0,dilation=1):
    n, c, h, w = input.shape
    kout, ki_c_g, kh, kw = kernel.shape
    pw = calc_pad_same(w, w, 1, kw)
    ph = calc_pad_same(h, h, 1, kh)
    pw_l = pw // 2
    pw_r = pw - pw_l
    ph_t = ph // 2
    ph_b = ph - ph_t

    input_ = F.pad(input, (pw_l, pw_r, ph_t, ph_b))
    result = F.conv2d(input_, kernel, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    assert result.shape == input.shape
    return result


def gradient_central_diff(input, cuda):
    return input, input
    kernel = [[1, 0, -1]]
    kernel_t = 0.5 * torch.Tensor(kernel) * -1.  # pytorch implements correlation instead of conv
    if type(cuda) is int:
        if cuda != -1:
            kernel_t = kernel_t.cuda(device=cuda)
    else:
        if cuda is True:
            kernel_t = kernel_t.cuda()
    n, c, h, w = input.shape

    x = conv2d_same(input, kernel_t.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    y = conv2d_same(input, kernel_t.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    return x, y


def compute_single_sided_diferences(o_x, o_y, input):
    # n,c,h,w
    #input = input.clone()
    o_y[:, :, 0, :] = input[:, :, 1, :].clone() - input[:, :, 0, :].clone()
    o_x[:, :, :, 0] = input[:, :, :, 1].clone() - input[:, :, :, 0].clone()
    # --
    o_y[:, :, -1, :] = input[:, :, -1, :].clone() - input[:, :, -2, :].clone()
    o_x[:, :, :, -1] = input[:, :, :, -1].clone() - input[:, :, :, -2].clone()
    return o_x, o_y


def numerical_gradients_2d(input, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input.shape
    assert h > 1 and w > 1
    x, y = gradient_central_diff(input, cuda)
    return x, y


def convTri(input, r, cuda=False):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius
    :param cuda: move the kernel to gpu
    :return:
    """
    if (r <= 1):
        raise ValueError()
    n, c, h, w = input.shape
    return input
    f = list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))
    kernel = torch.Tensor([f]) / (r + 1) ** 2
    if type(cuda) is int:
        if cuda != -1:
            kernel = kernel.cuda(device=cuda)
    else:
        if cuda is True:
            kernel = kernel.cuda()

    # padding w
    input_ = F.pad(input, (1, 1, 0, 0), mode='replicate')
    input_ = F.pad(input_, (r, r, 0, 0), mode='reflect')
    input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
    input_ = torch.cat(input_, 3)
    t = input_

    # padding h
    input_ = F.pad(input_, (0, 0, 1, 1), mode='replicate')
    input_ = F.pad(input_, (0, 0, r, r), mode='reflect')
    input_ = [input_[:, :, :r, :], t, input_[:, :, -r:, :]]
    input_ = torch.cat(input_, 2)

    output = F.conv2d(input_,
                      kernel.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
                      padding=0, groups=c)
    output = F.conv2d(output,
                      kernel.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
                      padding=0, groups=c)
    return output


def compute_grad_mag(E, cuda=False):
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    mag = torch.sqrt(torch.mul(Ox,Ox) + torch.mul(Oy,Oy) + 1e-6)
    mag = mag / mag.max()  # 对梯度幅值进行归一化，将其范围缩放到[0, 1]

    return mag


def perturbate_input_(input, n_elements=200):
    N, C, H, W = input.shape
    assert N == 1
    c_ = np.random.random_integers(0, C - 1, n_elements)
    h_ = np.random.random_integers(0, H - 1, n_elements)
    w_ = np.random.random_integers(0, W - 1, n_elements)
    for c_idx in c_:
        for h_idx in h_:
            for w_idx in w_:
                input[0, c_idx, h_idx, w_idx] = 1
    return input

def _sample_gumbel(shape, eps=1e-10):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).cuda()
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    assert logits.dim() == 3
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, 1)


def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    y = torch.eye(num_classes).cuda()
    return y[labels].permute(0,3,1,2)


