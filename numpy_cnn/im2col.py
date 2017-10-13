import numpy as np

def get_indices(in_shape, filter_shape, stride_shape, padding_shape):
    # c [f_H*f_W*in*D,1]
    # i [f_H*f_W*in*D, out_H*out_W]
    # j [f_H*f_W*in*D, out_H*out_W]
    BS, in_D, in_H, in_W = in_shape
    f_H, f_W = filter_shape
    s_H, s_W = stride_shape
    p_H, p_W = padding_shape
    # assert (in_H + 2 * p_H - f_H) % s_H == 0  # False - > warning
    # assert (in_W + 2 * p_W - f_W) % s_W == 0
    out_H = int((in_H + p_H - f_H) / s_H) + 1
    out_W = int((in_W + p_W - f_W) / s_W) + 1

    c = np.repeat(np.arange(in_D), f_H*f_W).reshape(-1,1) # [f_H*f_W*in*D, 1]

    i_col = np.repeat(np.arange(f_H), f_W)
    i_col = np.tile(i_col, in_D).reshape(-1,1) # [f_H*f_W*in*D, 1]
    i_row = s_H * np.repeat(np.arange(out_H), out_W).reshape(1,-1)
    i = i_col + i_row  # [f_H*f_W*in*D, out_H*out_W]

    j_col = np.tile(np.arange(f_W), f_H)
    j_col = np.tile(j_col, in_D).reshape(-1,1)
    j_row = s_W * np.tile(np.arange(out_W), out_H).reshape(1,-1)
    j = j_col + j_row

    return c, i, j

def im2col(in_im, filter_shape, stride_shape, padding_shape):
    # input : [BS, in_D, in_H, in_W]
    # output : [f_H*f_W*in_D, out_H*out_W*BS]
    in_D= in_im.shape[1]
    f_H, f_W = filter_shape
    p_H, p_W = padding_shape

    p_H_l = int(p_H/2)
    p_W_l = int(p_W / 2)
    in_padded = np.pad(in_im, ((0,0), (0,0), (p_H_l,p_H-p_H_l), (p_W_l, p_W-p_W_l)), mode='constant')

    c, i, j = get_indices(in_im.shape, filter_shape,  stride_shape, padding_shape)
    in_col = in_padded[:, c, i, j] # [BS, f_H*f_W*in_D, out_H*out_W]
    in_col = in_col.transpose( (1,2,0) ).reshape( (f_H*f_W*in_D,-1) )
    return in_col

def col2im(dout_col, input_shape, filter_shape, stride_shape, padding_shape):
    # dout_col :  [f_H*f_W*in_D, out_H *out_W * BS]
    # din : [BS, in_D, in_H, in_W]
    BS, in_D, in_H, in_W = input_shape
    f_H, f_W = filter_shape
    p_H, p_W = padding_shape
    din_padding = np.zeros(shape=[BS, in_D, in_H+p_H, in_W+p_W])

    c, i, j = get_indices(input_shape, filter_shape, stride_shape, padding_shape)
    dout_reshape = dout_col.reshape(f_H*f_W*in_D, -1, BS).transpose(2,0,1)
    np.add.at(din_padding, (slice(None), c, i, j), dout_reshape)
    if p_H != 0:
        din_padding = din_padding[:,:,int(p_H/2):-(p_H-int(p_H/2)),:]
    if p_W != 0:
        din_padding = din_padding[:,:,:,int(p_W/2):-(p_W-int(p_W/2))]

    return din_padding