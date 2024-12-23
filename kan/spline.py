import paddle


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    """
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    """
    """# x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                    grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]"""
    x = x.unsqueeze(axis=2)
    grid = grid.unsqueeze(axis=0)
    # print(x.shape, grid.shape)
    if k == 0:
        # print(11)
        value = ((x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])).astype(x.dtype)
    else:
        # print(22)
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)
        # print(33)
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :,
            :-(k + 1)]) * B_km1[:, :, :-1] + (grid[:, :, k + 1:] - x) / (grid
            [:, :, k + 1:] - grid[:, :, 1:-k]) * B_km1[:, :, 1:]
        # print(44)
    # print(99)
    # print(value.shape, value.dtype)
    # print(999)
    # print(paddle.nan_to_num(x=value.contiguous()))
    # print(9999)
    # value = paddle.nan_to_num(x=value.contiguous())
    return value


def coef2curve(x_eval, grid, coef, k, device='cpu'):
    """
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    """
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = paddle.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.place))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, lamb=1e-08):
    """
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    torch.Size([5, 13])
    """
    """
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    # coef = torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]"""
    # print(1)
    batch = tuple(x_eval.shape)[0]
    # print(2)
    in_dim = tuple(x_eval.shape)[1]
    # print(3)
    out_dim = tuple(y_eval.shape)[2]
    # print(4)
    n_coef = tuple(grid.shape)[1] - k - 1
    # print(5)
    mat = B_batch(x_eval, grid, k)
    # print(6)
    mat = mat.transpose(perm=[1, 0, 2])[:, None, :, :].expand(shape=[in_dim,
        out_dim, batch, n_coef])
    # print(7)
    y_eval = y_eval.transpose(perm=[1, 2, 0]).unsqueeze(axis=3)
    # print(8)
    device = mat.place
    XtX = paddle.einsum('ijmn,ijnp->ijmp', mat.transpose(perm=[0, 1, 3, 2]),
        mat)
    # print(9)
    Xty = paddle.einsum('ijmn,ijnp->ijmp', mat.transpose(perm=[0, 1, 3, 2]),
        y_eval)
    # print(10)
    n1, n2, n = tuple(XtX.shape)[0], tuple(XtX.shape)[1], tuple(XtX.shape)[2]
    # print(11)
    identity = paddle.eye(num_rows=n, num_columns=n)[None, None, :, :].expand(
        shape=[n1, n2, n, n])
    # print(12)
    A = XtX + lamb * identity
    # print(13)
    B = Xty
    # print(14)
    coef = (A.pinv() @ B)[:, :, :, 0]
    # print(15)
    return coef


def extend_grid(grid, k_extend=0):
    # print(1)
    # print(grid.shape)
    h = (grid[:, -2:-1] - grid[:, 0:1]) / (tuple(grid.shape)[1] - 1)
    # print(2)
    for i in range(k_extend):
        # print("i", i)
        grid = paddle.concat(x=[grid[:, 0:1] - h, grid], axis=1)
        grid = paddle.concat(x=[grid, grid[:, -2:-1] + h], axis=1)
    return grid
