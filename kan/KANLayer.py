import paddle
import numpy as np
from .spline import *
from .utils import sparse_mask


class KANLayer(paddle.nn.Layer):
    """
    KANLayer class


    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device

    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1,
        scale_base=1.0, scale_sp=1.0, base_fun=paddle.nn.Silu(), grid_eps=
        0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True,
        save_plot_data=True, device='cpu', sparse_init=False):
        """'
        initialize a KANLayer

        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device

        Returns:
        --------
            self

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        (3, 5)
        """
        super(KANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        grid = paddle.linspace(start=grid_range[0], stop=grid_range[1], num
            =num + 1)[None, :].expand(shape=[self.in_dim, num + 1])
        grid = extend_grid(grid, k_extend=k)
        out_10 = paddle.base.framework.EagerParamBase.from_tensor(tensor=grid)
        out_10.stop_gradient = not False
        self.grid = out_10
        noises = (paddle.rand(shape=[self.num + 1, self.in_dim, self.
            out_dim]) - 1 / 2) * noise_scale / num
        self.coef = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =curve2coef(self.grid[:, k:-k].transpose(perm=[1, 0]), noises,
            self.grid, k))
        if sparse_init:
            mask = sparse_mask(in_dim, out_dim)
        else:
            mask = 1.0
        out_11 = paddle.base.framework.EagerParamBase.from_tensor(tensor=
            paddle.ones(shape=[in_dim, out_dim]) * scale_base * mask)
        out_11.stop_gradient = not sb_trainable
        self.scale_base = out_11
        out_12 = paddle.base.framework.EagerParamBase.from_tensor(tensor=
            paddle.ones(shape=[in_dim, out_dim]) * scale_sp * mask)
        out_12.stop_gradient = not sp_trainable
        self.scale_sp = out_12
        self.base_fun = base_fun
        out_13 = paddle.base.framework.EagerParamBase.from_tensor(tensor=
            paddle.ones(shape=[in_dim, out_dim]))
        out_13.stop_gradient = not False
        self.mask = out_13
        self.grid_eps = grid_eps

    def forward(self, x):
        """
        KANLayer forward given input x

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        (torch.Size([100, 5]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]))
        """
        batch = tuple(x.shape)[0]
        preacts = x[:, None, :].clone().expand(shape=[batch, self.out_dim,
            self.in_dim])
        base = self.base_fun(x)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
        postspline = y.clone().transpose(perm=[0, 2, 1])
        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[
            None, :, :] * y
        y = self.mask[None, :, :] * y
        postacts = y.clone().transpose(perm=[0, 2, 1])
        y = paddle.sum(x=y, axis=1)
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x, mode='sample'):
        """
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        """
        batch = tuple(x.shape)[0]
        x_pos = (paddle.sort(x=x, axis=0), paddle.argsort(x=x, axis=0))[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = tuple(self.grid.shape)[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)
                ] + [-1+x_pos.shape[0]]
            grid_adaptive = x_pos[ids, :].transpose(perm=[1, 0]).contiguous()

            h = (grid_adaptive[:, (-2+grid_adaptive.shape[1]):(-1+grid_adaptive.shape[1])] - grid_adaptive[:, 0:1]) / num_interval
            grid_uniform = grid_adaptive[:, 0:1] + h * paddle.arange(end=
                num_interval + 1, dtype=x.dtype)[None, :].to(device=x.place
            )
            # print(f"grid_adaptive.shape = {grid_adaptive.shape}")
            # grid_uniform = grid_adaptive.index_select(paddle.to_tensor([0]), axis=1) + h * paddle.arange(end=
            #     num_interval + 1, dtype=x.dtype)[None, :].to(device=x.place
            # )

            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps
                ) * grid_adaptive
            return grid

        # print("get_grid...")

        grid = get_grid(num_interval)
        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.transpose(perm=[1, 0])
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)
        # print("get_grid 2...")

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        """
        update grid from a parent KANLayer & samples

        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        """
        batch = tuple(x.shape)[0]
        x_pos = (paddle.sort(x=x, axis=0), paddle.argsort(x=x, axis=0))[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = tuple(self.grid.shape)[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)
                ] + [-1]
            grid_adaptive = x_pos[ids, :].transpose(perm=[1, 0])
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * paddle.arange(end=
                num_interval + 1)[None, :].to(x.place)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps
                ) * grid_adaptive
            return grid
        grid = get_grid(num_interval)
        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.transpose(perm=[1, 0])
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def get_subset(self, in_id, out_id):
        """
        get a smaller KANLayer from a larger KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : KANLayer

        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        """
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=
            self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:, out_id]
        spb.scale_base.data = self.scale_base[in_id][:, out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:, out_id]
        spb.mask.data = self.mask[in_id][:, out_id]
        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb

    def swap(self, i1, i2, mode='in'):
        with paddle.no_grad():

            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:, i1], data[:, i2] = data[:, i2].clone(), data[:, i1
                        ].clone()
            if mode == 'in':
                swap_(self.grid.detach(), i1, i2, mode='in')
            swap_(self.coef.detach(), i1, i2, mode=mode)
            swap_(self.scale_base.detach(), i1, i2, mode=mode)
            swap_(self.scale_sp.detach(), i1, i2, mode=mode)
            swap_(self.mask.detach(), i1, i2, mode=mode)
