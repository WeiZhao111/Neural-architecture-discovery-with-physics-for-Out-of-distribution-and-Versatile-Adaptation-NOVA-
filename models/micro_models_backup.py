from models.micro_operations import *
from util.utils import drop_path
from util.prim_ops_set import *
from .fcn import FCNHead
from .base import BaseNet
from util.functional import *
from torch.nn.functional import interpolate, softmax
import time



DEFAULT_PADDINGS = {
    'none': 1,
    'skip_connect': 1,
    'avg_pool_3x3': 1,
    'max_pool_3x3': 1,
    'sep_conv_3x3': 1,
    'sep_conv_5x5': 2,
    'sep_conv_7x7': 3,
    'dil_conv_3x3': 2,
    'dil_conv_5x5': 4,
    'conv_7x1_1x7': 3,
}


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, SE=False):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        self.se_layer = None

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

        if SE:
            self.se_layer = SELayer(channel=self.multiplier * C)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)  # didn't consider not linking s0 and s1
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]

        if self.se_layer is None:
            return torch.cat([states[i] for i in self._concat], dim=1)
        else:
            return self.se_layer(torch.cat([states[i] for i in self._concat], dim=1))



class BuildCell(nn.Module):
    """Build a cell from genotype"""

    def __init__(self, genotype, c_prev_prev, c_prev, c, cell_type, dropout_prob=0):
        super(BuildCell, self).__init__()
        self.cell_type = cell_type

        # self.genotype = genotype
        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, stride=2, ops_order='act_weight_norm')
        else:
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c_prev, c, kernel_size=1, ops_order='act_weight_norm')

        if cell_type == 'up':
            op_names, idx = zip(*genotype.up)
            concat = genotype.up_concat
        else:
            op_names, idx = zip(*genotype.down)
            concat = genotype.down_concat
        self.dropout_prob = dropout_prob
        self._compile(c, op_names, idx, concat)

    def _compile(self, c, op_names, idx, concat):
        assert len(op_names) == len(idx)
        self._num_meta_node = len(op_names) // 2
        self._concat = concat
        self._multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, idx):
            op = OPS[name](c, None, affine=True, dp=self.dropout_prob)
            self._ops += [op]
        self._indices = idx

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._num_meta_node):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]

            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]

            h1 = op1(h1)
            h2 = op2(h2)

            # the size of h1 and h2 may be different, so we need interpolate
            if h1.size() != h2.size() :
                _, _, height1, width1 = h1.size()
                _, _, height2, width2 = h2.size()
                if height1 > height2 or width1 > width2:
                    h2 = interpolate(h2, (height1, width1))
                else:
                    h1 = interpolate(h1, (height2, width2))
            elif self.cell_type == 'up' and self._indices[2*i] == 1 and self._indices[2*i+1] == 1:  # both use node 1 as parent node
                 _, _, height1, width1 = h1.size()
                 h1 = interpolate(h1, (height1 + 1, width1 + 1))
                 h2 = interpolate(h2, (height1 + 1, width1 + 1))

            s = h1+h2
            states += [s]
            # print(self.genotype)
        return torch.cat([states[i] for i in self._concat], dim=1)



class NasUnet(BaseNet):
    """Construct a network"""

    def __init__(self, nclass, in_channels, backbone=None, aux=False,
                 c=48, depth=5, dropout_prob=0,
                 genotype=None, double_down_channel=False):

        super(NasUnet, self).__init__(nclass, aux, backbone, norm_layer=nn.GroupNorm)
        self._depth = depth
        self._double_down_channel = double_down_channel
        stem_multiplier = 4
        c_curr = stem_multiplier * c

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, c

        # the stem need a complicate mode
        self.stem0 = ConvOps(in_channels, c_prev_prev, kernel_size=1, ops_order='weight_norm')
        self.stem1 = ConvOps(in_channels, c_prev, kernel_size=3, stride=2, ops_order='weight_norm')

        assert depth >= 2 , 'depth must >= 2'

        self.down_cells = nn.ModuleList()
        self.up_cells = nn.ModuleList()
        down_cs_nfilters = []

        # create the encoder pathway and add to a list
        down_cs_nfilters += [c_prev]
        down_cs_nfilters += [c_prev_prev]
        for i in range(depth):
            c_curr = 2 * c_curr if self._double_down_channel else c_curr  # double the number of filters
            down_cell = BuildCell(genotype, c_prev_prev, c_prev, c_curr, cell_type='down', dropout_prob=dropout_prob)
            self.down_cells += [down_cell]
            c_prev_prev, c_prev = c_prev, down_cell._multiplier*c_curr
            down_cs_nfilters += [c_prev]

        # create the decoder pathway and add to a list
        for i in range(depth+1):
            # c_curr = c_curr // 2 if self._double_down_channel else c_curr  # halve the number of filters
            c_prev_prev = down_cs_nfilters[-(i + 2)] # the horizontal prev_prev input channel
            up_cell = BuildCell(genotype, c_prev_prev, c_prev, c_curr, cell_type='up',  dropout_prob=dropout_prob)
            self.up_cells += [up_cell]
            c_prev = up_cell._multiplier*c_curr
            c_curr = c_curr // 2 if self._double_down_channel else c_curr  # halve the number of filters

        self.nas_unet_head = ConvOps(c_prev, nclass, kernel_size=3, ops_order='weight')

        if self.aux:
            self.auxlayer = FCNHead(c_prev, nclass, nn.BatchNorm2d)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, nclass)

    # def forward(self, x, label):
    def forward(self, x):
        _, _, h, w = x.size()
        s0, s1 = self.stem0(x), self.stem1(x)
        down_cs = []

        # encoder pathway
        down_cs.append(s0)
        down_cs.append(s1)
        for i, cell in enumerate(self.down_cells):
            # Sharing a global N*M weights matrix
            # where M : normal + down
            s0, s1 = s1, cell(s0, s1)
            down_cs.append(s1)

        # decoder pathway
        for i, cell in enumerate(self.up_cells):
            # Sharing a global N*M weights matrix
            # where M : normal + up
            s0 = down_cs[-(i+2)] # horizon input
            s1 = cell(s0, s1)

        # outputs = self.nas_unet_head(s1)

        # outputs = CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC(s1, x, self.nas_unet_head)

        outputs = CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC_test1(s1, x, self.nas_unet_head)

        # outputs = CNN_DNN_Pseudo_Inverse_2D_DR(s1, x, label, self.nas_unet_head)

        # outputs = softmax(outputs, dim=1)  # chase_db1  (don't need for others)

        # out = self.global_pooling(s1)
        # outputs = self.classifier(out.view(out.size(0), -1))



        # outputs = []
        # outputs.append(output)

        # if self.aux: # use aux header
        #     auxout = self.auxlayer(s1)
        #     auxout = interpolate(auxout, (h,w), **self._up_kwargs)
        #     outputs.append(auxout)

        return outputs
    



def CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC(inputs, inputs0, conv2d_head):

    inputs_conv2d = inputs

    # start = time.time()
    # kernel size: 3*3, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (1, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 1, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 1, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, 1, 0, 0, 0, 0))

    # computational boundary
    x_l, x_u, y_l, y_u = -1, 1, -1, 1
    # x_l, x_u, y_l, y_u = 0, 1, 0, 1
    ext = [x_l, x_u, y_l, y_u]
    t_l, t_u = 0, 5

    # dx, dy, dt
    n_x, n_y, n_t = 128, 128, 101
    # dx = (x_u - x_l) / (n_x - 1) #/ 2
    # dy = (y_u - y_l) / (n_y - 1) #/ 2
    # dt = (t_u - t_l) / (n_t - 1) #/ 2
    dx = dy = 0.015625
    dt = 0.05

    n_padding = int((inputs.shape[2] - 2 - n_x)/2)
    n_kernel_size = 3

    x = np.linspace(x_l, x_u, n_x)
    y = np.linspace(y_l, y_u, n_y)
    x_grid, y_grid = np.meshgrid(x, y)

    # add BC indicator
    _boundary_left = (x_grid == x_l) & (y_grid != y_l) & (y_grid != y_u)
    _boundary_bottom = (y_grid == y_l) & (x_grid != x_l) & (x_grid != x_u)
    _boundary_right = (x_grid == x_u) & (y_grid != y_l) & (y_grid != y_u)
    _boundary_top = (y_grid == y_u) & (x_grid != x_l) & (x_grid != x_u)
    _boundary_left_bottom = (x_grid == x_l) & (y_grid == y_l)
    _boundary_right_bottom = (x_grid == x_u) & (y_grid == y_l)
    _boundary_right_top = (x_grid == x_u) & (y_grid == y_u)
    _boundary_left_top = (x_grid == x_l) & (y_grid == y_u)

    x_ind_bc = np.zeros((n_y, n_x))
    x_ind_bc[_boundary_left] = 1.
    x_ind_bc[_boundary_bottom] = 2.
    x_ind_bc[_boundary_right] = 3.
    x_ind_bc[_boundary_top] = 4.
    x_ind_bc[_boundary_left_bottom] = 5.
    x_ind_bc[_boundary_right_bottom] = 6.
    x_ind_bc[_boundary_right_top] = 7.
    x_ind_bc[_boundary_left_top] = 8.

    # pad BC indicator
    x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = np.expand_dims(x_ind_bc, axis=0)
    # x_ind_bc = torch.tensor(x_ind_bc)
    # ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], axis=0)
    # ind_bc = ind_bc.to(device)

    device = inputs.device

    inputs_np = inputs.detach().cpu().numpy()

    M_u = np.zeros((inputs_np.shape[0]*(inputs_np.shape[2] - 2)*(inputs_np.shape[3] - 2), inputs_np.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_np.shape[0]):
        for _x in np.arange(inputs_np.shape[2] - 2):
            for _y in np.arange(inputs_np.shape[3] - 2):
                M_u_part = inputs_np[_t:_t+1, :, _x:_x+3, _y:_y+3]
                M_u_part = M_u_part.reshape(M_u_part.shape[0], M_u_part.shape[1], -1)
                M_u_part = M_u_part.reshape(M_u_part.shape[0], -1)
                M_u[ni:ni+1, :] = M_u_part
                ni = ni + 1
    M_v = np.hstack([np.zeros_like(M_u), M_u])  # add the u part
    M_u = np.hstack([M_u, np.zeros_like(M_u)])  # add the v part

    # get the inputs after flatten
    M_uv_t0 = inputs0.detach().cpu().numpy()
    M_u_t0, M_v_t0 = M_uv_t0[:, 0, :, :].reshape(M_uv_t0.shape[0], -1), M_uv_t0[:, 1, :, :].reshape(M_uv_t0.shape[0], -1)
    M_u_t0, M_v_t0 = M_u_t0.flatten(), M_v_t0.flatten()

    # first iteration
    reg = 1e-1

    ### ------normal PDE------ ###  0
    _C = np.equal(ind_bc, 0)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix
    k = 5e-3
    Du = 1e-3
    Dv = 5e-3
    u0_0 = ut0.reshape(-1, 1)
    pde_M_u_normal_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_0*u0_0) + vC
    pde_M_u_normal_re = ut0/dt - k
    pde_M_v_normal_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_normal_re = vt0/dt

    ### ------left boundary------ ###  1
    _C = np.equal(ind_bc, 1)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uW = uC, vW = vC)
    u0_1 = ut0.reshape(-1, 1)
    pde_M_u_left_le = uC/dt - Du*(uE - uC)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_1*u0_1) + vC
    pde_M_u_left_re = ut0/dt - k
    pde_M_v_left_le = vC/dt - Dv*(vE - vC)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_left_re = vt0/dt

    ### ------bottom boundary------ ###  2
    _C = np.equal(ind_bc, 2)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    # uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uS = uC, vS = vC)
    u0_2 = ut0.reshape(-1, 1)
    pde_M_u_bottom_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - uC)/(dy*dy) - uC + uC*(u0_2*u0_2) + vC
    pde_M_u_bottom_re = ut0/dt - k
    pde_M_v_bottom_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(vN - vC)/(dy*dy) - uC + vC
    pde_M_v_bottom_re = vt0/dt

    ### ------right boundary------ ###  3
    _C = np.equal(ind_bc, 3)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uE = uC, vE = vC)
    u0_3 = ut0.reshape(-1, 1)
    pde_M_u_right_le = uC/dt - Du*(-uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_3*u0_3) + vC
    pde_M_u_right_re = ut0/dt - k
    pde_M_v_right_le = vC/dt - Dv*(-vC + vW)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_right_re = vt0/dt

    ### ------top boundary------ ###  4
    _C = np.equal(ind_bc, 4)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    # uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uN = uC, vN = vC)
    u0_4 = ut0.reshape(-1, 1)
    pde_M_u_top_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(-uC + uS)/(dy*dy) - uC + uC*(u0_4*u0_4) + vC
    pde_M_u_top_re = ut0/dt - k
    pde_M_v_top_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(-vC + vS)/(dy*dy) - uC + vC
    pde_M_v_top_re = vt0/dt

    # end = time.time()
    # runtime = end - start

    pde_M_u_le = np.vstack([pde_M_u_normal_le, pde_M_u_left_le, pde_M_u_bottom_le, pde_M_u_right_le, pde_M_u_top_le])
    pde_M_v_le = np.vstack([pde_M_v_normal_le, pde_M_v_left_le, pde_M_v_bottom_le, pde_M_v_right_le, pde_M_v_top_le])
    pde_M_u_re = np.vstack([pde_M_u_normal_re.reshape(-1, 1), pde_M_u_left_re.reshape(-1, 1), pde_M_u_bottom_re.reshape(-1, 1), pde_M_u_right_re.reshape(-1, 1), pde_M_u_top_re.reshape(-1, 1)])
    pde_M_v_re = np.vstack([pde_M_v_normal_re.reshape(-1, 1), pde_M_v_left_re.reshape(-1, 1), pde_M_v_bottom_re.reshape(-1, 1), pde_M_v_right_re.reshape(-1, 1), pde_M_v_top_re.reshape(-1, 1)])
    pde_M_le = np.vstack([pde_M_u_le, pde_M_v_le])
    pde_M_re = np.vstack([pde_M_u_re, pde_M_v_re])

    kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le.shape[1]) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

    # iteration to calculate the nonlinear term
    for i in np.arange(0):
        ### ------normal PDE------ ###  0
        _C = np.equal(ind_bc, 0)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_0 = uC @ kernel_parameters

        # compute PDE matrix
        pde_M_u_normal_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_0*u0_0) + vC

        ### ------left boundary------ ###  1
        _C = np.equal(ind_bc, 1)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_1 = uC @ kernel_parameters

        # compute PDE matrix (uW = uC, vW = vC)
        pde_M_u_left_le = uC/dt - Du*(uE - uC)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_1*u0_1) + vC

        ### ------bottom boundary------ ###  2
        _C = np.equal(ind_bc, 2)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        u0_2 = uC @ kernel_parameters

        # compute PDE matrix (uS = uC, vS = vC)
        pde_M_u_bottom_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - uC)/(dy*dy) - uC + uC*(u0_2*u0_2) + vC

        ### ------right boundary------ ###  3
        _C = np.equal(ind_bc, 3)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_3 = uC @ kernel_parameters

        # compute PDE matrix (uE = uC, vE = vC)
        pde_M_u_right_le = uC/dt - Du*(-uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_3*u0_3) + vC

        ### ------top boundary------ ###  4
        _C = np.equal(ind_bc, 4)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uS = M_u[_S, :]
        u0_4 = uC @ kernel_parameters

        # compute PDE matrix (uN = uC, vN = vC)
        pde_M_u_top_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(-uC + uS)/(dy*dy) - uC + uC*(u0_4*u0_4) + vC

        pde_M_u_le = np.vstack([pde_M_u_normal_le, pde_M_u_left_le, pde_M_u_bottom_le, pde_M_u_right_le, pde_M_u_top_le])
        pde_M_le = np.vstack([pde_M_u_le, pde_M_v_le])
        pde_M_re = np.vstack([pde_M_u_re, pde_M_v_re])

        kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le.shape[1]) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

        # test error
        # _C = np.equal(ind_bc, 0)
        # _C = _C.reshape(_C.shape[0], -1)
        # _C = _C.flatten()
        # uC = M_u[_C, :]
        # u0_0_new = uC @ kernel_parameters
        # error = np.mean((u0_0_new - u0_0)**2)
        # ttt = 1

    # pseudo inverse prediction results
    u, v = M_u @ kernel_parameters, M_v @ kernel_parameters
    u, v = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1)
    u, v = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    u, v = np.expand_dims(u, axis=1), np.expand_dims(v, axis=1)
    outputs_pseudo_inverse = np.concatenate([u, v], axis=1)
    outputs_pseudo_inverse = torch.tensor(outputs_pseudo_inverse).to(device).float()

    ssr = np.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # convolution 2d prediction results
    # conv2d_head = ConvOps(inputs.shape[1], inputs0.shape[1], kernel_size=3, ops_order='weight')
    # conv2d_head = conv2d_head.to(device)
    # outputs_conv2d = conv2d_head(inputs_conv2d).float()

    outputs_conv2d = conv2d_head(inputs_conv2d).float()


    return outputs_pseudo_inverse, outputs_conv2d, ssr







def CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC_test(inputs, inputs0, conv2d_head):

    inputs_conv2d = inputs

    # start = time.time()
    # kernel size: 1*1, stride: 1

    # computational boundary
    x_l, x_u, y_l, y_u = -1, 1, -1, 1
    # x_l, x_u, y_l, y_u = 0, 1, 0, 1
    ext = [x_l, x_u, y_l, y_u]
    t_l, t_u = 0, 5

    # dx, dy, dt
    n_x, n_y, n_t = 128, 128, 101
    # dx = (x_u - x_l) / (n_x - 1) #/ 2
    # dy = (y_u - y_l) / (n_y - 1) #/ 2
    # dt = (t_u - t_l) / (n_t - 1) #/ 2
    dx = dy = 0.015625
    dt = 0.05

    n_padding = int((inputs.shape[2] - n_x)/2)
    n_kernel_size = 1

    x = np.linspace(x_l, x_u, n_x)
    y = np.linspace(y_l, y_u, n_y)
    x_grid, y_grid = np.meshgrid(x, y)

    # add BC indicator
    _boundary_left = (x_grid == x_l) & (y_grid != y_l) & (y_grid != y_u)
    _boundary_bottom = (y_grid == y_l) & (x_grid != x_l) & (x_grid != x_u)
    _boundary_right = (x_grid == x_u) & (y_grid != y_l) & (y_grid != y_u)
    _boundary_top = (y_grid == y_u) & (x_grid != x_l) & (x_grid != x_u)
    _boundary_left_bottom = (x_grid == x_l) & (y_grid == y_l)
    _boundary_right_bottom = (x_grid == x_u) & (y_grid == y_l)
    _boundary_right_top = (x_grid == x_u) & (y_grid == y_u)
    _boundary_left_top = (x_grid == x_l) & (y_grid == y_u)

    x_ind_bc = np.zeros((n_y, n_x))
    x_ind_bc[_boundary_left] = 1.
    x_ind_bc[_boundary_bottom] = 2.
    x_ind_bc[_boundary_right] = 3.
    x_ind_bc[_boundary_top] = 4.
    x_ind_bc[_boundary_left_bottom] = 5.
    x_ind_bc[_boundary_right_bottom] = 6.
    x_ind_bc[_boundary_right_top] = 7.
    x_ind_bc[_boundary_left_top] = 8.

    # pad BC indicator
    x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = np.expand_dims(x_ind_bc, axis=0)
    # x_ind_bc = torch.tensor(x_ind_bc)
    # ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], axis=0)
    # ind_bc = ind_bc.to(device)

    device = inputs.device

    inputs_np = inputs.detach().cpu().numpy()

    M_u = np.zeros((inputs_np.shape[0]*inputs_np.shape[2]*inputs_np.shape[3], inputs_np.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_np.shape[0]):
        for _x in np.arange(inputs_np.shape[2]):
            for _y in np.arange(inputs_np.shape[3]):
                M_u_part = inputs_np[_t:_t+1, :, _x:_x+1, _y:_y+1]
                M_u_part = M_u_part.reshape(M_u_part.shape[0], M_u_part.shape[1], -1)
                M_u_part = M_u_part.reshape(M_u_part.shape[0], -1)
                M_u[ni:ni+1, :] = M_u_part
                ni = ni + 1
    M_v = np.hstack([np.zeros_like(M_u), M_u])  # add the u part
    M_u = np.hstack([M_u, np.zeros_like(M_u)])  # add the v part

    # get the inputs after flatten
    M_uv_t0 = inputs0.detach().cpu().numpy()
    M_u_t0, M_v_t0 = M_uv_t0[:, 0, :, :].reshape(M_uv_t0.shape[0], -1), M_uv_t0[:, 1, :, :].reshape(M_uv_t0.shape[0], -1)
    M_u_t0, M_v_t0 = M_u_t0.flatten(), M_v_t0.flatten()

    # first iteration
    reg = 1e-5

    ### ------normal PDE------ ###  0
    _C = np.equal(ind_bc, 0)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix
    k = 5e-3
    Du = 1e-3
    Dv = 5e-3
    u0_0 = ut0.reshape(-1, 1)
    pde_M_u_normal_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_0*u0_0) + vC
    pde_M_u_normal_re = ut0/dt - k
    pde_M_v_normal_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_normal_re = vt0/dt

    ### ------left boundary------ ###  1
    _C = np.equal(ind_bc, 1)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uW = uC, vW = vC)
    u0_1 = ut0.reshape(-1, 1)
    pde_M_u_left_le = uC/dt - Du*(uE - uC)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_1*u0_1) + vC
    pde_M_u_left_re = ut0/dt - k
    pde_M_v_left_le = vC/dt - Dv*(vE - vC)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_left_re = vt0/dt

    ### ------bottom boundary------ ###  2
    _C = np.equal(ind_bc, 2)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    # uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uS = uC, vS = vC)
    u0_2 = ut0.reshape(-1, 1)
    pde_M_u_bottom_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - uC)/(dy*dy) - uC + uC*(u0_2*u0_2) + vC
    pde_M_u_bottom_re = ut0/dt - k
    pde_M_v_bottom_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(vN - vC)/(dy*dy) - uC + vC
    pde_M_v_bottom_re = vt0/dt

    ### ------right boundary------ ###  3
    _C = np.equal(ind_bc, 3)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uE = uC, vE = vC)
    u0_3 = ut0.reshape(-1, 1)
    pde_M_u_right_le = uC/dt - Du*(-uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_3*u0_3) + vC
    pde_M_u_right_re = ut0/dt - k
    pde_M_v_right_le = vC/dt - Dv*(-vC + vW)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_right_re = vt0/dt

    ### ------top boundary------ ###  4
    _C = np.equal(ind_bc, 4)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    # uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uN = uC, vN = vC)
    u0_4 = ut0.reshape(-1, 1)
    pde_M_u_top_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(-uC + uS)/(dy*dy) - uC + uC*(u0_4*u0_4) + vC
    pde_M_u_top_re = ut0/dt - k
    pde_M_v_top_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(-vC + vS)/(dy*dy) - uC + vC
    pde_M_v_top_re = vt0/dt

    # end = time.time()
    # runtime = end - start

    pde_M_u_le = np.vstack([pde_M_u_normal_le, pde_M_u_left_le, pde_M_u_bottom_le, pde_M_u_right_le, pde_M_u_top_le])
    pde_M_v_le = np.vstack([pde_M_v_normal_le, pde_M_v_left_le, pde_M_v_bottom_le, pde_M_v_right_le, pde_M_v_top_le])
    pde_M_u_re = np.vstack([pde_M_u_normal_re.reshape(-1, 1), pde_M_u_left_re.reshape(-1, 1), pde_M_u_bottom_re.reshape(-1, 1), pde_M_u_right_re.reshape(-1, 1), pde_M_u_top_re.reshape(-1, 1)])
    pde_M_v_re = np.vstack([pde_M_v_normal_re.reshape(-1, 1), pde_M_v_left_re.reshape(-1, 1), pde_M_v_bottom_re.reshape(-1, 1), pde_M_v_right_re.reshape(-1, 1), pde_M_v_top_re.reshape(-1, 1)])
    pde_M_le = np.vstack([pde_M_u_le, pde_M_v_le])
    pde_M_re = np.vstack([pde_M_u_re, pde_M_v_re])

    kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le.shape[1]) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

    # iteration to calculate the nonlinear term
    for i in np.arange(0):
        ### ------normal PDE------ ###  0
        _C = np.equal(ind_bc, 0)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_0 = uC @ kernel_parameters

        # compute PDE matrix
        pde_M_u_normal_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_0*u0_0) + vC

        ### ------left boundary------ ###  1
        _C = np.equal(ind_bc, 1)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_1 = uC @ kernel_parameters

        # compute PDE matrix (uW = uC, vW = vC)
        pde_M_u_left_le = uC/dt - Du*(uE - uC)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_1*u0_1) + vC

        ### ------bottom boundary------ ###  2
        _C = np.equal(ind_bc, 2)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        u0_2 = uC @ kernel_parameters

        # compute PDE matrix (uS = uC, vS = vC)
        pde_M_u_bottom_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - uC)/(dy*dy) - uC + uC*(u0_2*u0_2) + vC

        ### ------right boundary------ ###  3
        _C = np.equal(ind_bc, 3)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_3 = uC @ kernel_parameters

        # compute PDE matrix (uE = uC, vE = vC)
        pde_M_u_right_le = uC/dt - Du*(-uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_3*u0_3) + vC

        ### ------top boundary------ ###  4
        _C = np.equal(ind_bc, 4)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uS = M_u[_S, :]
        u0_4 = uC @ kernel_parameters

        # compute PDE matrix (uN = uC, vN = vC)
        pde_M_u_top_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(-uC + uS)/(dy*dy) - uC + uC*(u0_4*u0_4) + vC

        pde_M_u_le = np.vstack([pde_M_u_normal_le, pde_M_u_left_le, pde_M_u_bottom_le, pde_M_u_right_le, pde_M_u_top_le])
        pde_M_le = np.vstack([pde_M_u_le, pde_M_v_le])
        pde_M_re = np.vstack([pde_M_u_re, pde_M_v_re])

        kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le.shape[1]) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

        # test error
        _C = np.equal(ind_bc, 0)
        _C = _C.reshape(_C.shape[0], -1)
        _C = _C.flatten()
        uC = M_u[_C, :]
        u0_0_new = uC @ kernel_parameters
        error = np.mean((u0_0_new - u0_0)**2)
        ttt = 1

    # pseudo inverse prediction results
    u, v = M_u @ kernel_parameters, M_v @ kernel_parameters
    u, v = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1)
    u, v = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    u, v = np.expand_dims(u, axis=1), np.expand_dims(v, axis=1)
    outputs_pseudo_inverse = np.concatenate([u, v], axis=1)
    outputs_pseudo_inverse = torch.tensor(outputs_pseudo_inverse).to(device).float()

    ssr = np.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # convolution 2d prediction results
    # conv2d_head = ConvOps(inputs.shape[1], inputs0.shape[1], kernel_size=3, ops_order='weight')
    # conv2d_head = conv2d_head.to(device)
    # outputs_conv2d = conv2d_head(inputs_conv2d).float()

    outputs_conv2d = conv2d_head(inputs_conv2d).float()


    return outputs_pseudo_inverse, outputs_conv2d, ssr



def CNN_PINN_Pseudo_Inverse_2D_DR_NeumannBC_test1(inputs, inputs0, conv2d_head):

    inputs_conv2d = inputs

    # start = time.time()
    # kernel size: 5*5, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (2, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 2, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 2, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, 2, 0, 0, 0, 0))

    # computational boundary
    x_l, x_u, y_l, y_u = -1, 1, -1, 1
    # x_l, x_u, y_l, y_u = 0, 1, 0, 1
    ext = [x_l, x_u, y_l, y_u]
    t_l, t_u = 0, 5

    # dx, dy, dt
    n_x, n_y, n_t = 128, 128, 101
    # dx = (x_u - x_l) / (n_x - 1) #/ 2
    # dy = (y_u - y_l) / (n_y - 1) #/ 2
    # dt = (t_u - t_l) / (n_t - 1) #/ 2
    dx = dy = 0.015625
    dt = 0.05

    n_padding = int((inputs0.shape[2] - n_x)/2)
    n_kernel_size = 5

    x = np.linspace(x_l, x_u, n_x)
    y = np.linspace(y_l, y_u, n_y)
    x_grid, y_grid = np.meshgrid(x, y)

    # add BC indicator
    _boundary_left = (x_grid == x_l) & (y_grid != y_l) & (y_grid != y_u)
    _boundary_bottom = (y_grid == y_l) & (x_grid != x_l) & (x_grid != x_u)
    _boundary_right = (x_grid == x_u) & (y_grid != y_l) & (y_grid != y_u)
    _boundary_top = (y_grid == y_u) & (x_grid != x_l) & (x_grid != x_u)
    _boundary_left_bottom = (x_grid == x_l) & (y_grid == y_l)
    _boundary_right_bottom = (x_grid == x_u) & (y_grid == y_l)
    _boundary_right_top = (x_grid == x_u) & (y_grid == y_u)
    _boundary_left_top = (x_grid == x_l) & (y_grid == y_u)

    x_ind_bc = np.zeros((n_y, n_x))
    x_ind_bc[_boundary_left] = 1.
    x_ind_bc[_boundary_bottom] = 2.
    x_ind_bc[_boundary_right] = 3.
    x_ind_bc[_boundary_top] = 4.
    x_ind_bc[_boundary_left_bottom] = 5.
    x_ind_bc[_boundary_right_bottom] = 6.
    x_ind_bc[_boundary_right_top] = 7.
    x_ind_bc[_boundary_left_top] = 8.

    # pad BC indicator
    x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = np.expand_dims(x_ind_bc, axis=0)
    # x_ind_bc = torch.tensor(x_ind_bc)
    # ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], axis=0)
    # ind_bc = ind_bc.to(device)

    device = inputs.device

    inputs_np = inputs.detach().cpu().numpy()

    M_u = np.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_conv2d.shape[0]):
        for _x in np.arange(inputs_conv2d.shape[2]):
            for _y in np.arange(inputs_conv2d.shape[3]):
                M_u_part = inputs_np[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                M_u_part = M_u_part.reshape(M_u_part.shape[0], M_u_part.shape[1], -1)
                M_u_part = M_u_part.reshape(M_u_part.shape[0], -1)
                M_u[ni:ni+1, :] = M_u_part
                ni = ni + 1
    M_v = np.hstack([np.zeros_like(M_u), M_u])  # add the u part
    M_u = np.hstack([M_u, np.zeros_like(M_u)])  # add the v part

    # get the inputs after flatten
    M_uv_t0 = inputs0.detach().cpu().numpy()
    M_u_t0, M_v_t0 = M_uv_t0[:, 0, :, :].reshape(M_uv_t0.shape[0], -1), M_uv_t0[:, 1, :, :].reshape(M_uv_t0.shape[0], -1)
    M_u_t0, M_v_t0 = M_u_t0.flatten(), M_v_t0.flatten()

    # first iteration
    reg = 1e-2

    ### ------normal PDE------ ###  0
    _C = np.equal(ind_bc, 0)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix
    k = 5e-3
    Du = 1e-3
    Dv = 5e-3
    u0_0 = ut0.reshape(-1, 1)
    pde_M_u_normal_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_0*u0_0) + vC
    pde_M_u_normal_re = ut0/dt - k
    pde_M_v_normal_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_normal_re = vt0/dt

    ### ------left boundary------ ###  1
    _C = np.equal(ind_bc, 1)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uW = uC, vW = vC)
    u0_1 = ut0.reshape(-1, 1)
    pde_M_u_left_le = uC/dt - Du*(uE - uC)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_1*u0_1) + vC
    pde_M_u_left_re = ut0/dt - k
    pde_M_v_left_le = vC/dt - Dv*(vE - vC)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_left_re = vt0/dt

    ### ------bottom boundary------ ###  2
    _C = np.equal(ind_bc, 2)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    # uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uS = uC, vS = vC)
    u0_2 = ut0.reshape(-1, 1)
    pde_M_u_bottom_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - uC)/(dy*dy) - uC + uC*(u0_2*u0_2) + vC
    pde_M_u_bottom_re = ut0/dt - k
    pde_M_v_bottom_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(vN - vC)/(dy*dy) - uC + vC
    pde_M_v_bottom_re = vt0/dt

    ### ------right boundary------ ###  3
    _C = np.equal(ind_bc, 3)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uE = uC, vE = vC)
    u0_3 = ut0.reshape(-1, 1)
    pde_M_u_right_le = uC/dt - Du*(-uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_3*u0_3) + vC
    pde_M_u_right_re = ut0/dt - k
    pde_M_v_right_le = vC/dt - Dv*(-vC + vW)/(dx*dx) - Dv*(vN - 2*vC + vS)/(dy*dy) - uC + vC
    pde_M_v_right_re = vt0/dt

    ### ------top boundary------ ###  4
    _C = np.equal(ind_bc, 4)
    _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
    _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
    _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
    _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC = M_u[_C, :], M_v[_C, :]
    uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW = M_u[_W, :], M_v[_W, :]
    # uN, vN = M_u[_N, :], M_v[_N, :]
    uS, vS = M_u[_S, :], M_v[_S, :]
    ut0, vt0 = M_u_t0[_C], M_v_t0[_C]

    # compute PDE matrix (uN = uC, vN = vC)
    u0_4 = ut0.reshape(-1, 1)
    pde_M_u_top_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(-uC + uS)/(dy*dy) - uC + uC*(u0_4*u0_4) + vC
    pde_M_u_top_re = ut0/dt - k
    pde_M_v_top_le = vC/dt - Dv*(vE - 2*vC + vW)/(dx*dx) - Dv*(-vC + vS)/(dy*dy) - uC + vC
    pde_M_v_top_re = vt0/dt

    # end = time.time()
    # runtime = end - start

    

    pde_M_u_le = np.vstack([pde_M_u_normal_le, pde_M_u_left_le, pde_M_u_bottom_le, pde_M_u_right_le, pde_M_u_top_le])
    pde_M_v_le = np.vstack([pde_M_v_normal_le, pde_M_v_left_le, pde_M_v_bottom_le, pde_M_v_right_le, pde_M_v_top_le])
    pde_M_u_re = np.vstack([pde_M_u_normal_re.reshape(-1, 1), pde_M_u_left_re.reshape(-1, 1), pde_M_u_bottom_re.reshape(-1, 1), pde_M_u_right_re.reshape(-1, 1), pde_M_u_top_re.reshape(-1, 1)])
    pde_M_v_re = np.vstack([pde_M_v_normal_re.reshape(-1, 1), pde_M_v_left_re.reshape(-1, 1), pde_M_v_bottom_re.reshape(-1, 1), pde_M_v_right_re.reshape(-1, 1), pde_M_v_top_re.reshape(-1, 1)])
    pde_M_le = np.vstack([pde_M_u_le, pde_M_v_le])
    pde_M_re = np.vstack([pde_M_u_re, pde_M_v_re])

    start = time.time()
    kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le.shape[1]) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re
    end = time.time()
    runtime = end - start

    # iteration to calculate the nonlinear term
    for i in np.arange(0):
        ### ------normal PDE------ ###  0
        _C = np.equal(ind_bc, 0)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_0 = uC @ kernel_parameters

        # compute PDE matrix
        pde_M_u_normal_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_0*u0_0) + vC

        ### ------left boundary------ ###  1
        _C = np.equal(ind_bc, 1)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_1 = uC @ kernel_parameters

        # compute PDE matrix (uW = uC, vW = vC)
        pde_M_u_left_le = uC/dt - Du*(uE - uC)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_1*u0_1) + vC

        ### ------bottom boundary------ ###  2
        _C = np.equal(ind_bc, 2)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        u0_2 = uC @ kernel_parameters

        # compute PDE matrix (uS = uC, vS = vC)
        pde_M_u_bottom_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(uN - uC)/(dy*dy) - uC + uC*(u0_2*u0_2) + vC

        ### ------right boundary------ ###  3
        _C = np.equal(ind_bc, 3)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uW = M_u[_W, :]
        uN = M_u[_N, :]
        uS = M_u[_S, :]
        u0_3 = uC @ kernel_parameters

        # compute PDE matrix (uE = uC, vE = vC)
        pde_M_u_right_le = uC/dt - Du*(-uC + uW)/(dx*dx) - Du*(uN - 2*uC + uS)/(dy*dy) - uC + uC*(u0_3*u0_3) + vC

        ### ------top boundary------ ###  4
        _C = np.equal(ind_bc, 4)
        _E = np.pad(_C[:, :, :-1], ((0, 0), (0, 0), (1, 0)), 'constant')
        _W = np.pad(_C[:, :, 1:], ((0, 0), (0, 0), (0, 1)), 'constant')
        _N = np.pad(_C[:, :-1, :], ((0, 0), (1, 0), (0, 0)), 'constant')
        _S = np.pad(_C[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC = M_u[_C, :], M_v[_C, :]
        uE = M_u[_E, :]
        uW = M_u[_W, :]
        uS = M_u[_S, :]
        u0_4 = uC @ kernel_parameters

        # compute PDE matrix (uN = uC, vN = vC)
        pde_M_u_top_le = uC/dt - Du*(uE - 2*uC + uW)/(dx*dx) - Du*(-uC + uS)/(dy*dy) - uC + uC*(u0_4*u0_4) + vC

        pde_M_u_le = np.vstack([pde_M_u_normal_le, pde_M_u_left_le, pde_M_u_bottom_le, pde_M_u_right_le, pde_M_u_top_le])
        pde_M_le = np.vstack([pde_M_u_le, pde_M_v_le])
        pde_M_re = np.vstack([pde_M_u_re, pde_M_v_re])

        kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le.shape[1]) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

        # test error
        _C = np.equal(ind_bc, 0)
        _C = _C.reshape(_C.shape[0], -1)
        _C = _C.flatten()
        uC = M_u[_C, :]
        u0_0_new = uC @ kernel_parameters
        error = np.mean((u0_0_new - u0_0)**2)
        ttt = 1

    # pseudo inverse prediction results
    u, v = M_u @ kernel_parameters, M_v @ kernel_parameters
    u, v = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1)
    u, v = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    u, v = np.expand_dims(u, axis=1), np.expand_dims(v, axis=1)
    outputs_pseudo_inverse = np.concatenate([u, v], axis=1)
    outputs_pseudo_inverse = torch.tensor(outputs_pseudo_inverse).to(device).float()

    ssr = np.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # convolution 2d prediction results
    # conv2d_head = ConvOps(inputs.shape[1], inputs0.shape[1], kernel_size=3, ops_order='weight')
    # conv2d_head = conv2d_head.to(device)
    # outputs_conv2d = conv2d_head(inputs_conv2d).float()

    outputs_conv2d = conv2d_head(inputs_conv2d).float()


    return outputs_pseudo_inverse, outputs_conv2d, ssr




def CNN_DNN_Pseudo_Inverse_2D_DR(inputs, inputs0, labels, conv2d_head):

    inputs_conv2d = inputs

    # start = time.time()
    # kernel size: 3*3, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (1, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 1, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 1, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, 1, 0, 0, 0, 0))

    # computational boundary
    x_l, x_u, y_l, y_u = -1, 1, -1, 1
    # x_l, x_u, y_l, y_u = 0, 1, 0, 1
    ext = [x_l, x_u, y_l, y_u]
    t_l, t_u = 0, 5

    # dx, dy, dt
    n_x, n_y, n_t = 128, 128, 101
    # dx = (x_u - x_l) / (n_x - 1) #/ 2
    # dy = (y_u - y_l) / (n_y - 1) #/ 2
    # dt = (t_u - t_l) / (n_t - 1) #/ 2
    dx = dy = 0.015625
    dt = 0.05

    n_padding = int((inputs.shape[2] - 2 - n_x)/2)
    n_kernel_size = 3

    x = np.linspace(x_l, x_u, n_x)
    y = np.linspace(y_l, y_u, n_y)
    x_grid, y_grid = np.meshgrid(x, y)

    device = inputs.device

    inputs_np = inputs.detach().cpu().numpy()

    M_u = np.zeros((inputs_np.shape[0]*(inputs_np.shape[2] - 2)*(inputs_np.shape[3] - 2), inputs_np.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_np.shape[0]):
        for _x in np.arange(inputs_np.shape[2] - 2):
            for _y in np.arange(inputs_np.shape[3] - 2):
                M_u_part = inputs_np[_t:_t+1, :, _x:_x+3, _y:_y+3]
                M_u_part = M_u_part.reshape(M_u_part.shape[0], M_u_part.shape[1], -1)
                M_u_part = M_u_part.reshape(M_u_part.shape[0], -1)
                M_u[ni:ni+1, :] = M_u_part
                ni = ni + 1
    M_v = np.hstack([np.zeros_like(M_u), M_u])  # add the u part
    M_u = np.hstack([M_u, np.zeros_like(M_u)])  # add the v part

    # get the inputs after flatten
    M_uv_t0 = inputs0.detach().cpu().numpy()
    M_u_t0, M_v_t0 = M_uv_t0[:, 0, :, :].reshape(M_uv_t0.shape[0], -1), M_uv_t0[:, 1, :, :].reshape(M_uv_t0.shape[0], -1)
    M_u_t0, M_v_t0 = M_u_t0.flatten(), M_v_t0.flatten()

    # first iteration
    reg = 1e-2

    uC, vC = M_u, M_v

    labels = labels.detach().cpu().numpy()
    u_labels, v_labels = labels[:, 0, :, :], labels[:, 1, :, :]
    u_labels, v_labels = u_labels.reshape(u_labels.shape[0], -1), v_labels.reshape(v_labels.shape[0], -1)
    u_labels, v_labels = u_labels.flatten(), v_labels.flatten()
    u_labels, v_labels = u_labels.reshape(-1, 1), v_labels.reshape(-1, 1)

    pde_M_le = np.vstack([uC, vC])
    pde_M_re = np.vstack([u_labels, v_labels])

    kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le.shape[1]) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

    # pseudo inverse prediction results
    u, v = M_u @ kernel_parameters, M_v @ kernel_parameters
    u, v = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1)
    u, v = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    u, v = np.expand_dims(u, axis=1), np.expand_dims(v, axis=1)
    outputs_pseudo_inverse = np.concatenate([u, v], axis=1)
    outputs_pseudo_inverse = torch.tensor(outputs_pseudo_inverse).to(device).float()

    ssr = np.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # convolution 2d prediction results
    # conv2d_head = ConvOps(inputs.shape[1], inputs0.shape[1], kernel_size=3, ops_order='weight')
    # conv2d_head = conv2d_head.to(device)
    # outputs_conv2d = conv2d_head(inputs_conv2d).float()

    outputs_conv2d = conv2d_head(inputs_conv2d).float()


    return outputs_pseudo_inverse, outputs_conv2d, ssr




class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, SE=False):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE=SE)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)  # previous cell outputs
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class PyramidNetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, increment=4, SE=False):
        super(PyramidNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE=SE)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

            C_curr += increment

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


if __name__ == '__main__':
    import util.utils as utils
    import models.micro_genotypes as genotypes

    genome = genotypes.NSGANet
    # model = AlterPyramidNetworkCIFAR(30, 10, 20, True, genome, 6, SE=False)
    model = PyramidNetworkCIFAR(48, 10, 20, True, genome, 22, SE=True)
    # model = NetworkCIFAR(34, 10, 20, True, genome, SE=True)
    # model = GradPyramidNetworkCIFAR(34, 10, 20, True, genome, 4)
    model.droprate = 0.0

    # calculate number of trainable parameters
    print("param size = {}MB".format(utils.count_parameters_in_MB(model)))
