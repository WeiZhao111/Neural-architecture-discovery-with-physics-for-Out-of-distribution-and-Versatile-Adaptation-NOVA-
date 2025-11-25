from models.micro_operations import *
from util.utils import drop_path
from util.prim_ops_set import *
from .fcn import FCNHead
from .base import BaseNet
from util.functional import *
from torch.nn.functional import interpolate, softmax
import time
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from models.diffusion_related import Index_Transformation, Index_Transformation_01


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


    # def forward(self, x, source_term, targets):
    # def forward(self, x, source_term, targets, lamda, flag):
    # def forward(self, x, targets):
    # def forward(self, x, targets, lamda, flag):
    # def forward(self, x, lamda, reg, flag):
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

        outputs_pseudo_inverse, outputs_conv2d, ssr = CNN_PINN_Pseudo_Inverse_2D_NS_DirichletBC(s1, x, self.nas_unet_head)
        # outputs = CNN_PINN_Pseudo_Inverse_2D_NS_DirichletBC_grad(s1, x, self.nas_unet_head)

        # if flag == 0:
        #     outputs = self.nas_unet_head(s1)
        # else:
        #     outputs = CNN_PINN_Pseudo_Inverse_2D_NS_DirichletBC_reg(s1, x, self.nas_unet_head, lamda, reg)

        # outputs = []
        # outputs.append(output)

        # if self.aux: # use aux header
        #     auxout = self.auxlayer(s1)
        #     auxout = interpolate(auxout, (h,w), **self._up_kwargs)
        #     outputs.append(auxout)

        return outputs_pseudo_inverse, ssr, x






class NasUnet_PI(BaseNet):
    """Construct a network"""

    def __init__(self, original_model, nclass, in_channels, backbone=None, aux=False,
                 c=48, depth=5, dropout_prob=0,
                 genotype=None, double_down_channel=False):

        super(NasUnet_PI, self).__init__(nclass, aux, backbone, norm_layer=nn.GroupNorm)

        # the stem need a complicate mode
        # self.stem0 = ConvOps(in_channels, c_prev_prev, kernel_size=1, ops_order='weight_norm')
        # self.stem1 = ConvOps(in_channels, c_prev, kernel_size=3, stride=2, ops_order='weight_norm')
        self.stem0 = original_model.stem0
        self.stem1 = original_model.stem1

        assert depth >= 2 , 'depth must >= 2'

        # self.down_cells = nn.ModuleList()
        # self.up_cells = nn.ModuleList()
        # down_cs_nfilters = []

        # # create the encoder pathway and add to a list
        # down_cs_nfilters += [c_prev]
        # down_cs_nfilters += [c_prev_prev]
        # for i in range(depth):
        #     c_curr = 2 * c_curr if self._double_down_channel else c_curr  # double the number of filters
        #     down_cell = BuildCell(genotype, c_prev_prev, c_prev, c_curr, cell_type='down', dropout_prob=dropout_prob)
        #     self.down_cells += [down_cell]
        #     c_prev_prev, c_prev = c_prev, down_cell._multiplier*c_curr
        #     down_cs_nfilters += [c_prev]

        # # create the decoder pathway and add to a list
        # for i in range(depth+1):
        #     # c_curr = c_curr // 2 if self._double_down_channel else c_curr  # halve the number of filters
        #     c_prev_prev = down_cs_nfilters[-(i + 2)] # the horizontal prev_prev input channel
        #     up_cell = BuildCell(genotype, c_prev_prev, c_prev, c_curr, cell_type='up',  dropout_prob=dropout_prob)
        #     self.up_cells += [up_cell]
        #     c_prev = up_cell._multiplier*c_curr
        #     c_curr = c_curr // 2 if self._double_down_channel else c_curr  # halve the number of filters
        
        self.down_cells = original_model.down_cells
        self.up_cells = original_model.up_cells

        # self.nas_unet_head = ConvOps(c_prev, nclass, kernel_size=3, ops_order='weight')
        self.nas_unet_head = original_model.nas_unet_head

        self.PI_last_layer = PseudoInverseLayer()

        self.threshold_grad = ThresholdGrad()


        # if self.aux:
        #     self.auxlayer = FCNHead(c_prev, nclass, nn.BatchNorm2d)
        
        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(c_prev, nclass)


    # def forward(self, x, source_term, targets):
    # def forward(self, x, source_term, targets, lamda, flag):
    # def forward(self, x, targets):
    # def forward(self, x, targets, lamda, flag):
    # def forward(self, x, targets, lamda, reg, flag):
    def forward(self, x):
        
        x = torch.nn.functional.interpolate(x, size=(32, 64), mode='bilinear', align_corners=True)
        x = torch.nn.functional.pad(x, (0, 0, 16, 16, 0, 0, 0, 0), 'constant', torch.max(x).detach().cpu().float())
        
        
        x = torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        # threshold = (torch.max(x) + torch.min(x))/2
        # x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        # x = self.threshold_grad(x)
        # x = self.sigmoid_layer(x)
        x = torch.sigmoid(x)
        x = Index_Transformation_01(x)
        x = Index_Transformation(x)
        # outputs = x

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

        outputs, ssr = self.PI_last_layer(s1, x)

        return outputs, ssr, x


class MyRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        rounded = torch.round(input)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class ThresholdGrad(nn.Module):
    def _init_(self):
        super(ThresholdGrad, self)._init_()

    def forward(self, input):
        return MyRoundFunction.apply(input)




class PseudoInverseLayer(nn.Module):
    def __init__(self):

        super(PseudoInverseLayer, self).__init__()
    
    def CNN_PINN_Pseudo_Inverse_2D_NS_DirichletBC(self, inputs, inputs0):

        inputs_conv2d = inputs

        n_kernel_size = 7
        n_p = int(np.floor(n_kernel_size/2))
        # kernel size: 1*1, 3*3, 5*5, stride: 1
        inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
        inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
        inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
        inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

        # computational boundary
        # x_l, x_u, y_l, y_u = 0, 2, 0, 1
        x_l, x_u, y_l, y_u = 0, 2, 0, 2

        # dx, dy
        # n_x, n_y = 64, 64
        n_x, n_y = 128, 128
        # n_x, n_y = 512, 512
        dx = (x_u - x_l) / n_x  #/ 2
        dy = (y_u - y_l) / n_y  #/ 2

        x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
        y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)
        x_grid, y_grid = np.meshgrid(x, y)

        # n_padding = int((inputs0.shape[2] - n_x)/2)

        device = inputs.device

        x_ind_bc = inputs0[0, 0, :, :]

        # pad BC indicator
        # x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
        x_ind_bc = torch.unsqueeze(x_ind_bc, axis=0)
        ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)

        # inputs_np = inputs.detach().cpu().numpy()

        M0 = torch.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
        ni = 0
        for _t in np.arange(inputs_conv2d.shape[0]):
            for _x in np.arange(inputs_conv2d.shape[2]):
                for _y in np.arange(inputs_conv2d.shape[3]):
                    M0_part = inputs[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                    M0_part = M0_part.reshape(M0_part.shape[0], M0_part.shape[1], -1)
                    M0_part = M0_part.reshape(M0_part.shape[0], -1)
                    M0[ni:ni+1, :] = M0_part
                    ni = ni + 1

        M_u = torch.hstack([M0, torch.zeros_like(M0), torch.zeros_like(M0)])
        M_v = torch.hstack([torch.zeros_like(M0), M0, torch.zeros_like(M0)])
        M_p = torch.hstack([torch.zeros_like(M0), torch.zeros_like(M0), M0])
        # M_u, M_v, M_p = torch.tensor(M_u).to(device), torch.tensor(M_v).to(device), torch.tensor(M_p).to(device)
        M_u, M_v, M_p = M_u.to(device).double(), M_v.to(device).double(), M_p.to(device).double()

        # get BC
        # M_u_bc, M_v_bc, M_p_bc = inputs0[:, 1, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 2, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 3, :, :].reshape(inputs0.shape[0], -1)
        # M_u_bc, M_v_bc, M_p_bc = M_u_bc.flatten(), M_v_bc.flatten(), M_p_bc.flatten()

        M_u_bc_t = M_v_bc_t = M_p_bc_t = torch.zeros((1, 128, 128))
        index_u = torch.eq(ind_bc, 15) | torch.eq(ind_bc, 16) | torch.eq(ind_bc, 17) | torch.eq(ind_bc, 18) | torch.eq(ind_bc, 19) | torch.eq(ind_bc, 20)
        M_u_bc_t[index_u] = 1
        M_u_bc_t, M_v_bc_t, M_p_bc_t = M_u_bc_t.reshape(1, -1), M_v_bc_t.reshape(1, -1), M_p_bc_t.reshape(1, -1)
        M_u_bc_t, M_v_bc_t, M_p_bc_t = M_u_bc_t.flatten().to(device), M_v_bc_t.flatten().to(device), M_p_bc_t.flatten().to(device)


        # first iteration
        reg = 1e-4
        # reg = 5e-5
        # reg = 1.83673469e+03  # kernel 5

        ### ------normal PDE------ ###  0
        _C = torch.eq(ind_bc, 0)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :] 
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        Re = 500
        # Re = 0.1
        pde_normal_1_le = u_x + v_y
        pde_normal_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_normal_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_normal_1_re = torch.zeros(pde_normal_1_le.shape[0], 1)
        pde_normal_2_re = torch.zeros(pde_normal_2_le.shape[0], 1)
        pde_normal_3_re = torch.zeros(pde_normal_3_le.shape[0], 1)
        pde_normal_1_re, pde_normal_2_re, pde_normal_3_re = pde_normal_1_re.to(device), pde_normal_2_re.to(device), pde_normal_3_re.to(device)


        ### ----------W_1---------- ###  1
        _C = torch.eq(ind_bc, 1)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_W_1_1_le = u_x + v_y
        pde_W_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_W_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_W_1_1_re = torch.zeros(pde_W_1_1_le.shape[0], 1)
        pde_W_1_2_re = torch.zeros(pde_W_1_2_le.shape[0], 1)
        pde_W_1_3_re = torch.zeros(pde_W_1_3_le.shape[0], 1)
        pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re = pde_W_1_1_re.to(device), pde_W_1_2_re.to(device), pde_W_1_3_re.to(device)


        ### ----------S_1---------- ###  2
        _C = torch.eq(ind_bc, 2)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_S_1_1_le = u_x + v_y
        pde_S_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_S_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_S_1_1_re = torch.zeros(pde_S_1_1_le.shape[0], 1)
        pde_S_1_2_re = torch.zeros(pde_S_1_2_le.shape[0], 1)
        pde_S_1_3_re = torch.zeros(pde_S_1_3_le.shape[0], 1)
        pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re = pde_S_1_1_re.to(device), pde_S_1_2_re.to(device), pde_S_1_3_re.to(device)


        ### ----------E_1---------- ###  3
        _C = torch.eq(ind_bc, 3)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_E_1_1_le = u_x + v_y
        pde_E_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_E_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_E_1_1_re = torch.zeros(pde_E_1_1_le.shape[0], 1)
        pde_E_1_2_re = torch.zeros(pde_E_1_2_le.shape[0], 1)
        pde_E_1_3_re = torch.zeros(pde_E_1_3_le.shape[0], 1)
        pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re = pde_E_1_1_re.to(device), pde_E_1_2_re.to(device), pde_E_1_3_re.to(device)


        ### ----------N_1---------- ###  4
        _C = torch.eq(ind_bc, 4)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_N_1_1_le = u_x + v_y
        pde_N_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_N_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_N_1_1_re = torch.zeros(pde_N_1_1_le.shape[0], 1)
        pde_N_1_2_re = torch.zeros(pde_N_1_2_le.shape[0], 1)
        pde_N_1_3_re = torch.zeros(pde_N_1_3_le.shape[0], 1)
        pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re = pde_N_1_1_re.to(device), pde_N_1_2_re.to(device), pde_N_1_3_re.to(device)


        ### ----------WS_2---------- ###  5
        _C = torch.eq(ind_bc, 5)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_WS_2_1_le = u_x + v_y
        pde_WS_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WS_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_WS_2_1_re = torch.zeros(pde_WS_2_1_le.shape[0], 1)
        pde_WS_2_2_re = torch.zeros(pde_WS_2_2_le.shape[0], 1)
        pde_WS_2_3_re = torch.zeros(pde_WS_2_3_le.shape[0], 1)
        pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re = pde_WS_2_1_re.to(device), pde_WS_2_2_re.to(device), pde_WS_2_3_re.to(device)


        ### ----------WE_2---------- ###  6
        _C = torch.eq(ind_bc, 6)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_WE_2_1_le = u_x + v_y
        pde_WE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_WE_2_1_re = torch.zeros(pde_WE_2_1_le.shape[0], 1)
        pde_WE_2_2_re = torch.zeros(pde_WE_2_2_le.shape[0], 1)
        pde_WE_2_3_re = torch.zeros(pde_WE_2_3_le.shape[0], 1)
        pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re = pde_WE_2_1_re.to(device), pde_WE_2_2_re.to(device), pde_WE_2_3_re.to(device)


        ### ----------WN_2---------- ###  7
        _C = torch.eq(ind_bc, 7)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uW, vW, pW = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_WN_2_1_le = u_x + v_y
        pde_WN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_WN_2_1_re = torch.zeros(pde_WN_2_1_le.shape[0], 1)
        pde_WN_2_2_re = torch.zeros(pde_WN_2_2_le.shape[0], 1)
        pde_WN_2_3_re = torch.zeros(pde_WN_2_3_le.shape[0], 1)
        pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re = pde_WN_2_1_re.to(device), pde_WN_2_2_re.to(device), pde_WN_2_3_re.to(device)


        ### ----------SE_2---------- ###  8
        _C = torch.eq(ind_bc, 8)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_SE_2_1_le = u_x + v_y
        pde_SE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_SE_2_1_re = torch.zeros(pde_SE_2_1_le.shape[0], 1)
        pde_SE_2_2_re = torch.zeros(pde_SE_2_2_le.shape[0], 1)
        pde_SE_2_3_re = torch.zeros(pde_SE_2_3_le.shape[0], 1)
        pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re = pde_SE_2_1_re.to(device), pde_SE_2_2_re.to(device), pde_SE_2_3_re.to(device)


        ### ----------SN_2---------- ###  9
        _C = torch.eq(ind_bc, 9)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_SN_2_1_le = u_x + v_y
        pde_SN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_SN_2_1_re = torch.zeros(pde_SN_2_1_le.shape[0], 1)
        pde_SN_2_2_re = torch.zeros(pde_SN_2_2_le.shape[0], 1)
        pde_SN_2_3_re = torch.zeros(pde_SN_2_3_le.shape[0], 1)
        pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re = pde_SN_2_1_re.to(device), pde_SN_2_2_re.to(device), pde_SN_2_3_re.to(device)


        ### ----------EN_2---------- ###  10
        _C = torch.eq(ind_bc, 10)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_EN_2_1_le = u_x + v_y
        pde_EN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_EN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_EN_2_1_re = torch.zeros(pde_EN_2_1_le.shape[0], 1)
        pde_EN_2_2_re = torch.zeros(pde_EN_2_2_le.shape[0], 1)
        pde_EN_2_3_re = torch.zeros(pde_EN_2_3_le.shape[0], 1)
        pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re = pde_EN_2_1_re.to(device), pde_EN_2_2_re.to(device), pde_EN_2_3_re.to(device)


        ### ----------WSE_3---------- ###  11
        _C = torch.eq(ind_bc, 11)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_WSE_3_1_le = u_x + v_y
        pde_WSE_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WSE_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_WSE_3_1_re = torch.zeros(pde_WSE_3_1_le.shape[0], 1)
        pde_WSE_3_2_re = torch.zeros(pde_WSE_3_2_le.shape[0], 1)
        pde_WSE_3_3_re = torch.zeros(pde_WSE_3_3_le.shape[0], 1)
        pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re = pde_WSE_3_1_re.to(device), pde_WSE_3_2_re.to(device), pde_WSE_3_3_re.to(device)


        ### ----------SEN_3---------- ###  12
        _C = torch.eq(ind_bc, 12)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_SEN_3_1_le = u_x + v_y
        pde_SEN_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SEN_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_SEN_3_1_re = torch.zeros(pde_SEN_3_1_le.shape[0], 1)
        pde_SEN_3_2_re = torch.zeros(pde_SEN_3_2_le.shape[0], 1)
        pde_SEN_3_3_re = torch.zeros(pde_SEN_3_3_le.shape[0], 1)
        pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re = pde_SEN_3_1_re.to(device), pde_SEN_3_2_re.to(device), pde_SEN_3_3_re.to(device)


        ### ----------ENW_3---------- ###  13
        _C = torch.eq(ind_bc, 13)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_ENW_3_1_le = u_x + v_y
        pde_ENW_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_ENW_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_ENW_3_1_re = torch.zeros(pde_ENW_3_1_le.shape[0], 1)
        pde_ENW_3_2_re = torch.zeros(pde_ENW_3_2_le.shape[0], 1)
        pde_ENW_3_3_re = torch.zeros(pde_ENW_3_3_le.shape[0], 1)
        pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re = pde_ENW_3_1_re.to(device), pde_ENW_3_2_re.to(device), pde_ENW_3_3_re.to(device)


        ### ----------NWS_3---------- ###  14
        _C = torch.eq(ind_bc, 14)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        pde_NWS_3_1_le = u_x + v_y
        pde_NWS_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_NWS_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        pde_NWS_3_1_re = torch.zeros(pde_NWS_3_1_le.shape[0], 1)
        pde_NWS_3_2_re = torch.zeros(pde_NWS_3_2_le.shape[0], 1)
        pde_NWS_3_3_re = torch.zeros(pde_NWS_3_3_le.shape[0], 1)
        pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re = pde_NWS_3_1_re.to(device), pde_NWS_3_2_re.to(device), pde_NWS_3_3_re.to(device)


        ### ----------left---------- ###  15
        _C = torch.eq(ind_bc, 15)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        uW, vW, pW = uC, vC, pC

        # compute PDE matrix
        pde_left_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_left_1_re = torch.zeros(pde_left_1_le.shape[0], 1)
        pde_left_2_re = torch.zeros(pde_left_2_le.shape[0], 1)
        pde_left_3_re = torch.zeros(pde_left_3_le.shape[0], 1)
        pde_left_1_re, pde_left_2_re, pde_left_3_re = pde_left_1_re.to(device), pde_left_2_re.to(device), pde_left_3_re.to(device)

        pde_left_4_le = uC
        pde_left_5_le = vC
        pde_left_4_re = uC_bc_t
        pde_left_5_re = vC_bc_t


        ### ----------left_E---------- ###  16
        _C = torch.eq(ind_bc, 16)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        pde_left_E_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_E_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_E_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_left_E_1_re = torch.zeros(pde_left_E_1_le.shape[0], 1)
        pde_left_E_2_re = torch.zeros(pde_left_E_2_le.shape[0], 1)
        pde_left_E_3_re = torch.zeros(pde_left_E_3_le.shape[0], 1)
        pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re = pde_left_E_1_re.to(device), pde_left_E_2_re.to(device), pde_left_E_3_re.to(device)

        pde_left_E_4_le = uC
        pde_left_E_5_le = vC
        pde_left_E_4_re = uC_bc_t
        pde_left_E_5_re = vC_bc_t


        ### ----------left_N---------- ###  17
        _C = torch.eq(ind_bc, 17)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uW, vW, pW = uC, vC, pC
        uN, vN, pN = -uC, -vC, pC

        # compute PDE matrix
        pde_left_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_left_N_1_re = torch.zeros(pde_left_N_1_le.shape[0], 1)
        pde_left_N_2_re = torch.zeros(pde_left_N_2_le.shape[0], 1)
        pde_left_N_3_re = torch.zeros(pde_left_N_3_le.shape[0], 1)
        pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re = pde_left_N_1_re.to(device), pde_left_N_2_re.to(device), pde_left_N_3_re.to(device)

        pde_left_N_4_le = uC
        pde_left_N_5_le = vC
        pde_left_N_4_re = uC_bc_t
        pde_left_N_5_re = vC_bc_t


        ### ----------left_NE---------- ###  18
        _C = torch.eq(ind_bc, 18)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uN, vN, pN = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        pde_left_NE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_NE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_NE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_left_NE_1_re = torch.zeros(pde_left_NE_1_le.shape[0], 1)
        pde_left_NE_2_re = torch.zeros(pde_left_NE_2_le.shape[0], 1)
        pde_left_NE_3_re = torch.zeros(pde_left_NE_3_le.shape[0], 1)
        pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re = pde_left_NE_1_re.to(device), pde_left_NE_2_re.to(device), pde_left_NE_3_re.to(device)

        pde_left_NE_4_le = uC
        pde_left_NE_5_le = vC
        pde_left_NE_4_re = uC_bc_t
        pde_left_NE_5_re = vC_bc_t


        ### ----------left_S---------- ###  19
        _C = torch.eq(ind_bc, 19)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uW, vW, pW = uC, vC, pC
        uS, vS, pS = -uC, -vC, pC

        # compute PDE matrix
        pde_left_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_left_S_1_re = torch.zeros(pde_left_S_1_le.shape[0], 1)
        pde_left_S_2_re = torch.zeros(pde_left_S_2_le.shape[0], 1)
        pde_left_S_3_re = torch.zeros(pde_left_S_3_le.shape[0], 1)
        pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re = pde_left_S_1_re.to(device), pde_left_S_2_re.to(device), pde_left_S_3_re.to(device)

        pde_left_S_4_le = uC
        pde_left_S_5_le = vC
        pde_left_S_4_re = uC_bc_t
        pde_left_S_5_re = vC_bc_t


        ### ----------left_SE---------- ###  20
        _C = torch.eq(ind_bc, 20)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        pde_left_SE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_SE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_SE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_left_SE_1_re = torch.zeros(pde_left_SE_1_le.shape[0], 1)
        pde_left_SE_2_re = torch.zeros(pde_left_SE_2_le.shape[0], 1)
        pde_left_SE_3_re = torch.zeros(pde_left_SE_3_le.shape[0], 1)
        pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re = pde_left_SE_1_re.to(device), pde_left_SE_2_re.to(device), pde_left_SE_3_re.to(device)

        pde_left_SE_4_le = uC
        pde_left_SE_5_le = vC
        pde_left_SE_4_re = uC_bc_t
        pde_left_SE_5_re = vC_bc_t


        ### ----------right---------- ###  21
        _C = torch.eq(ind_bc, 21)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0 (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        uE, vE, pE = uC, vC, -pC

        # compute PDE matrix
        pde_right_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_right_1_re = torch.zeros(pde_right_1_le.shape[0], 1)
        pde_right_2_re = torch.zeros(pde_right_2_le.shape[0], 1)
        pde_right_3_re = torch.zeros(pde_right_3_le.shape[0], 1)
        pde_right_1_re, pde_right_2_re, pde_right_3_re = pde_right_1_re.to(device), pde_right_2_re.to(device), pde_right_3_re.to(device)
        pde_right_4_le = pC
        pde_right_4_re = pC_bc_t


        ### ----------right_W---------- ###  22
        _C = torch.eq(ind_bc, 22)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        pde_right_W_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_W_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_W_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_right_W_1_re = torch.zeros(pde_right_W_1_le.shape[0], 1)
        pde_right_W_2_re = torch.zeros(pde_right_W_2_le.shape[0], 1)
        pde_right_W_3_re = torch.zeros(pde_right_W_3_le.shape[0], 1)
        pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re = pde_right_W_1_re.to(device), pde_right_W_2_re.to(device), pde_right_W_3_re.to(device)
        pde_right_W_4_le = pC
        pde_right_W_4_re = pC_bc_t


        ### ----------right_N---------- ###  23
        _C = torch.eq(ind_bc, 23)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uE, vE, pE = uC, vC, -pC
        uN, vN, pN = -uC, -vC, pC

        # compute PDE matrix
        pde_right_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_right_N_1_re = torch.zeros(pde_right_N_1_le.shape[0], 1)
        pde_right_N_2_re = torch.zeros(pde_right_N_2_le.shape[0], 1)
        pde_right_N_3_re = torch.zeros(pde_right_N_3_le.shape[0], 1)
        pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re = pde_right_N_1_re.to(device), pde_right_N_2_re.to(device), pde_right_N_3_re.to(device)
        pde_right_N_4_le = pC
        pde_right_N_4_re = pC_bc_t


        ### ----------right_NW---------- ###  24
        _C = torch.eq(ind_bc, 24)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        pde_right_NW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_NW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_NW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_right_NW_1_re = torch.zeros(pde_right_NW_1_le.shape[0], 1)
        pde_right_NW_2_re = torch.zeros(pde_right_NW_2_le.shape[0], 1)
        pde_right_NW_3_re = torch.zeros(pde_right_NW_3_le.shape[0], 1)
        pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re = pde_right_NW_1_re.to(device), pde_right_NW_2_re.to(device), pde_right_NW_3_re.to(device)
        pde_right_NW_4_le = pC
        pde_right_NW_4_re = pC_bc_t


        ### ----------right_S---------- ###  25
        _C = torch.eq(ind_bc, 25)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uE, vE, pE = uC, vC, -pC
        uS, vS, pS = -uC, -vC, pC

        # compute PDE matrix
        pde_right_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_right_S_1_re = torch.zeros(pde_right_S_1_le.shape[0], 1)
        pde_right_S_2_re = torch.zeros(pde_right_S_2_le.shape[0], 1)
        pde_right_S_3_re = torch.zeros(pde_right_S_3_le.shape[0], 1)
        pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re = pde_right_S_1_re.to(device), pde_right_S_2_re.to(device), pde_right_S_3_re.to(device)
        pde_right_S_4_le = pC
        pde_right_S_4_re = pC_bc_t


        ### ----------right_SW---------- ###  26
        _C = torch.eq(ind_bc, 26)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
        u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uS, vS, pS = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        pde_right_SW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_SW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_SW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        pde_right_SW_1_re = torch.zeros(pde_right_SW_1_le.shape[0], 1)
        pde_right_SW_2_re = torch.zeros(pde_right_SW_2_le.shape[0], 1)
        pde_right_SW_3_re = torch.zeros(pde_right_SW_3_le.shape[0], 1)
        pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re = pde_right_SW_1_re.to(device), pde_right_SW_2_re.to(device), pde_right_SW_3_re.to(device)
        pde_right_SW_4_le = pC
        pde_right_SW_4_re = pC_bc_t


        # lamda_bc = 14

        # lamda_bc = 25  # kernel 5
        lamda_bc = 3

        # lamda_bc = 1
        # lamda_bc = 5  # kernel 7

        # pde_M_normal_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le], axis=0)
        # pde_M_bc_le = lamda_bc*torch.cat([pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
        #                                   pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
        #                                   pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
        #                                   pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
        #                                   pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
        #                                   pde_left_1_le, pde_left_2_le, pde_left_3_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, \
        #                                   pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, \
        #                                   pde_right_1_le, pde_right_2_le, pde_right_3_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, \
        #                                   pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le], axis=0)
        # pde_M_le = torch.cat([pde_M_normal_le, pde_M_bc_le], axis=0)

        # pde_M_normal_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re], axis=0)
        # pde_M_bc_re = lamda_bc*torch.cat([pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
        #                                   pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
        #                                   pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
        #                                   pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
        #                                   pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
        #                                   pde_left_1_re, pde_left_2_re, pde_left_3_re, pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re, pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re, \
        #                                   pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re, pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re, pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re, \
        #                                   pde_right_1_re, pde_right_2_re, pde_right_3_re, pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re, pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re, \
        #                                   pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re, pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re, pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re], axis=0)
        # pde_M_re = torch.cat([pde_M_normal_re, pde_M_bc_re], axis=0)


        # pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
        #                       pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
        #                       pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
        #                       pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
        #                       pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
        #                       lamda_bc*pde_left_1_le, lamda_bc*pde_left_2_le, lamda_bc*pde_left_3_le, lamda_bc*pde_left_E_1_le, lamda_bc*pde_left_E_2_le, lamda_bc*pde_left_E_3_le, lamda_bc*pde_left_N_1_le, lamda_bc*pde_left_N_2_le, lamda_bc*pde_left_N_3_le, \
        #                       lamda_bc*pde_left_NE_1_le, lamda_bc*pde_left_NE_2_le, lamda_bc*pde_left_NE_3_le, lamda_bc*pde_left_S_1_le, lamda_bc*pde_left_S_2_le, lamda_bc*pde_left_S_3_le, lamda_bc*pde_left_SE_1_le, lamda_bc*pde_left_SE_2_le, lamda_bc*pde_left_SE_3_le, \
        #                       lamda_bc*pde_right_1_le, lamda_bc*pde_right_2_le, lamda_bc*pde_right_3_le, lamda_bc*pde_right_W_1_le, lamda_bc*pde_right_W_2_le, lamda_bc*pde_right_W_3_le, lamda_bc*pde_right_N_1_le, lamda_bc*pde_right_N_2_le, lamda_bc*pde_right_N_3_le, \
        #                       lamda_bc*pde_right_NW_1_le, lamda_bc*pde_right_NW_2_le, lamda_bc*pde_right_NW_3_le, lamda_bc*pde_right_S_1_le, lamda_bc*pde_right_S_2_le, lamda_bc*pde_right_S_3_le, lamda_bc*pde_right_SW_1_le, lamda_bc*pde_right_SW_2_le, lamda_bc*pde_right_SW_3_le], axis=0)


        # pde_M_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re, pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
        #                       pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
        #                       pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
        #                       pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
        #                       pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
        #                       lamda_bc*pde_left_1_re, lamda_bc*pde_left_2_re, lamda_bc*pde_left_3_re, lamda_bc*pde_left_E_1_re, lamda_bc*pde_left_E_2_re, lamda_bc*pde_left_E_3_re, lamda_bc*pde_left_N_1_re, lamda_bc*pde_left_N_2_re, lamda_bc*pde_left_N_3_re, \
        #                       lamda_bc*pde_left_NE_1_re, lamda_bc*pde_left_NE_2_re, lamda_bc*pde_left_NE_3_re, lamda_bc*pde_left_S_1_re, lamda_bc*pde_left_S_2_re, lamda_bc*pde_left_S_3_re, lamda_bc*pde_left_SE_1_re, lamda_bc*pde_left_SE_2_re, lamda_bc*pde_left_SE_3_re, \
        #                       lamda_bc*pde_right_1_re, lamda_bc*pde_right_2_re, lamda_bc*pde_right_3_re, lamda_bc*pde_right_W_1_re, lamda_bc*pde_right_W_2_re, lamda_bc*pde_right_W_3_re, lamda_bc*pde_right_N_1_re, lamda_bc*pde_right_N_2_re, lamda_bc*pde_right_N_3_re, \
        #                       lamda_bc*pde_right_NW_1_re, lamda_bc*pde_right_NW_2_re, lamda_bc*pde_right_NW_3_re, lamda_bc*pde_right_S_1_re, lamda_bc*pde_right_S_2_re, lamda_bc*pde_right_S_3_re, lamda_bc*pde_right_SW_1_re, lamda_bc*pde_right_SW_2_re, lamda_bc*pde_right_SW_3_re], axis=0)


        pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
                            pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
                            pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
                            pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
                            pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
                            pde_left_1_le, pde_left_2_le, pde_left_3_le, lamda_bc*pde_left_4_le, lamda_bc*pde_left_5_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, lamda_bc*pde_left_E_4_le, lamda_bc*pde_left_E_5_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, lamda_bc*pde_left_N_4_le, lamda_bc*pde_left_N_5_le, \
                            pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, lamda_bc*pde_left_NE_4_le, lamda_bc*pde_left_NE_5_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, lamda_bc*pde_left_S_4_le, lamda_bc*pde_left_S_5_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, lamda_bc*pde_left_SE_4_le, lamda_bc*pde_left_SE_5_le, \
                            pde_right_1_le, pde_right_2_le, pde_right_3_le, lamda_bc*pde_right_4_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, lamda_bc*pde_right_W_4_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, lamda_bc*pde_right_N_4_le, \
                            pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, lamda_bc*pde_right_NW_4_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, lamda_bc*pde_right_S_4_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le, lamda_bc*pde_right_SW_4_le], axis=0)


        pde_M_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re, pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
                            pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
                            pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
                            pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
                            pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
                            pde_left_1_re, pde_left_2_re, pde_left_3_re, lamda_bc*pde_left_4_re, lamda_bc*pde_left_5_re, pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re, lamda_bc*pde_left_E_4_re, lamda_bc*pde_left_E_5_re, pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re, lamda_bc*pde_left_N_4_re, lamda_bc*pde_left_N_5_re,  \
                            pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re, lamda_bc*pde_left_NE_4_re, lamda_bc*pde_left_NE_5_re, pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re, lamda_bc*pde_left_S_4_re, lamda_bc*pde_left_S_5_re, pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re, lamda_bc*pde_left_SE_4_re, lamda_bc*pde_left_SE_5_re,  \
                            pde_right_1_re, pde_right_2_re, pde_right_3_re, lamda_bc*pde_right_4_re, pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re, lamda_bc*pde_right_W_4_re, pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re, lamda_bc*pde_right_N_4_re, \
                            pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re, lamda_bc*pde_right_NW_4_re, pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re, lamda_bc*pde_right_S_4_re, pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re, lamda_bc*pde_right_SW_4_re], axis=0)
        pde_M_le = pde_M_le.double()  # change to 64 bits
        pde_M_re = pde_M_re.double()  # change to 64 bits

        # start = time.time()
        kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re
        # end = time.time()
        # runtime = end - start

        # iteration to calculate the nonlinear term
        for i in np.arange(0):
            ### ------normal PDE------ ###  0
            _C = torch.eq(ind_bc, 0)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :] 
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters
            u0C_old, v0C_old = u0C, v0C

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_normal_1_le = u_x + v_y
            pde_normal_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_normal_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_normal_1_re = torch.zeros(pde_normal_1_le.shape[0], 1)
            # pde_normal_2_re = torch.zeros(pde_normal_2_le.shape[0], 1)
            # pde_normal_3_re = torch.zeros(pde_normal_3_le.shape[0], 1)
            # pde_normal_1_re, pde_normal_2_re, pde_normal_3_re = pde_normal_1_re.to(device), pde_normal_2_re.to(device), pde_normal_3_re.to(device)


            ### ----------W_1---------- ###  1
            _C = torch.eq(ind_bc, 1)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            uW, vW, pW = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_W_1_1_le = u_x + v_y
            pde_W_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_W_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_W_1_1_re = torch.zeros(pde_W_1_1_le.shape[0], 1)
            # pde_W_1_2_re = torch.zeros(pde_W_1_2_le.shape[0], 1)
            # pde_W_1_3_re = torch.zeros(pde_W_1_3_le.shape[0], 1)
            # pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re = pde_W_1_1_re.to(device), pde_W_1_2_re.to(device), pde_W_1_3_re.to(device)


            ### ----------S_1---------- ###  2
            _C = torch.eq(ind_bc, 2)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            uS, vS, pS = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_S_1_1_le = u_x + v_y
            pde_S_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_S_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_S_1_1_re = torch.zeros(pde_S_1_1_le.shape[0], 1)
            # pde_S_1_2_re = torch.zeros(pde_S_1_2_le.shape[0], 1)
            # pde_S_1_3_re = torch.zeros(pde_S_1_3_le.shape[0], 1)
            # pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re = pde_S_1_1_re.to(device), pde_S_1_2_re.to(device), pde_S_1_3_re.to(device)


            ### ----------E_1---------- ###  3
            _C = torch.eq(ind_bc, 3)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            uE, vE, pE = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_E_1_1_le = u_x + v_y
            pde_E_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_E_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_E_1_1_re = torch.zeros(pde_E_1_1_le.shape[0], 1)
            # pde_E_1_2_re = torch.zeros(pde_E_1_2_le.shape[0], 1)
            # pde_E_1_3_re = torch.zeros(pde_E_1_3_le.shape[0], 1)
            # pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re = pde_E_1_1_re.to(device), pde_E_1_2_re.to(device), pde_E_1_3_re.to(device)


            ### ----------N_1---------- ###  4
            _C = torch.eq(ind_bc, 4)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            uN, vN, pN = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_N_1_1_le = u_x + v_y
            pde_N_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_N_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_N_1_1_re = torch.zeros(pde_N_1_1_le.shape[0], 1)
            # pde_N_1_2_re = torch.zeros(pde_N_1_2_le.shape[0], 1)
            # pde_N_1_3_re = torch.zeros(pde_N_1_3_le.shape[0], 1)
            # pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re = pde_N_1_1_re.to(device), pde_N_1_2_re.to(device), pde_N_1_3_re.to(device)


            ### ----------WS_2---------- ###  5
            _C = torch.eq(ind_bc, 5)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            uW, vW, pW = -uC, -vC, pC
            uS, vS, pS = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_WS_2_1_le = u_x + v_y
            pde_WS_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_WS_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_WS_2_1_re = torch.zeros(pde_WS_2_1_le.shape[0], 1)
            # pde_WS_2_2_re = torch.zeros(pde_WS_2_2_le.shape[0], 1)
            # pde_WS_2_3_re = torch.zeros(pde_WS_2_3_le.shape[0], 1)
            # pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re = pde_WS_2_1_re.to(device), pde_WS_2_2_re.to(device), pde_WS_2_3_re.to(device)


            ### ----------WE_2---------- ###  6
            _C = torch.eq(ind_bc, 6)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            uW, vW, pW = -uC, -vC, pC
            uE, vE, pE = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_WE_2_1_le = u_x + v_y
            pde_WE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_WE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_WE_2_1_re = torch.zeros(pde_WE_2_1_le.shape[0], 1)
            # pde_WE_2_2_re = torch.zeros(pde_WE_2_2_le.shape[0], 1)
            # pde_WE_2_3_re = torch.zeros(pde_WE_2_3_le.shape[0], 1)
            # pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re = pde_WE_2_1_re.to(device), pde_WE_2_2_re.to(device), pde_WE_2_3_re.to(device)


            ### ----------WN_2---------- ###  7
            _C = torch.eq(ind_bc, 7)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            uW, vW, pW = -uC, -vC, pC
            uN, vN, pN = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_WN_2_1_le = u_x + v_y
            pde_WN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_WN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_WN_2_1_re = torch.zeros(pde_WN_2_1_le.shape[0], 1)
            # pde_WN_2_2_re = torch.zeros(pde_WN_2_2_le.shape[0], 1)
            # pde_WN_2_3_re = torch.zeros(pde_WN_2_3_le.shape[0], 1)
            # pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re = pde_WN_2_1_re.to(device), pde_WN_2_2_re.to(device), pde_WN_2_3_re.to(device)


            ### ----------SE_2---------- ###  8
            _C = torch.eq(ind_bc, 8)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            uS, vS, pS = -uC, -vC, pC
            uE, vE, pE = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_SE_2_1_le = u_x + v_y
            pde_SE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_SE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_SE_2_1_re = torch.zeros(pde_SE_2_1_le.shape[0], 1)
            # pde_SE_2_2_re = torch.zeros(pde_SE_2_2_le.shape[0], 1)
            # pde_SE_2_3_re = torch.zeros(pde_SE_2_3_le.shape[0], 1)
            # pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re = pde_SE_2_1_re.to(device), pde_SE_2_2_re.to(device), pde_SE_2_3_re.to(device)


            ### ----------SN_2---------- ###  9
            _C = torch.eq(ind_bc, 9)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            uS, vS, pS = -uC, -vC, pC
            uN, vN, pN = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_SN_2_1_le = u_x + v_y
            pde_SN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_SN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_SN_2_1_re = torch.zeros(pde_SN_2_1_le.shape[0], 1)
            # pde_SN_2_2_re = torch.zeros(pde_SN_2_2_le.shape[0], 1)
            # pde_SN_2_3_re = torch.zeros(pde_SN_2_3_le.shape[0], 1)
            # pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re = pde_SN_2_1_re.to(device), pde_SN_2_2_re.to(device), pde_SN_2_3_re.to(device)


            ### ----------EN_2---------- ###  10
            _C = torch.eq(ind_bc, 10)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            uE, vE, pE = -uC, -vC, pC
            uN, vN, pN = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_EN_2_1_le = u_x + v_y
            pde_EN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_EN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_EN_2_1_re = torch.zeros(pde_EN_2_1_le.shape[0], 1)
            # pde_EN_2_2_re = torch.zeros(pde_EN_2_2_le.shape[0], 1)
            # pde_EN_2_3_re = torch.zeros(pde_EN_2_3_le.shape[0], 1)
            # pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re = pde_EN_2_1_re.to(device), pde_EN_2_2_re.to(device), pde_EN_2_3_re.to(device)


            ### ----------WSE_3---------- ###  11
            _C = torch.eq(ind_bc, 11)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            uW, vW, pW = -uC, -vC, pC
            uS, vS, pS = -uC, -vC, pC
            uE, vE, pE = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_WSE_3_1_le = u_x + v_y
            pde_WSE_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_WSE_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_WSE_3_1_re = torch.zeros(pde_WSE_3_1_le.shape[0], 1)
            # pde_WSE_3_2_re = torch.zeros(pde_WSE_3_2_le.shape[0], 1)
            # pde_WSE_3_3_re = torch.zeros(pde_WSE_3_3_le.shape[0], 1)
            # pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re = pde_WSE_3_1_re.to(device), pde_WSE_3_2_re.to(device), pde_WSE_3_3_re.to(device)


            ### ----------SEN_3---------- ###  12
            _C = torch.eq(ind_bc, 12)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            uS, vS, pS = -uC, -vC, pC
            uE, vE, pE = -uC, -vC, pC
            uN, vN, pN = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_SEN_3_1_le = u_x + v_y
            pde_SEN_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_SEN_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_SEN_3_1_re = torch.zeros(pde_SEN_3_1_le.shape[0], 1)
            # pde_SEN_3_2_re = torch.zeros(pde_SEN_3_2_le.shape[0], 1)
            # pde_SEN_3_3_re = torch.zeros(pde_SEN_3_3_le.shape[0], 1)
            # pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re = pde_SEN_3_1_re.to(device), pde_SEN_3_2_re.to(device), pde_SEN_3_3_re.to(device)


            ### ----------ENW_3---------- ###  13
            _C = torch.eq(ind_bc, 13)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            uE, vE, pE = -uC, -vC, pC
            uN, vN, pN = -uC, -vC, pC
            uW, vW, pW = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_ENW_3_1_le = u_x + v_y
            pde_ENW_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_ENW_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_ENW_3_1_re = torch.zeros(pde_ENW_3_1_le.shape[0], 1)
            # pde_ENW_3_2_re = torch.zeros(pde_ENW_3_2_le.shape[0], 1)
            # pde_ENW_3_3_re = torch.zeros(pde_ENW_3_3_le.shape[0], 1)
            # pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re = pde_ENW_3_1_re.to(device), pde_ENW_3_2_re.to(device), pde_ENW_3_3_re.to(device)


            ### ----------NWS_3---------- ###  14
            _C = torch.eq(ind_bc, 14)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            uN, vN, pN = -uC, -vC, pC
            uW, vW, pW = -uC, -vC, pC
            uS, vS, pS = -uC, -vC, pC

            # numerical differentiation PDE (n-pde)
            u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
            u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
            v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
            p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

            UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
            UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

            # compute PDE matrix
            # pde_NWS_3_1_le = u_x + v_y
            pde_NWS_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
            pde_NWS_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
            # pde_NWS_3_1_re = torch.zeros(pde_NWS_3_1_le.shape[0], 1)
            # pde_NWS_3_2_re = torch.zeros(pde_NWS_3_2_le.shape[0], 1)
            # pde_NWS_3_3_re = torch.zeros(pde_NWS_3_3_le.shape[0], 1)
            # pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re = pde_NWS_3_1_re.to(device), pde_NWS_3_2_re.to(device), pde_NWS_3_3_re.to(device)


            ### ----------left---------- ###  15
            _C = torch.eq(ind_bc, 15)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (uW + uC)/2 = 1, uW = 2 - uC (not working)
            # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
            uW, vW, pW = uC, vC, pC

            # compute PDE matrix
            # pde_left_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_left_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_left_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_left_1_re = torch.zeros(pde_left_1_le.shape[0], 1)
            # pde_left_2_re = torch.zeros(pde_left_2_le.shape[0], 1)
            # pde_left_3_re = torch.zeros(pde_left_3_le.shape[0], 1)
            # pde_left_1_re, pde_left_2_re, pde_left_3_re = pde_left_1_re.to(device), pde_left_2_re.to(device), pde_left_3_re.to(device)

            # pde_left_4_le = uC
            # pde_left_5_le = vC
            # pde_left_4_re = uC_bc_t
            # pde_left_5_re = vC_bc_t


            ### ----------left_E---------- ###  16
            _C = torch.eq(ind_bc, 16)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (uW + uC)/2 = 1, uW = 2 - uC (not working)
            # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            uW, vW, pW = uC, vC, pC
            uE, vE, pE = -uC, -vC, pC

            # compute PDE matrix
            # pde_left_E_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_left_E_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_left_E_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_left_E_1_re = torch.zeros(pde_left_E_1_le.shape[0], 1)
            # pde_left_E_2_re = torch.zeros(pde_left_E_2_le.shape[0], 1)
            # pde_left_E_3_re = torch.zeros(pde_left_E_3_le.shape[0], 1)
            # pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re = pde_left_E_1_re.to(device), pde_left_E_2_re.to(device), pde_left_E_3_re.to(device)

            # pde_left_E_4_le = uC
            # pde_left_E_5_le = vC
            # pde_left_E_4_re = uC_bc_t
            # pde_left_E_5_re = vC_bc_t


            ### ----------left_N---------- ###  17
            _C = torch.eq(ind_bc, 17)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (uW + uC)/2 = 1, uW = 2 - uC (not working)
            # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            uW, vW, pW = uC, vC, pC
            uN, vN, pN = -uC, -vC, pC

            # compute PDE matrix
            # pde_left_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_left_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_left_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_left_N_1_re = torch.zeros(pde_left_N_1_le.shape[0], 1)
            # pde_left_N_2_re = torch.zeros(pde_left_N_2_le.shape[0], 1)
            # pde_left_N_3_re = torch.zeros(pde_left_N_3_le.shape[0], 1)
            # pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re = pde_left_N_1_re.to(device), pde_left_N_2_re.to(device), pde_left_N_3_re.to(device)

            # pde_left_N_4_le = uC
            # pde_left_N_5_le = vC
            # pde_left_N_4_re = uC_bc_t
            # pde_left_N_5_re = vC_bc_t


            ### ----------left_NE---------- ###  18
            _C = torch.eq(ind_bc, 18)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (uW + uC)/2 = 1, uW = 2 - uC (not working)
            # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            uW, vW, pW = uC, vC, pC
            uN, vN, pN = -uC, -vC, pC
            uE, vE, pE = -uC, -vC, pC

            # compute PDE matrix
            # pde_left_NE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_left_NE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_left_NE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_left_NE_1_re = torch.zeros(pde_left_NE_1_le.shape[0], 1)
            # pde_left_NE_2_re = torch.zeros(pde_left_NE_2_le.shape[0], 1)
            # pde_left_NE_3_re = torch.zeros(pde_left_NE_3_le.shape[0], 1)
            # pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re = pde_left_NE_1_re.to(device), pde_left_NE_2_re.to(device), pde_left_NE_3_re.to(device)

            # pde_left_NE_4_le = uC
            # pde_left_NE_5_le = vC
            # pde_left_NE_4_re = uC_bc_t
            # pde_left_NE_5_re = vC_bc_t


            ### ----------left_S---------- ###  19
            _C = torch.eq(ind_bc, 19)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (uW + uC)/2 = 1, uW = 2 - uC (not working)
            # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            uW, vW, pW = uC, vC, pC
            uS, vS, pS = -uC, -vC, pC

            # compute PDE matrix
            # pde_left_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_left_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_left_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_left_S_1_re = torch.zeros(pde_left_S_1_le.shape[0], 1)
            # pde_left_S_2_re = torch.zeros(pde_left_S_2_le.shape[0], 1)
            # pde_left_S_3_re = torch.zeros(pde_left_S_3_le.shape[0], 1)
            # pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re = pde_left_S_1_re.to(device), pde_left_S_2_re.to(device), pde_left_S_3_re.to(device)

            # pde_left_S_4_le = uC
            # pde_left_S_5_le = vC
            # pde_left_S_4_re = uC_bc_t
            # pde_left_S_5_re = vC_bc_t


            ### ----------left_SE---------- ###  20
            _C = torch.eq(ind_bc, 20)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (uW + uC)/2 = 1, uW = 2 - uC (not working)
            # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
            uW, vW, pW = uC, vC, pC
            uS, vS, pS = -uC, -vC, pC
            uE, vE, pE = -uC, -vC, pC

            # compute PDE matrix
            # pde_left_SE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_left_SE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_left_SE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_left_SE_1_re = torch.zeros(pde_left_SE_1_le.shape[0], 1)
            # pde_left_SE_2_re = torch.zeros(pde_left_SE_2_le.shape[0], 1)
            # pde_left_SE_3_re = torch.zeros(pde_left_SE_3_le.shape[0], 1)
            # pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re = pde_left_SE_1_re.to(device), pde_left_SE_2_re.to(device), pde_left_SE_3_re.to(device)

            # pde_left_SE_4_le = uC
            # pde_left_SE_5_le = vC
            # pde_left_SE_4_re = uC_bc_t
            # pde_left_SE_5_re = vC_bc_t


            ### ----------right---------- ###  21
            _C = torch.eq(ind_bc, 21)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE = M_u[_E, :], M_v[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (pE + pC)/2 = 0 (not working)
            # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
            uE, vE, pE = uC, vC, -pC

            # compute PDE matrix
            # pde_right_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_right_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_right_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_right_1_re = torch.zeros(pde_right_1_le.shape[0], 1)
            # pde_right_2_re = torch.zeros(pde_right_2_le.shape[0], 1)
            # pde_right_3_re = torch.zeros(pde_right_3_le.shape[0], 1)
            # pde_right_1_re, pde_right_2_re, pde_right_3_re = pde_right_1_re.to(device), pde_right_2_re.to(device), pde_right_3_re.to(device)
            # pde_right_4_le = pC
            # pde_right_4_re = pC_bc_t


            ### ----------right_W---------- ###  22
            _C = torch.eq(ind_bc, 22)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE = M_u[_E, :], M_v[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (pE + pC)/2 = 0  (not working)
            # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            uE, vE, pE = uC, vC, -pC
            uW, vW, pW = -uC, -vC, pC

            # compute PDE matrix
            # pde_right_W_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_right_W_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_right_W_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_right_W_1_re = torch.zeros(pde_right_W_1_le.shape[0], 1)
            # pde_right_W_2_re = torch.zeros(pde_right_W_2_le.shape[0], 1)
            # pde_right_W_3_re = torch.zeros(pde_right_W_3_le.shape[0], 1)
            # pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re = pde_right_W_1_re.to(device), pde_right_W_2_re.to(device), pde_right_W_3_re.to(device)
            # pde_right_W_4_le = pC
            # pde_right_W_4_re = pC_bc_t


            ### ----------right_N---------- ###  23
            _C = torch.eq(ind_bc, 23)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE = M_u[_E, :], M_v[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (pE + pC)/2 = 0  (not working)
            # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            uE, vE, pE = uC, vC, -pC
            uN, vN, pN = -uC, -vC, pC

            # compute PDE matrix
            # pde_right_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_right_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_right_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_right_N_1_re = torch.zeros(pde_right_N_1_le.shape[0], 1)
            # pde_right_N_2_re = torch.zeros(pde_right_N_2_le.shape[0], 1)
            # pde_right_N_3_re = torch.zeros(pde_right_N_3_le.shape[0], 1)
            # pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re = pde_right_N_1_re.to(device), pde_right_N_2_re.to(device), pde_right_N_3_re.to(device)
            # pde_right_N_4_le = pC
            # pde_right_N_4_re = pC_bc_t


            ### ----------right_NW---------- ###  24
            _C = torch.eq(ind_bc, 24)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE = M_u[_E, :], M_v[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (pE + pC)/2 = 0  (not working)
            # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
            # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            uE, vE, pE = uC, vC, -pC
            uN, vN, pN = -uC, -vC, pC
            uW, vW, pW = -uC, -vC, pC

            # compute PDE matrix
            # pde_right_NW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_right_NW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_right_NW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_right_NW_1_re = torch.zeros(pde_right_NW_1_le.shape[0], 1)
            # pde_right_NW_2_re = torch.zeros(pde_right_NW_2_le.shape[0], 1)
            # pde_right_NW_3_re = torch.zeros(pde_right_NW_3_le.shape[0], 1)
            # pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re = pde_right_NW_1_re.to(device), pde_right_NW_2_re.to(device), pde_right_NW_3_re.to(device)
            # pde_right_NW_4_le = pC
            # pde_right_NW_4_re = pC_bc_t


            ### ----------right_S---------- ###  25
            _C = torch.eq(ind_bc, 25)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE = M_u[_E, :], M_v[_E, :]
            uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (pE + pC)/2 = 0  (not working)
            # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            uE, vE, pE = uC, vC, -pC
            uS, vS, pS = -uC, -vC, pC

            # compute PDE matrix
            # pde_right_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_right_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_right_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_right_S_1_re = torch.zeros(pde_right_S_1_le.shape[0], 1)
            # pde_right_S_2_re = torch.zeros(pde_right_S_2_le.shape[0], 1)
            # pde_right_S_3_re = torch.zeros(pde_right_S_3_le.shape[0], 1)
            # pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re = pde_right_S_1_re.to(device), pde_right_S_2_re.to(device), pde_right_S_3_re.to(device)
            # pde_right_S_4_le = pC
            # pde_right_S_4_re = pC_bc_t


            ### ----------right_SW---------- ###  26
            _C = torch.eq(ind_bc, 26)
            _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
            _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
            _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
            _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

            # flatten the index
            _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
            _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

            uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
            # uE, vE = M_u[_E, :], M_v[_E, :]
            # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
            uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
            # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

            u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

            # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

            # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

            # (pE + pC)/2 = 0  (not working)
            # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
            # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
            # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
            uE, vE, pE = uC, vC, -pC
            uS, vS, pS = -uC, -vC, pC
            uW, vW, pW = -uC, -vC, pC

            # compute PDE matrix
            # pde_right_SW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
            pde_right_SW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
            pde_right_SW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
            # pde_right_SW_1_re = torch.zeros(pde_right_SW_1_le.shape[0], 1)
            # pde_right_SW_2_re = torch.zeros(pde_right_SW_2_le.shape[0], 1)
            # pde_right_SW_3_re = torch.zeros(pde_right_SW_3_le.shape[0], 1)
            # pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re = pde_right_SW_1_re.to(device), pde_right_SW_2_re.to(device), pde_right_SW_3_re.to(device)
            # pde_right_SW_4_le = pC
            # pde_right_SW_4_re = pC_bc_t



            pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
                                pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
                                pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
                                pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
                                pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
                                pde_left_1_le, pde_left_2_le, pde_left_3_le, lamda_bc*pde_left_4_le, lamda_bc*pde_left_5_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, lamda_bc*pde_left_E_4_le, lamda_bc*pde_left_E_5_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, lamda_bc*pde_left_N_4_le, lamda_bc*pde_left_N_5_le, \
                                pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, lamda_bc*pde_left_NE_4_le, lamda_bc*pde_left_NE_5_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, lamda_bc*pde_left_S_4_le, lamda_bc*pde_left_S_5_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, lamda_bc*pde_left_SE_4_le, lamda_bc*pde_left_SE_5_le, \
                                pde_right_1_le, pde_right_2_le, pde_right_3_le, lamda_bc*pde_right_4_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, lamda_bc*pde_right_W_4_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, lamda_bc*pde_right_N_4_le, \
                                pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, lamda_bc*pde_right_NW_4_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, lamda_bc*pde_right_S_4_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le, lamda_bc*pde_right_SW_4_le], axis=0)

            pde_M_le = pde_M_le.double()  # change to 64 bits
            pde_M_re = pde_M_re.double()  # change to 64 bits
            kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

            # test error
            _C = torch.eq(ind_bc, 0)
            _C = _C.reshape(_C.shape[0], -1)
            _C = _C.flatten()
            uC = M_u[_C, :]
            u0C_new = uC @ kernel_parameters
            error = torch.mean((u0C_new - u0C_old)**2)
            ttt = 1


        # pseudo inverse prediction results
        M_u, M_v, M_p = M_u.double(), M_v.double(), M_p.double()
        u, v, p = M_u @ kernel_parameters, M_v @ kernel_parameters, M_p @ kernel_parameters
        u, v, p = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1), p.reshape(inputs0.shape[0], -1)
        u, v, p = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), p.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
        u, v, p = torch.unsqueeze(u, axis=1), torch.unsqueeze(v, axis=1), torch.unsqueeze(p, axis=1)
        outputs_pseudo_inverse = torch.concat([u, v, p], axis=1)
        outputs_pseudo_inverse = outputs_pseudo_inverse.float()
        # outputs_pseudo_inverse = outputs_pseudo_inverse.requires_grad_(True)

        ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

        return outputs_pseudo_inverse, ssr


    def forward(self, x, input0):
        output, ssr = self.CNN_PINN_Pseudo_Inverse_2D_NS_DirichletBC(x, input0)
        return output, ssr
   






def CNN_PINN_Pseudo_Inverse_2D_NS_DirichletBC(inputs, inputs0, conv2d_head):

    inputs_conv2d = inputs

    n_kernel_size = 7
    n_p = int(np.floor(n_kernel_size/2))
    # kernel size: 1*1, 3*3, 5*5, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

    # computational boundary
    # x_l, x_u, y_l, y_u = 0, 2, 0, 1
    x_l, x_u, y_l, y_u = 0, 2, 0, 2

    # dx, dy
    # n_x, n_y = 64, 64
    n_x, n_y = 128, 128
    # n_x, n_y = 512, 512
    dx = (x_u - x_l) / n_x  #/ 2
    dy = (y_u - y_l) / n_y  #/ 2

    x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
    y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)
    x_grid, y_grid = np.meshgrid(x, y)

    # n_padding = int((inputs0.shape[2] - n_x)/2)

    device = inputs.device

    x_ind_bc = inputs0[0, 0, :, :]

    # pad BC indicator
    # x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=0)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)

    inputs_np = inputs.detach().cpu().numpy()

    M0 = np.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_conv2d.shape[0]):
        for _x in np.arange(inputs_conv2d.shape[2]):
            for _y in np.arange(inputs_conv2d.shape[3]):
                M0_part = inputs_np[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                M0_part = M0_part.reshape(M0_part.shape[0], M0_part.shape[1], -1)
                M0_part = M0_part.reshape(M0_part.shape[0], -1)
                M0[ni:ni+1, :] = M0_part
                ni = ni + 1

    M_u = np.hstack([M0, np.zeros_like(M0), np.zeros_like(M0)])
    M_v = np.hstack([np.zeros_like(M0), M0, np.zeros_like(M0)])
    M_p = np.hstack([np.zeros_like(M0), np.zeros_like(M0), M0])
    M_u, M_v, M_p = torch.tensor(M_u).to(device), torch.tensor(M_v).to(device), torch.tensor(M_p).to(device)

    # get BC
    # M_u_bc, M_v_bc, M_p_bc = inputs0[:, 1, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 2, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 3, :, :].reshape(inputs0.shape[0], -1)
    # M_u_bc, M_v_bc, M_p_bc = M_u_bc.flatten(), M_v_bc.flatten(), M_p_bc.flatten()

    # M_u_bc_t = M_v_bc_t = M_p_bc_t = torch.zeros((1, 128, 128))
    M_u_bc_t = M_v_bc_t = M_p_bc_t = torch.zeros((1, n_x, n_y))
    index_u = torch.eq(ind_bc, 15) | torch.eq(ind_bc, 16) | torch.eq(ind_bc, 17) | torch.eq(ind_bc, 18) | torch.eq(ind_bc, 19) | torch.eq(ind_bc, 20)
    M_u_bc_t[index_u] = 1
    M_u_bc_t, M_v_bc_t, M_p_bc_t = M_u_bc_t.reshape(1, -1), M_v_bc_t.reshape(1, -1), M_p_bc_t.reshape(1, -1)
    M_u_bc_t, M_v_bc_t, M_p_bc_t = M_u_bc_t.flatten().to(device), M_v_bc_t.flatten().to(device), M_p_bc_t.flatten().to(device)

    # first iteration
    reg = 1e-4  # kernel 7 best
    # reg = 6e-5
    # reg = 1.83673469e+03  # kernel 5

    ### ------normal PDE------ ###  0
    _C = torch.eq(ind_bc, 0)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :] 
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    Re = 500
    # Re = 0.1
    pde_normal_1_le = u_x + v_y
    pde_normal_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_normal_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_normal_1_re = torch.zeros(pde_normal_1_le.shape[0], 1, requires_grad=True)
    pde_normal_2_re = torch.zeros(pde_normal_2_le.shape[0], 1, requires_grad=True)
    pde_normal_3_re = torch.zeros(pde_normal_3_le.shape[0], 1, requires_grad=True)
    pde_normal_1_re, pde_normal_2_re, pde_normal_3_re = pde_normal_1_re.to(device), pde_normal_2_re.to(device), pde_normal_3_re.to(device)


    ### ----------W_1---------- ###  1
    _C = torch.eq(ind_bc, 1)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_W_1_1_le = u_x + v_y
    pde_W_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_W_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_W_1_1_re = torch.zeros(pde_W_1_1_le.shape[0], 1)
    pde_W_1_2_re = torch.zeros(pde_W_1_2_le.shape[0], 1)
    pde_W_1_3_re = torch.zeros(pde_W_1_3_le.shape[0], 1)
    pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re = pde_W_1_1_re.to(device), pde_W_1_2_re.to(device), pde_W_1_3_re.to(device)


    ### ----------S_1---------- ###  2
    _C = torch.eq(ind_bc, 2)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_S_1_1_le = u_x + v_y
    pde_S_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_S_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_S_1_1_re = torch.zeros(pde_S_1_1_le.shape[0], 1)
    pde_S_1_2_re = torch.zeros(pde_S_1_2_le.shape[0], 1)
    pde_S_1_3_re = torch.zeros(pde_S_1_3_le.shape[0], 1)
    pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re = pde_S_1_1_re.to(device), pde_S_1_2_re.to(device), pde_S_1_3_re.to(device)


    ### ----------E_1---------- ###  3
    _C = torch.eq(ind_bc, 3)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_E_1_1_le = u_x + v_y
    pde_E_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_E_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_E_1_1_re = torch.zeros(pde_E_1_1_le.shape[0], 1)
    pde_E_1_2_re = torch.zeros(pde_E_1_2_le.shape[0], 1)
    pde_E_1_3_re = torch.zeros(pde_E_1_3_le.shape[0], 1)
    pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re = pde_E_1_1_re.to(device), pde_E_1_2_re.to(device), pde_E_1_3_re.to(device)


    ### ----------N_1---------- ###  4
    _C = torch.eq(ind_bc, 4)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_N_1_1_le = u_x + v_y
    pde_N_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_N_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_N_1_1_re = torch.zeros(pde_N_1_1_le.shape[0], 1)
    pde_N_1_2_re = torch.zeros(pde_N_1_2_le.shape[0], 1)
    pde_N_1_3_re = torch.zeros(pde_N_1_3_le.shape[0], 1)
    pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re = pde_N_1_1_re.to(device), pde_N_1_2_re.to(device), pde_N_1_3_re.to(device)


    ### ----------WS_2---------- ###  5
    _C = torch.eq(ind_bc, 5)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WS_2_1_le = u_x + v_y
    pde_WS_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WS_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WS_2_1_re = torch.zeros(pde_WS_2_1_le.shape[0], 1)
    pde_WS_2_2_re = torch.zeros(pde_WS_2_2_le.shape[0], 1)
    pde_WS_2_3_re = torch.zeros(pde_WS_2_3_le.shape[0], 1)
    pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re = pde_WS_2_1_re.to(device), pde_WS_2_2_re.to(device), pde_WS_2_3_re.to(device)


    ### ----------WE_2---------- ###  6
    _C = torch.eq(ind_bc, 6)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WE_2_1_le = u_x + v_y
    pde_WE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WE_2_1_re = torch.zeros(pde_WE_2_1_le.shape[0], 1)
    pde_WE_2_2_re = torch.zeros(pde_WE_2_2_le.shape[0], 1)
    pde_WE_2_3_re = torch.zeros(pde_WE_2_3_le.shape[0], 1)
    pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re = pde_WE_2_1_re.to(device), pde_WE_2_2_re.to(device), pde_WE_2_3_re.to(device)


    ### ----------WN_2---------- ###  7
    _C = torch.eq(ind_bc, 7)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uW, vW, pW = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WN_2_1_le = u_x + v_y
    pde_WN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WN_2_1_re = torch.zeros(pde_WN_2_1_le.shape[0], 1)
    pde_WN_2_2_re = torch.zeros(pde_WN_2_2_le.shape[0], 1)
    pde_WN_2_3_re = torch.zeros(pde_WN_2_3_le.shape[0], 1)
    pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re = pde_WN_2_1_re.to(device), pde_WN_2_2_re.to(device), pde_WN_2_3_re.to(device)


    ### ----------SE_2---------- ###  8
    _C = torch.eq(ind_bc, 8)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SE_2_1_le = u_x + v_y
    pde_SE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SE_2_1_re = torch.zeros(pde_SE_2_1_le.shape[0], 1)
    pde_SE_2_2_re = torch.zeros(pde_SE_2_2_le.shape[0], 1)
    pde_SE_2_3_re = torch.zeros(pde_SE_2_3_le.shape[0], 1)
    pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re = pde_SE_2_1_re.to(device), pde_SE_2_2_re.to(device), pde_SE_2_3_re.to(device)


    ### ----------SN_2---------- ###  9
    _C = torch.eq(ind_bc, 9)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SN_2_1_le = u_x + v_y
    pde_SN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SN_2_1_re = torch.zeros(pde_SN_2_1_le.shape[0], 1)
    pde_SN_2_2_re = torch.zeros(pde_SN_2_2_le.shape[0], 1)
    pde_SN_2_3_re = torch.zeros(pde_SN_2_3_le.shape[0], 1)
    pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re = pde_SN_2_1_re.to(device), pde_SN_2_2_re.to(device), pde_SN_2_3_re.to(device)


    ### ----------EN_2---------- ###  10
    _C = torch.eq(ind_bc, 10)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_EN_2_1_le = u_x + v_y
    pde_EN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_EN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_EN_2_1_re = torch.zeros(pde_EN_2_1_le.shape[0], 1)
    pde_EN_2_2_re = torch.zeros(pde_EN_2_2_le.shape[0], 1)
    pde_EN_2_3_re = torch.zeros(pde_EN_2_3_le.shape[0], 1)
    pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re = pde_EN_2_1_re.to(device), pde_EN_2_2_re.to(device), pde_EN_2_3_re.to(device)


    ### ----------WSE_3---------- ###  11
    _C = torch.eq(ind_bc, 11)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WSE_3_1_le = u_x + v_y
    pde_WSE_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WSE_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WSE_3_1_re = torch.zeros(pde_WSE_3_1_le.shape[0], 1)
    pde_WSE_3_2_re = torch.zeros(pde_WSE_3_2_le.shape[0], 1)
    pde_WSE_3_3_re = torch.zeros(pde_WSE_3_3_le.shape[0], 1)
    pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re = pde_WSE_3_1_re.to(device), pde_WSE_3_2_re.to(device), pde_WSE_3_3_re.to(device)


    ### ----------SEN_3---------- ###  12
    _C = torch.eq(ind_bc, 12)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SEN_3_1_le = u_x + v_y
    pde_SEN_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SEN_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SEN_3_1_re = torch.zeros(pde_SEN_3_1_le.shape[0], 1)
    pde_SEN_3_2_re = torch.zeros(pde_SEN_3_2_le.shape[0], 1)
    pde_SEN_3_3_re = torch.zeros(pde_SEN_3_3_le.shape[0], 1)
    pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re = pde_SEN_3_1_re.to(device), pde_SEN_3_2_re.to(device), pde_SEN_3_3_re.to(device)


    ### ----------ENW_3---------- ###  13
    _C = torch.eq(ind_bc, 13)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_ENW_3_1_le = u_x + v_y
    pde_ENW_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_ENW_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_ENW_3_1_re = torch.zeros(pde_ENW_3_1_le.shape[0], 1)
    pde_ENW_3_2_re = torch.zeros(pde_ENW_3_2_le.shape[0], 1)
    pde_ENW_3_3_re = torch.zeros(pde_ENW_3_3_le.shape[0], 1)
    pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re = pde_ENW_3_1_re.to(device), pde_ENW_3_2_re.to(device), pde_ENW_3_3_re.to(device)


    ### ----------NWS_3---------- ###  14
    _C = torch.eq(ind_bc, 14)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_NWS_3_1_le = u_x + v_y
    pde_NWS_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_NWS_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_NWS_3_1_re = torch.zeros(pde_NWS_3_1_le.shape[0], 1)
    pde_NWS_3_2_re = torch.zeros(pde_NWS_3_2_le.shape[0], 1)
    pde_NWS_3_3_re = torch.zeros(pde_NWS_3_3_le.shape[0], 1)
    pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re = pde_NWS_3_1_re.to(device), pde_NWS_3_2_re.to(device), pde_NWS_3_3_re.to(device)


    ### ----------left---------- ###  15
    _C = torch.eq(ind_bc, 15)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    uW, vW, pW = uC, vC, pC

    # compute PDE matrix
    pde_left_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_1_re = torch.zeros(pde_left_1_le.shape[0], 1)
    pde_left_2_re = torch.zeros(pde_left_2_le.shape[0], 1)
    pde_left_3_re = torch.zeros(pde_left_3_le.shape[0], 1)
    pde_left_1_re, pde_left_2_re, pde_left_3_re = pde_left_1_re.to(device), pde_left_2_re.to(device), pde_left_3_re.to(device)

    pde_left_4_le = uC
    pde_left_5_le = vC
    pde_left_4_re = uC_bc_t
    pde_left_5_re = vC_bc_t


    ### ----------left_E---------- ###  16
    _C = torch.eq(ind_bc, 16)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_E_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_E_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_E_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_E_1_re = torch.zeros(pde_left_E_1_le.shape[0], 1)
    pde_left_E_2_re = torch.zeros(pde_left_E_2_le.shape[0], 1)
    pde_left_E_3_re = torch.zeros(pde_left_E_3_le.shape[0], 1)
    pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re = pde_left_E_1_re.to(device), pde_left_E_2_re.to(device), pde_left_E_3_re.to(device)

    pde_left_E_4_le = uC
    pde_left_E_5_le = vC
    pde_left_E_4_re = uC_bc_t
    pde_left_E_5_re = vC_bc_t


    ### ----------left_N---------- ###  17
    _C = torch.eq(ind_bc, 17)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uW, vW, pW = uC, vC, pC
    uN, vN, pN = -uC, -vC, pC

    # compute PDE matrix
    pde_left_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_N_1_re = torch.zeros(pde_left_N_1_le.shape[0], 1)
    pde_left_N_2_re = torch.zeros(pde_left_N_2_le.shape[0], 1)
    pde_left_N_3_re = torch.zeros(pde_left_N_3_le.shape[0], 1)
    pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re = pde_left_N_1_re.to(device), pde_left_N_2_re.to(device), pde_left_N_3_re.to(device)

    pde_left_N_4_le = uC
    pde_left_N_5_le = vC
    pde_left_N_4_re = uC_bc_t
    pde_left_N_5_re = vC_bc_t


    ### ----------left_NE---------- ###  18
    _C = torch.eq(ind_bc, 18)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uN, vN, pN = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_NE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_NE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_NE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_NE_1_re = torch.zeros(pde_left_NE_1_le.shape[0], 1)
    pde_left_NE_2_re = torch.zeros(pde_left_NE_2_le.shape[0], 1)
    pde_left_NE_3_re = torch.zeros(pde_left_NE_3_le.shape[0], 1)
    pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re = pde_left_NE_1_re.to(device), pde_left_NE_2_re.to(device), pde_left_NE_3_re.to(device)

    pde_left_NE_4_le = uC
    pde_left_NE_5_le = vC
    pde_left_NE_4_re = uC_bc_t
    pde_left_NE_5_re = vC_bc_t


    ### ----------left_S---------- ###  19
    _C = torch.eq(ind_bc, 19)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uW, vW, pW = uC, vC, pC
    uS, vS, pS = -uC, -vC, pC

    # compute PDE matrix
    pde_left_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_S_1_re = torch.zeros(pde_left_S_1_le.shape[0], 1)
    pde_left_S_2_re = torch.zeros(pde_left_S_2_le.shape[0], 1)
    pde_left_S_3_re = torch.zeros(pde_left_S_3_le.shape[0], 1)
    pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re = pde_left_S_1_re.to(device), pde_left_S_2_re.to(device), pde_left_S_3_re.to(device)

    pde_left_S_4_le = uC
    pde_left_S_5_le = vC
    pde_left_S_4_re = uC_bc_t
    pde_left_S_5_re = vC_bc_t


    ### ----------left_SE---------- ###  20
    _C = torch.eq(ind_bc, 20)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_SE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_SE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_SE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_SE_1_re = torch.zeros(pde_left_SE_1_le.shape[0], 1)
    pde_left_SE_2_re = torch.zeros(pde_left_SE_2_le.shape[0], 1)
    pde_left_SE_3_re = torch.zeros(pde_left_SE_3_le.shape[0], 1)
    pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re = pde_left_SE_1_re.to(device), pde_left_SE_2_re.to(device), pde_left_SE_3_re.to(device)

    pde_left_SE_4_le = uC
    pde_left_SE_5_le = vC
    pde_left_SE_4_re = uC_bc_t
    pde_left_SE_5_re = vC_bc_t


    ### ----------right---------- ###  21
    _C = torch.eq(ind_bc, 21)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0 (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    uE, vE, pE = uC, vC, -pC

    # compute PDE matrix
    pde_right_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_1_re = torch.zeros(pde_right_1_le.shape[0], 1)
    pde_right_2_re = torch.zeros(pde_right_2_le.shape[0], 1)
    pde_right_3_re = torch.zeros(pde_right_3_le.shape[0], 1)
    pde_right_1_re, pde_right_2_re, pde_right_3_re = pde_right_1_re.to(device), pde_right_2_re.to(device), pde_right_3_re.to(device)
    pde_right_4_le = pC
    pde_right_4_re = pC_bc_t


    ### ----------right_W---------- ###  22
    _C = torch.eq(ind_bc, 22)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_W_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_W_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_W_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_W_1_re = torch.zeros(pde_right_W_1_le.shape[0], 1)
    pde_right_W_2_re = torch.zeros(pde_right_W_2_le.shape[0], 1)
    pde_right_W_3_re = torch.zeros(pde_right_W_3_le.shape[0], 1)
    pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re = pde_right_W_1_re.to(device), pde_right_W_2_re.to(device), pde_right_W_3_re.to(device)
    pde_right_W_4_le = pC
    pde_right_W_4_re = pC_bc_t


    ### ----------right_N---------- ###  23
    _C = torch.eq(ind_bc, 23)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uE, vE, pE = uC, vC, -pC
    uN, vN, pN = -uC, -vC, pC

    # compute PDE matrix
    pde_right_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_N_1_re = torch.zeros(pde_right_N_1_le.shape[0], 1)
    pde_right_N_2_re = torch.zeros(pde_right_N_2_le.shape[0], 1)
    pde_right_N_3_re = torch.zeros(pde_right_N_3_le.shape[0], 1)
    pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re = pde_right_N_1_re.to(device), pde_right_N_2_re.to(device), pde_right_N_3_re.to(device)
    pde_right_N_4_le = pC
    pde_right_N_4_re = pC_bc_t


    ### ----------right_NW---------- ###  24
    _C = torch.eq(ind_bc, 24)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_NW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_NW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_NW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_NW_1_re = torch.zeros(pde_right_NW_1_le.shape[0], 1)
    pde_right_NW_2_re = torch.zeros(pde_right_NW_2_le.shape[0], 1)
    pde_right_NW_3_re = torch.zeros(pde_right_NW_3_le.shape[0], 1)
    pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re = pde_right_NW_1_re.to(device), pde_right_NW_2_re.to(device), pde_right_NW_3_re.to(device)
    pde_right_NW_4_le = pC
    pde_right_NW_4_re = pC_bc_t


    ### ----------right_S---------- ###  25
    _C = torch.eq(ind_bc, 25)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uE, vE, pE = uC, vC, -pC
    uS, vS, pS = -uC, -vC, pC

    # compute PDE matrix
    pde_right_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_S_1_re = torch.zeros(pde_right_S_1_le.shape[0], 1)
    pde_right_S_2_re = torch.zeros(pde_right_S_2_le.shape[0], 1)
    pde_right_S_3_re = torch.zeros(pde_right_S_3_le.shape[0], 1)
    pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re = pde_right_S_1_re.to(device), pde_right_S_2_re.to(device), pde_right_S_3_re.to(device)
    pde_right_S_4_le = pC
    pde_right_S_4_re = pC_bc_t


    ### ----------right_SW---------- ###  26
    _C = torch.eq(ind_bc, 26)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uS, vS, pS = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_SW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_SW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_SW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_SW_1_re = torch.zeros(pde_right_SW_1_le.shape[0], 1)
    pde_right_SW_2_re = torch.zeros(pde_right_SW_2_le.shape[0], 1)
    pde_right_SW_3_re = torch.zeros(pde_right_SW_3_le.shape[0], 1)
    pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re = pde_right_SW_1_re.to(device), pde_right_SW_2_re.to(device), pde_right_SW_3_re.to(device)
    pde_right_SW_4_le = pC
    pde_right_SW_4_re = pC_bc_t


    # lamda_bc = 1
    lamda_bc = 3
    # lamda_bc = 5

    # pde_M_normal_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le], axis=0)
    # pde_M_bc_le = lamda_bc*torch.cat([pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
    #                                   pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
    #                                   pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
    #                                   pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
    #                                   pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
    #                                   pde_left_1_le, pde_left_2_le, pde_left_3_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, \
    #                                   pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, \
    #                                   pde_right_1_le, pde_right_2_le, pde_right_3_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, \
    #                                   pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le], axis=0)
    # pde_M_le = torch.cat([pde_M_normal_le, pde_M_bc_le], axis=0)

    # pde_M_normal_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re], axis=0)
    # pde_M_bc_re = lamda_bc*torch.cat([pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
    #                                   pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
    #                                   pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
    #                                   pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
    #                                   pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
    #                                   pde_left_1_re, pde_left_2_re, pde_left_3_re, pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re, pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re, \
    #                                   pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re, pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re, pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re, \
    #                                   pde_right_1_re, pde_right_2_re, pde_right_3_re, pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re, pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re, \
    #                                   pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re, pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re, pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re], axis=0)
    # pde_M_re = torch.cat([pde_M_normal_re, pde_M_bc_re], axis=0)


    # pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
    #                       pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
    #                       pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
    #                       pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
    #                       pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
    #                       lamda_bc*pde_left_1_le, lamda_bc*pde_left_2_le, lamda_bc*pde_left_3_le, lamda_bc*pde_left_E_1_le, lamda_bc*pde_left_E_2_le, lamda_bc*pde_left_E_3_le, lamda_bc*pde_left_N_1_le, lamda_bc*pde_left_N_2_le, lamda_bc*pde_left_N_3_le, \
    #                       lamda_bc*pde_left_NE_1_le, lamda_bc*pde_left_NE_2_le, lamda_bc*pde_left_NE_3_le, lamda_bc*pde_left_S_1_le, lamda_bc*pde_left_S_2_le, lamda_bc*pde_left_S_3_le, lamda_bc*pde_left_SE_1_le, lamda_bc*pde_left_SE_2_le, lamda_bc*pde_left_SE_3_le, \
    #                       lamda_bc*pde_right_1_le, lamda_bc*pde_right_2_le, lamda_bc*pde_right_3_le, lamda_bc*pde_right_W_1_le, lamda_bc*pde_right_W_2_le, lamda_bc*pde_right_W_3_le, lamda_bc*pde_right_N_1_le, lamda_bc*pde_right_N_2_le, lamda_bc*pde_right_N_3_le, \
    #                       lamda_bc*pde_right_NW_1_le, lamda_bc*pde_right_NW_2_le, lamda_bc*pde_right_NW_3_le, lamda_bc*pde_right_S_1_le, lamda_bc*pde_right_S_2_le, lamda_bc*pde_right_S_3_le, lamda_bc*pde_right_SW_1_le, lamda_bc*pde_right_SW_2_le, lamda_bc*pde_right_SW_3_le], axis=0)


    # pde_M_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re, pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
    #                       pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
    #                       pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
    #                       pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
    #                       pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
    #                       lamda_bc*pde_left_1_re, lamda_bc*pde_left_2_re, lamda_bc*pde_left_3_re, lamda_bc*pde_left_E_1_re, lamda_bc*pde_left_E_2_re, lamda_bc*pde_left_E_3_re, lamda_bc*pde_left_N_1_re, lamda_bc*pde_left_N_2_re, lamda_bc*pde_left_N_3_re, \
    #                       lamda_bc*pde_left_NE_1_re, lamda_bc*pde_left_NE_2_re, lamda_bc*pde_left_NE_3_re, lamda_bc*pde_left_S_1_re, lamda_bc*pde_left_S_2_re, lamda_bc*pde_left_S_3_re, lamda_bc*pde_left_SE_1_re, lamda_bc*pde_left_SE_2_re, lamda_bc*pde_left_SE_3_re, \
    #                       lamda_bc*pde_right_1_re, lamda_bc*pde_right_2_re, lamda_bc*pde_right_3_re, lamda_bc*pde_right_W_1_re, lamda_bc*pde_right_W_2_re, lamda_bc*pde_right_W_3_re, lamda_bc*pde_right_N_1_re, lamda_bc*pde_right_N_2_re, lamda_bc*pde_right_N_3_re, \
    #                       lamda_bc*pde_right_NW_1_re, lamda_bc*pde_right_NW_2_re, lamda_bc*pde_right_NW_3_re, lamda_bc*pde_right_S_1_re, lamda_bc*pde_right_S_2_re, lamda_bc*pde_right_S_3_re, lamda_bc*pde_right_SW_1_re, lamda_bc*pde_right_SW_2_re, lamda_bc*pde_right_SW_3_re], axis=0)


    pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
                          pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
                          pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
                          pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
                          pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
                          pde_left_1_le, pde_left_2_le, pde_left_3_le, lamda_bc*pde_left_4_le, lamda_bc*pde_left_5_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, lamda_bc*pde_left_E_4_le, lamda_bc*pde_left_E_5_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, lamda_bc*pde_left_N_4_le, lamda_bc*pde_left_N_5_le, \
                          pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, lamda_bc*pde_left_NE_4_le, lamda_bc*pde_left_NE_5_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, lamda_bc*pde_left_S_4_le, lamda_bc*pde_left_S_5_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, lamda_bc*pde_left_SE_4_le, lamda_bc*pde_left_SE_5_le, \
                          pde_right_1_le, pde_right_2_le, pde_right_3_le, lamda_bc*pde_right_4_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, lamda_bc*pde_right_W_4_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, lamda_bc*pde_right_N_4_le, \
                          pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, lamda_bc*pde_right_NW_4_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, lamda_bc*pde_right_S_4_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le, lamda_bc*pde_right_SW_4_le], axis=0)


    pde_M_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re, pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
                          pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
                          pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
                          pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
                          pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
                          pde_left_1_re, pde_left_2_re, pde_left_3_re, lamda_bc*pde_left_4_re, lamda_bc*pde_left_5_re, pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re, lamda_bc*pde_left_E_4_re, lamda_bc*pde_left_E_5_re, pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re, lamda_bc*pde_left_N_4_re, lamda_bc*pde_left_N_5_re,  \
                          pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re, lamda_bc*pde_left_NE_4_re, lamda_bc*pde_left_NE_5_re, pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re, lamda_bc*pde_left_S_4_re, lamda_bc*pde_left_S_5_re, pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re, lamda_bc*pde_left_SE_4_re, lamda_bc*pde_left_SE_5_re,  \
                          pde_right_1_re, pde_right_2_re, pde_right_3_re, lamda_bc*pde_right_4_re, pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re, lamda_bc*pde_right_W_4_re, pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re, lamda_bc*pde_right_N_4_re, \
                          pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re, lamda_bc*pde_right_NW_4_re, pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re, lamda_bc*pde_right_S_4_re, pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re, lamda_bc*pde_right_SW_4_re], axis=0)
    
    pde_M_re = pde_M_re.double()  # change to 64 bits

    pde_M_le1 = pde_M_le.detach().cpu().numpy()
    pde_M_re1 = pde_M_re.detach().cpu().numpy()

    # start = time.time()
    # kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re

    kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le1.shape[1]) + (pde_M_le1.T @ pde_M_le1)) @ pde_M_le1.T @ pde_M_re1
    kernel_parameters = torch.tensor(kernel_parameters).to(device)

    # end = time.time()
    # runtime = end - start

    # iteration to calculate the nonlinear term
    for i in np.arange(5):
        ### ------normal PDE------ ###  0
        _C = torch.eq(ind_bc, 0)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :] 
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters
        u0C_old, v0C_old = u0C, v0C

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_normal_1_le = u_x + v_y
        pde_normal_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_normal_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_normal_1_re = torch.zeros(pde_normal_1_le.shape[0], 1)
        # pde_normal_2_re = torch.zeros(pde_normal_2_le.shape[0], 1)
        # pde_normal_3_re = torch.zeros(pde_normal_3_le.shape[0], 1)
        # pde_normal_1_re, pde_normal_2_re, pde_normal_3_re = pde_normal_1_re.to(device), pde_normal_2_re.to(device), pde_normal_3_re.to(device)


        ### ----------W_1---------- ###  1
        _C = torch.eq(ind_bc, 1)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_W_1_1_le = u_x + v_y
        pde_W_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_W_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_W_1_1_re = torch.zeros(pde_W_1_1_le.shape[0], 1)
        # pde_W_1_2_re = torch.zeros(pde_W_1_2_le.shape[0], 1)
        # pde_W_1_3_re = torch.zeros(pde_W_1_3_le.shape[0], 1)
        # pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re = pde_W_1_1_re.to(device), pde_W_1_2_re.to(device), pde_W_1_3_re.to(device)


        ### ----------S_1---------- ###  2
        _C = torch.eq(ind_bc, 2)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_S_1_1_le = u_x + v_y
        pde_S_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_S_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_S_1_1_re = torch.zeros(pde_S_1_1_le.shape[0], 1)
        # pde_S_1_2_re = torch.zeros(pde_S_1_2_le.shape[0], 1)
        # pde_S_1_3_re = torch.zeros(pde_S_1_3_le.shape[0], 1)
        # pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re = pde_S_1_1_re.to(device), pde_S_1_2_re.to(device), pde_S_1_3_re.to(device)


        ### ----------E_1---------- ###  3
        _C = torch.eq(ind_bc, 3)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_E_1_1_le = u_x + v_y
        pde_E_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_E_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_E_1_1_re = torch.zeros(pde_E_1_1_le.shape[0], 1)
        # pde_E_1_2_re = torch.zeros(pde_E_1_2_le.shape[0], 1)
        # pde_E_1_3_re = torch.zeros(pde_E_1_3_le.shape[0], 1)
        # pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re = pde_E_1_1_re.to(device), pde_E_1_2_re.to(device), pde_E_1_3_re.to(device)


        ### ----------N_1---------- ###  4
        _C = torch.eq(ind_bc, 4)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_N_1_1_le = u_x + v_y
        pde_N_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_N_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_N_1_1_re = torch.zeros(pde_N_1_1_le.shape[0], 1)
        # pde_N_1_2_re = torch.zeros(pde_N_1_2_le.shape[0], 1)
        # pde_N_1_3_re = torch.zeros(pde_N_1_3_le.shape[0], 1)
        # pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re = pde_N_1_1_re.to(device), pde_N_1_2_re.to(device), pde_N_1_3_re.to(device)


        ### ----------WS_2---------- ###  5
        _C = torch.eq(ind_bc, 5)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WS_2_1_le = u_x + v_y
        pde_WS_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WS_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WS_2_1_re = torch.zeros(pde_WS_2_1_le.shape[0], 1)
        # pde_WS_2_2_re = torch.zeros(pde_WS_2_2_le.shape[0], 1)
        # pde_WS_2_3_re = torch.zeros(pde_WS_2_3_le.shape[0], 1)
        # pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re = pde_WS_2_1_re.to(device), pde_WS_2_2_re.to(device), pde_WS_2_3_re.to(device)


        ### ----------WE_2---------- ###  6
        _C = torch.eq(ind_bc, 6)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WE_2_1_le = u_x + v_y
        pde_WE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WE_2_1_re = torch.zeros(pde_WE_2_1_le.shape[0], 1)
        # pde_WE_2_2_re = torch.zeros(pde_WE_2_2_le.shape[0], 1)
        # pde_WE_2_3_re = torch.zeros(pde_WE_2_3_le.shape[0], 1)
        # pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re = pde_WE_2_1_re.to(device), pde_WE_2_2_re.to(device), pde_WE_2_3_re.to(device)


        ### ----------WN_2---------- ###  7
        _C = torch.eq(ind_bc, 7)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uW, vW, pW = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WN_2_1_le = u_x + v_y
        pde_WN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WN_2_1_re = torch.zeros(pde_WN_2_1_le.shape[0], 1)
        # pde_WN_2_2_re = torch.zeros(pde_WN_2_2_le.shape[0], 1)
        # pde_WN_2_3_re = torch.zeros(pde_WN_2_3_le.shape[0], 1)
        # pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re = pde_WN_2_1_re.to(device), pde_WN_2_2_re.to(device), pde_WN_2_3_re.to(device)


        ### ----------SE_2---------- ###  8
        _C = torch.eq(ind_bc, 8)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SE_2_1_le = u_x + v_y
        pde_SE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SE_2_1_re = torch.zeros(pde_SE_2_1_le.shape[0], 1)
        # pde_SE_2_2_re = torch.zeros(pde_SE_2_2_le.shape[0], 1)
        # pde_SE_2_3_re = torch.zeros(pde_SE_2_3_le.shape[0], 1)
        # pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re = pde_SE_2_1_re.to(device), pde_SE_2_2_re.to(device), pde_SE_2_3_re.to(device)


        ### ----------SN_2---------- ###  9
        _C = torch.eq(ind_bc, 9)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SN_2_1_le = u_x + v_y
        pde_SN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SN_2_1_re = torch.zeros(pde_SN_2_1_le.shape[0], 1)
        # pde_SN_2_2_re = torch.zeros(pde_SN_2_2_le.shape[0], 1)
        # pde_SN_2_3_re = torch.zeros(pde_SN_2_3_le.shape[0], 1)
        # pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re = pde_SN_2_1_re.to(device), pde_SN_2_2_re.to(device), pde_SN_2_3_re.to(device)


        ### ----------EN_2---------- ###  10
        _C = torch.eq(ind_bc, 10)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_EN_2_1_le = u_x + v_y
        pde_EN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_EN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_EN_2_1_re = torch.zeros(pde_EN_2_1_le.shape[0], 1)
        # pde_EN_2_2_re = torch.zeros(pde_EN_2_2_le.shape[0], 1)
        # pde_EN_2_3_re = torch.zeros(pde_EN_2_3_le.shape[0], 1)
        # pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re = pde_EN_2_1_re.to(device), pde_EN_2_2_re.to(device), pde_EN_2_3_re.to(device)


        ### ----------WSE_3---------- ###  11
        _C = torch.eq(ind_bc, 11)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WSE_3_1_le = u_x + v_y
        pde_WSE_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WSE_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WSE_3_1_re = torch.zeros(pde_WSE_3_1_le.shape[0], 1)
        # pde_WSE_3_2_re = torch.zeros(pde_WSE_3_2_le.shape[0], 1)
        # pde_WSE_3_3_re = torch.zeros(pde_WSE_3_3_le.shape[0], 1)
        # pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re = pde_WSE_3_1_re.to(device), pde_WSE_3_2_re.to(device), pde_WSE_3_3_re.to(device)


        ### ----------SEN_3---------- ###  12
        _C = torch.eq(ind_bc, 12)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SEN_3_1_le = u_x + v_y
        pde_SEN_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SEN_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SEN_3_1_re = torch.zeros(pde_SEN_3_1_le.shape[0], 1)
        # pde_SEN_3_2_re = torch.zeros(pde_SEN_3_2_le.shape[0], 1)
        # pde_SEN_3_3_re = torch.zeros(pde_SEN_3_3_le.shape[0], 1)
        # pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re = pde_SEN_3_1_re.to(device), pde_SEN_3_2_re.to(device), pde_SEN_3_3_re.to(device)


        ### ----------ENW_3---------- ###  13
        _C = torch.eq(ind_bc, 13)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_ENW_3_1_le = u_x + v_y
        pde_ENW_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_ENW_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_ENW_3_1_re = torch.zeros(pde_ENW_3_1_le.shape[0], 1)
        # pde_ENW_3_2_re = torch.zeros(pde_ENW_3_2_le.shape[0], 1)
        # pde_ENW_3_3_re = torch.zeros(pde_ENW_3_3_le.shape[0], 1)
        # pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re = pde_ENW_3_1_re.to(device), pde_ENW_3_2_re.to(device), pde_ENW_3_3_re.to(device)


        ### ----------NWS_3---------- ###  14
        _C = torch.eq(ind_bc, 14)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_NWS_3_1_le = u_x + v_y
        pde_NWS_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_NWS_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_NWS_3_1_re = torch.zeros(pde_NWS_3_1_le.shape[0], 1)
        # pde_NWS_3_2_re = torch.zeros(pde_NWS_3_2_le.shape[0], 1)
        # pde_NWS_3_3_re = torch.zeros(pde_NWS_3_3_le.shape[0], 1)
        # pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re = pde_NWS_3_1_re.to(device), pde_NWS_3_2_re.to(device), pde_NWS_3_3_re.to(device)


        ### ----------left---------- ###  15
        _C = torch.eq(ind_bc, 15)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        uW, vW, pW = uC, vC, pC

        # compute PDE matrix
        # pde_left_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_1_re = torch.zeros(pde_left_1_le.shape[0], 1)
        # pde_left_2_re = torch.zeros(pde_left_2_le.shape[0], 1)
        # pde_left_3_re = torch.zeros(pde_left_3_le.shape[0], 1)
        # pde_left_1_re, pde_left_2_re, pde_left_3_re = pde_left_1_re.to(device), pde_left_2_re.to(device), pde_left_3_re.to(device)

        # pde_left_4_le = uC
        # pde_left_5_le = vC
        # pde_left_4_re = uC_bc_t
        # pde_left_5_re = vC_bc_t


        ### ----------left_E---------- ###  16
        _C = torch.eq(ind_bc, 16)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_E_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_E_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_E_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_E_1_re = torch.zeros(pde_left_E_1_le.shape[0], 1)
        # pde_left_E_2_re = torch.zeros(pde_left_E_2_le.shape[0], 1)
        # pde_left_E_3_re = torch.zeros(pde_left_E_3_le.shape[0], 1)
        # pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re = pde_left_E_1_re.to(device), pde_left_E_2_re.to(device), pde_left_E_3_re.to(device)

        # pde_left_E_4_le = uC
        # pde_left_E_5_le = vC
        # pde_left_E_4_re = uC_bc_t
        # pde_left_E_5_re = vC_bc_t


        ### ----------left_N---------- ###  17
        _C = torch.eq(ind_bc, 17)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uW, vW, pW = uC, vC, pC
        uN, vN, pN = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_N_1_re = torch.zeros(pde_left_N_1_le.shape[0], 1)
        # pde_left_N_2_re = torch.zeros(pde_left_N_2_le.shape[0], 1)
        # pde_left_N_3_re = torch.zeros(pde_left_N_3_le.shape[0], 1)
        # pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re = pde_left_N_1_re.to(device), pde_left_N_2_re.to(device), pde_left_N_3_re.to(device)

        # pde_left_N_4_le = uC
        # pde_left_N_5_le = vC
        # pde_left_N_4_re = uC_bc_t
        # pde_left_N_5_re = vC_bc_t


        ### ----------left_NE---------- ###  18
        _C = torch.eq(ind_bc, 18)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uN, vN, pN = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_NE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_NE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_NE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_NE_1_re = torch.zeros(pde_left_NE_1_le.shape[0], 1)
        # pde_left_NE_2_re = torch.zeros(pde_left_NE_2_le.shape[0], 1)
        # pde_left_NE_3_re = torch.zeros(pde_left_NE_3_le.shape[0], 1)
        # pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re = pde_left_NE_1_re.to(device), pde_left_NE_2_re.to(device), pde_left_NE_3_re.to(device)

        # pde_left_NE_4_le = uC
        # pde_left_NE_5_le = vC
        # pde_left_NE_4_re = uC_bc_t
        # pde_left_NE_5_re = vC_bc_t


        ### ----------left_S---------- ###  19
        _C = torch.eq(ind_bc, 19)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uW, vW, pW = uC, vC, pC
        uS, vS, pS = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_S_1_re = torch.zeros(pde_left_S_1_le.shape[0], 1)
        # pde_left_S_2_re = torch.zeros(pde_left_S_2_le.shape[0], 1)
        # pde_left_S_3_re = torch.zeros(pde_left_S_3_le.shape[0], 1)
        # pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re = pde_left_S_1_re.to(device), pde_left_S_2_re.to(device), pde_left_S_3_re.to(device)

        # pde_left_S_4_le = uC
        # pde_left_S_5_le = vC
        # pde_left_S_4_re = uC_bc_t
        # pde_left_S_5_re = vC_bc_t


        ### ----------left_SE---------- ###  20
        _C = torch.eq(ind_bc, 20)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_SE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_SE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_SE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_SE_1_re = torch.zeros(pde_left_SE_1_le.shape[0], 1)
        # pde_left_SE_2_re = torch.zeros(pde_left_SE_2_le.shape[0], 1)
        # pde_left_SE_3_re = torch.zeros(pde_left_SE_3_le.shape[0], 1)
        # pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re = pde_left_SE_1_re.to(device), pde_left_SE_2_re.to(device), pde_left_SE_3_re.to(device)

        # pde_left_SE_4_le = uC
        # pde_left_SE_5_le = vC
        # pde_left_SE_4_re = uC_bc_t
        # pde_left_SE_5_re = vC_bc_t


        ### ----------right---------- ###  21
        _C = torch.eq(ind_bc, 21)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0 (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        uE, vE, pE = uC, vC, -pC

        # compute PDE matrix
        # pde_right_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_1_re = torch.zeros(pde_right_1_le.shape[0], 1)
        # pde_right_2_re = torch.zeros(pde_right_2_le.shape[0], 1)
        # pde_right_3_re = torch.zeros(pde_right_3_le.shape[0], 1)
        # pde_right_1_re, pde_right_2_re, pde_right_3_re = pde_right_1_re.to(device), pde_right_2_re.to(device), pde_right_3_re.to(device)
        # pde_right_4_le = pC
        # pde_right_4_re = pC_bc_t


        ### ----------right_W---------- ###  22
        _C = torch.eq(ind_bc, 22)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_W_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_W_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_W_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_W_1_re = torch.zeros(pde_right_W_1_le.shape[0], 1)
        # pde_right_W_2_re = torch.zeros(pde_right_W_2_le.shape[0], 1)
        # pde_right_W_3_re = torch.zeros(pde_right_W_3_le.shape[0], 1)
        # pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re = pde_right_W_1_re.to(device), pde_right_W_2_re.to(device), pde_right_W_3_re.to(device)
        # pde_right_W_4_le = pC
        # pde_right_W_4_re = pC_bc_t


        ### ----------right_N---------- ###  23
        _C = torch.eq(ind_bc, 23)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uE, vE, pE = uC, vC, -pC
        uN, vN, pN = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_N_1_re = torch.zeros(pde_right_N_1_le.shape[0], 1)
        # pde_right_N_2_re = torch.zeros(pde_right_N_2_le.shape[0], 1)
        # pde_right_N_3_re = torch.zeros(pde_right_N_3_le.shape[0], 1)
        # pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re = pde_right_N_1_re.to(device), pde_right_N_2_re.to(device), pde_right_N_3_re.to(device)
        # pde_right_N_4_le = pC
        # pde_right_N_4_re = pC_bc_t


        ### ----------right_NW---------- ###  24
        _C = torch.eq(ind_bc, 24)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_NW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_NW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_NW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_NW_1_re = torch.zeros(pde_right_NW_1_le.shape[0], 1)
        # pde_right_NW_2_re = torch.zeros(pde_right_NW_2_le.shape[0], 1)
        # pde_right_NW_3_re = torch.zeros(pde_right_NW_3_le.shape[0], 1)
        # pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re = pde_right_NW_1_re.to(device), pde_right_NW_2_re.to(device), pde_right_NW_3_re.to(device)
        # pde_right_NW_4_le = pC
        # pde_right_NW_4_re = pC_bc_t


        ### ----------right_S---------- ###  25
        _C = torch.eq(ind_bc, 25)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uE, vE, pE = uC, vC, -pC
        uS, vS, pS = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_S_1_re = torch.zeros(pde_right_S_1_le.shape[0], 1)
        # pde_right_S_2_re = torch.zeros(pde_right_S_2_le.shape[0], 1)
        # pde_right_S_3_re = torch.zeros(pde_right_S_3_le.shape[0], 1)
        # pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re = pde_right_S_1_re.to(device), pde_right_S_2_re.to(device), pde_right_S_3_re.to(device)
        # pde_right_S_4_le = pC
        # pde_right_S_4_re = pC_bc_t


        ### ----------right_SW---------- ###  26
        _C = torch.eq(ind_bc, 26)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uS, vS, pS = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_SW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_SW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_SW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_SW_1_re = torch.zeros(pde_right_SW_1_le.shape[0], 1)
        # pde_right_SW_2_re = torch.zeros(pde_right_SW_2_le.shape[0], 1)
        # pde_right_SW_3_re = torch.zeros(pde_right_SW_3_le.shape[0], 1)
        # pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re = pde_right_SW_1_re.to(device), pde_right_SW_2_re.to(device), pde_right_SW_3_re.to(device)
        # pde_right_SW_4_le = pC
        # pde_right_SW_4_re = pC_bc_t



        pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
                              pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
                              pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
                              pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
                              pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
                              pde_left_1_le, pde_left_2_le, pde_left_3_le, lamda_bc*pde_left_4_le, lamda_bc*pde_left_5_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, lamda_bc*pde_left_E_4_le, lamda_bc*pde_left_E_5_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, lamda_bc*pde_left_N_4_le, lamda_bc*pde_left_N_5_le, \
                              pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, lamda_bc*pde_left_NE_4_le, lamda_bc*pde_left_NE_5_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, lamda_bc*pde_left_S_4_le, lamda_bc*pde_left_S_5_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, lamda_bc*pde_left_SE_4_le, lamda_bc*pde_left_SE_5_le, \
                              pde_right_1_le, pde_right_2_le, pde_right_3_le, lamda_bc*pde_right_4_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, lamda_bc*pde_right_W_4_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, lamda_bc*pde_right_N_4_le, \
                              pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, lamda_bc*pde_right_NW_4_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, lamda_bc*pde_right_S_4_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le, lamda_bc*pde_right_SW_4_le], axis=0)

        pde_M_le1 = pde_M_le.detach().cpu().numpy()
        kernel_parameters = np.linalg.inv(reg*np.eye(pde_M_le1.shape[1]) + (pde_M_le1.T @ pde_M_le1)) @ pde_M_le1.T @ pde_M_re1
        kernel_parameters = torch.tensor(kernel_parameters).to(device)
        
        # kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re


        # test error
        _C = torch.eq(ind_bc, 0)
        _C = _C.reshape(_C.shape[0], -1)
        _C = _C.flatten()
        uC = M_u[_C, :]
        u0C_new = uC @ kernel_parameters
        error = torch.mean((u0C_new - u0C_old)**2)
        ttt = 1

   
    # pseudo inverse prediction results
    u, v, p = M_u @ kernel_parameters, M_v @ kernel_parameters, M_p @ kernel_parameters
    u, v, p = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1), p.reshape(inputs0.shape[0], -1)
    u, v, p = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), p.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    u, v, p = torch.unsqueeze(u, axis=1), torch.unsqueeze(v, axis=1), torch.unsqueeze(p, axis=1)
    outputs_pseudo_inverse = torch.concat([u, v, p], axis=1)
    outputs_pseudo_inverse = outputs_pseudo_inverse.float()
    # outputs_pseudo_inverse = outputs_pseudo_inverse.requires_grad_(True)

    # convolution 2d prediction results
    outputs_conv2d = conv2d_head(inputs_conv2d).float()

    ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    return outputs_pseudo_inverse, outputs_conv2d, ssr






def CNN_PINN_Pseudo_Inverse_2D_NS_DirichletBC_reg(inputs, inputs0, conv2d_head, lamda_bc, reg):

    inputs_conv2d = inputs

    n_kernel_size = 5
    n_p = int(np.floor(n_kernel_size/2))
    # kernel size: 1*1, 3*3, 5*5, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

    # computational boundary
    # x_l, x_u, y_l, y_u = 0, 2, 0, 1
    x_l, x_u, y_l, y_u = 0, 2, 0, 2

    # dx, dy
    # n_x, n_y = 64, 64
    n_x, n_y = 128, 128
    # n_x, n_y = 512, 512
    dx = (x_u - x_l) / n_x  #/ 2
    dy = (y_u - y_l) / n_y  #/ 2

    x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
    y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)
    x_grid, y_grid = np.meshgrid(x, y)

    # n_padding = int((inputs0.shape[2] - n_x)/2)

    device = inputs.device

    x_ind_bc = inputs0[0, 0, :, :]

    # pad BC indicator
    # x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=0)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)

    inputs_np = inputs.detach().cpu().numpy()

    M0 = np.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_conv2d.shape[0]):
        for _x in np.arange(inputs_conv2d.shape[2]):
            for _y in np.arange(inputs_conv2d.shape[3]):
                M0_part = inputs_np[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                M0_part = M0_part.reshape(M0_part.shape[0], M0_part.shape[1], -1)
                M0_part = M0_part.reshape(M0_part.shape[0], -1)
                M0[ni:ni+1, :] = M0_part
                ni = ni + 1

    M_u = np.hstack([M0, np.zeros_like(M0), np.zeros_like(M0)])
    M_v = np.hstack([np.zeros_like(M0), M0, np.zeros_like(M0)])
    M_p = np.hstack([np.zeros_like(M0), np.zeros_like(M0), M0])
    M_u, M_v, M_p = torch.tensor(M_u).to(device), torch.tensor(M_v).to(device), torch.tensor(M_p).to(device)

    # get BC
    # M_u_bc, M_v_bc, M_p_bc = inputs0[:, 1, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 2, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 3, :, :].reshape(inputs0.shape[0], -1)
    # M_u_bc, M_v_bc, M_p_bc = M_u_bc.flatten(), M_v_bc.flatten(), M_p_bc.flatten()

    # M_u_bc_t = M_v_bc_t = M_p_bc_t = torch.zeros((1, 128, 128))
    M_u_bc_t = M_v_bc_t = M_p_bc_t = torch.zeros((1, n_x, n_y))
    index_u = torch.eq(ind_bc, 15) | torch.eq(ind_bc, 16) | torch.eq(ind_bc, 17) | torch.eq(ind_bc, 18) | torch.eq(ind_bc, 19) | torch.eq(ind_bc, 20)
    M_u_bc_t[index_u] = 1
    M_u_bc_t, M_v_bc_t, M_p_bc_t = M_u_bc_t.reshape(1, -1), M_v_bc_t.reshape(1, -1), M_p_bc_t.reshape(1, -1)
    M_u_bc_t, M_v_bc_t, M_p_bc_t = M_u_bc_t.flatten().to(device), M_v_bc_t.flatten().to(device), M_p_bc_t.flatten().to(device)

    # first iteration
    # reg = 1e-4  # kernel 7 best
    # reg = 5e-4
    # reg = 1.83673469e+03  # kernel 5

    ### ------normal PDE------ ###  0
    _C = torch.eq(ind_bc, 0)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :] 
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    Re = 500
    # Re = 0.1
    pde_normal_1_le = u_x + v_y
    pde_normal_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_normal_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_normal_1_re = torch.zeros(pde_normal_1_le.shape[0], 1, requires_grad=True)
    pde_normal_2_re = torch.zeros(pde_normal_2_le.shape[0], 1, requires_grad=True)
    pde_normal_3_re = torch.zeros(pde_normal_3_le.shape[0], 1, requires_grad=True)
    pde_normal_1_re, pde_normal_2_re, pde_normal_3_re = pde_normal_1_re.to(device), pde_normal_2_re.to(device), pde_normal_3_re.to(device)


    ### ----------W_1---------- ###  1
    _C = torch.eq(ind_bc, 1)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_W_1_1_le = u_x + v_y
    pde_W_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_W_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_W_1_1_re = torch.zeros(pde_W_1_1_le.shape[0], 1)
    pde_W_1_2_re = torch.zeros(pde_W_1_2_le.shape[0], 1)
    pde_W_1_3_re = torch.zeros(pde_W_1_3_le.shape[0], 1)
    pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re = pde_W_1_1_re.to(device), pde_W_1_2_re.to(device), pde_W_1_3_re.to(device)


    ### ----------S_1---------- ###  2
    _C = torch.eq(ind_bc, 2)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_S_1_1_le = u_x + v_y
    pde_S_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_S_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_S_1_1_re = torch.zeros(pde_S_1_1_le.shape[0], 1)
    pde_S_1_2_re = torch.zeros(pde_S_1_2_le.shape[0], 1)
    pde_S_1_3_re = torch.zeros(pde_S_1_3_le.shape[0], 1)
    pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re = pde_S_1_1_re.to(device), pde_S_1_2_re.to(device), pde_S_1_3_re.to(device)


    ### ----------E_1---------- ###  3
    _C = torch.eq(ind_bc, 3)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_E_1_1_le = u_x + v_y
    pde_E_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_E_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_E_1_1_re = torch.zeros(pde_E_1_1_le.shape[0], 1)
    pde_E_1_2_re = torch.zeros(pde_E_1_2_le.shape[0], 1)
    pde_E_1_3_re = torch.zeros(pde_E_1_3_le.shape[0], 1)
    pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re = pde_E_1_1_re.to(device), pde_E_1_2_re.to(device), pde_E_1_3_re.to(device)


    ### ----------N_1---------- ###  4
    _C = torch.eq(ind_bc, 4)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_N_1_1_le = u_x + v_y
    pde_N_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_N_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_N_1_1_re = torch.zeros(pde_N_1_1_le.shape[0], 1)
    pde_N_1_2_re = torch.zeros(pde_N_1_2_le.shape[0], 1)
    pde_N_1_3_re = torch.zeros(pde_N_1_3_le.shape[0], 1)
    pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re = pde_N_1_1_re.to(device), pde_N_1_2_re.to(device), pde_N_1_3_re.to(device)


    ### ----------WS_2---------- ###  5
    _C = torch.eq(ind_bc, 5)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WS_2_1_le = u_x + v_y
    pde_WS_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WS_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WS_2_1_re = torch.zeros(pde_WS_2_1_le.shape[0], 1)
    pde_WS_2_2_re = torch.zeros(pde_WS_2_2_le.shape[0], 1)
    pde_WS_2_3_re = torch.zeros(pde_WS_2_3_le.shape[0], 1)
    pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re = pde_WS_2_1_re.to(device), pde_WS_2_2_re.to(device), pde_WS_2_3_re.to(device)


    ### ----------WE_2---------- ###  6
    _C = torch.eq(ind_bc, 6)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WE_2_1_le = u_x + v_y
    pde_WE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WE_2_1_re = torch.zeros(pde_WE_2_1_le.shape[0], 1)
    pde_WE_2_2_re = torch.zeros(pde_WE_2_2_le.shape[0], 1)
    pde_WE_2_3_re = torch.zeros(pde_WE_2_3_le.shape[0], 1)
    pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re = pde_WE_2_1_re.to(device), pde_WE_2_2_re.to(device), pde_WE_2_3_re.to(device)


    ### ----------WN_2---------- ###  7
    _C = torch.eq(ind_bc, 7)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uW, vW, pW = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WN_2_1_le = u_x + v_y
    pde_WN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WN_2_1_re = torch.zeros(pde_WN_2_1_le.shape[0], 1)
    pde_WN_2_2_re = torch.zeros(pde_WN_2_2_le.shape[0], 1)
    pde_WN_2_3_re = torch.zeros(pde_WN_2_3_le.shape[0], 1)
    pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re = pde_WN_2_1_re.to(device), pde_WN_2_2_re.to(device), pde_WN_2_3_re.to(device)


    ### ----------SE_2---------- ###  8
    _C = torch.eq(ind_bc, 8)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SE_2_1_le = u_x + v_y
    pde_SE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SE_2_1_re = torch.zeros(pde_SE_2_1_le.shape[0], 1)
    pde_SE_2_2_re = torch.zeros(pde_SE_2_2_le.shape[0], 1)
    pde_SE_2_3_re = torch.zeros(pde_SE_2_3_le.shape[0], 1)
    pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re = pde_SE_2_1_re.to(device), pde_SE_2_2_re.to(device), pde_SE_2_3_re.to(device)


    ### ----------SN_2---------- ###  9
    _C = torch.eq(ind_bc, 9)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SN_2_1_le = u_x + v_y
    pde_SN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SN_2_1_re = torch.zeros(pde_SN_2_1_le.shape[0], 1)
    pde_SN_2_2_re = torch.zeros(pde_SN_2_2_le.shape[0], 1)
    pde_SN_2_3_re = torch.zeros(pde_SN_2_3_le.shape[0], 1)
    pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re = pde_SN_2_1_re.to(device), pde_SN_2_2_re.to(device), pde_SN_2_3_re.to(device)


    ### ----------EN_2---------- ###  10
    _C = torch.eq(ind_bc, 10)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_EN_2_1_le = u_x + v_y
    pde_EN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_EN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_EN_2_1_re = torch.zeros(pde_EN_2_1_le.shape[0], 1)
    pde_EN_2_2_re = torch.zeros(pde_EN_2_2_le.shape[0], 1)
    pde_EN_2_3_re = torch.zeros(pde_EN_2_3_le.shape[0], 1)
    pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re = pde_EN_2_1_re.to(device), pde_EN_2_2_re.to(device), pde_EN_2_3_re.to(device)


    ### ----------WSE_3---------- ###  11
    _C = torch.eq(ind_bc, 11)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WSE_3_1_le = u_x + v_y
    pde_WSE_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WSE_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WSE_3_1_re = torch.zeros(pde_WSE_3_1_le.shape[0], 1)
    pde_WSE_3_2_re = torch.zeros(pde_WSE_3_2_le.shape[0], 1)
    pde_WSE_3_3_re = torch.zeros(pde_WSE_3_3_le.shape[0], 1)
    pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re = pde_WSE_3_1_re.to(device), pde_WSE_3_2_re.to(device), pde_WSE_3_3_re.to(device)


    ### ----------SEN_3---------- ###  12
    _C = torch.eq(ind_bc, 12)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SEN_3_1_le = u_x + v_y
    pde_SEN_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SEN_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SEN_3_1_re = torch.zeros(pde_SEN_3_1_le.shape[0], 1)
    pde_SEN_3_2_re = torch.zeros(pde_SEN_3_2_le.shape[0], 1)
    pde_SEN_3_3_re = torch.zeros(pde_SEN_3_3_le.shape[0], 1)
    pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re = pde_SEN_3_1_re.to(device), pde_SEN_3_2_re.to(device), pde_SEN_3_3_re.to(device)


    ### ----------ENW_3---------- ###  13
    _C = torch.eq(ind_bc, 13)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_ENW_3_1_le = u_x + v_y
    pde_ENW_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_ENW_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_ENW_3_1_re = torch.zeros(pde_ENW_3_1_le.shape[0], 1)
    pde_ENW_3_2_re = torch.zeros(pde_ENW_3_2_le.shape[0], 1)
    pde_ENW_3_3_re = torch.zeros(pde_ENW_3_3_le.shape[0], 1)
    pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re = pde_ENW_3_1_re.to(device), pde_ENW_3_2_re.to(device), pde_ENW_3_3_re.to(device)


    ### ----------NWS_3---------- ###  14
    _C = torch.eq(ind_bc, 14)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_NWS_3_1_le = u_x + v_y
    pde_NWS_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_NWS_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_NWS_3_1_re = torch.zeros(pde_NWS_3_1_le.shape[0], 1)
    pde_NWS_3_2_re = torch.zeros(pde_NWS_3_2_le.shape[0], 1)
    pde_NWS_3_3_re = torch.zeros(pde_NWS_3_3_le.shape[0], 1)
    pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re = pde_NWS_3_1_re.to(device), pde_NWS_3_2_re.to(device), pde_NWS_3_3_re.to(device)


    ### ----------left---------- ###  15
    _C = torch.eq(ind_bc, 15)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    uW, vW, pW = uC, vC, pC

    # compute PDE matrix
    pde_left_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_1_re = torch.zeros(pde_left_1_le.shape[0], 1)
    pde_left_2_re = torch.zeros(pde_left_2_le.shape[0], 1)
    pde_left_3_re = torch.zeros(pde_left_3_le.shape[0], 1)
    pde_left_1_re, pde_left_2_re, pde_left_3_re = pde_left_1_re.to(device), pde_left_2_re.to(device), pde_left_3_re.to(device)

    pde_left_4_le = uC
    pde_left_5_le = vC
    pde_left_4_re = uC_bc_t
    pde_left_5_re = vC_bc_t


    ### ----------left_E---------- ###  16
    _C = torch.eq(ind_bc, 16)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_E_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_E_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_E_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_E_1_re = torch.zeros(pde_left_E_1_le.shape[0], 1)
    pde_left_E_2_re = torch.zeros(pde_left_E_2_le.shape[0], 1)
    pde_left_E_3_re = torch.zeros(pde_left_E_3_le.shape[0], 1)
    pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re = pde_left_E_1_re.to(device), pde_left_E_2_re.to(device), pde_left_E_3_re.to(device)

    pde_left_E_4_le = uC
    pde_left_E_5_le = vC
    pde_left_E_4_re = uC_bc_t
    pde_left_E_5_re = vC_bc_t


    ### ----------left_N---------- ###  17
    _C = torch.eq(ind_bc, 17)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uW, vW, pW = uC, vC, pC
    uN, vN, pN = -uC, -vC, pC

    # compute PDE matrix
    pde_left_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_N_1_re = torch.zeros(pde_left_N_1_le.shape[0], 1)
    pde_left_N_2_re = torch.zeros(pde_left_N_2_le.shape[0], 1)
    pde_left_N_3_re = torch.zeros(pde_left_N_3_le.shape[0], 1)
    pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re = pde_left_N_1_re.to(device), pde_left_N_2_re.to(device), pde_left_N_3_re.to(device)

    pde_left_N_4_le = uC
    pde_left_N_5_le = vC
    pde_left_N_4_re = uC_bc_t
    pde_left_N_5_re = vC_bc_t


    ### ----------left_NE---------- ###  18
    _C = torch.eq(ind_bc, 18)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uN, vN, pN = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_NE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_NE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_NE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_NE_1_re = torch.zeros(pde_left_NE_1_le.shape[0], 1)
    pde_left_NE_2_re = torch.zeros(pde_left_NE_2_le.shape[0], 1)
    pde_left_NE_3_re = torch.zeros(pde_left_NE_3_le.shape[0], 1)
    pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re = pde_left_NE_1_re.to(device), pde_left_NE_2_re.to(device), pde_left_NE_3_re.to(device)

    pde_left_NE_4_le = uC
    pde_left_NE_5_le = vC
    pde_left_NE_4_re = uC_bc_t
    pde_left_NE_5_re = vC_bc_t


    ### ----------left_S---------- ###  19
    _C = torch.eq(ind_bc, 19)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uW, vW, pW = uC, vC, pC
    uS, vS, pS = -uC, -vC, pC

    # compute PDE matrix
    pde_left_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_S_1_re = torch.zeros(pde_left_S_1_le.shape[0], 1)
    pde_left_S_2_re = torch.zeros(pde_left_S_2_le.shape[0], 1)
    pde_left_S_3_re = torch.zeros(pde_left_S_3_le.shape[0], 1)
    pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re = pde_left_S_1_re.to(device), pde_left_S_2_re.to(device), pde_left_S_3_re.to(device)

    pde_left_S_4_le = uC
    pde_left_S_5_le = vC
    pde_left_S_4_re = uC_bc_t
    pde_left_S_5_re = vC_bc_t


    ### ----------left_SE---------- ###  20
    _C = torch.eq(ind_bc, 20)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_SE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_SE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_SE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_SE_1_re = torch.zeros(pde_left_SE_1_le.shape[0], 1)
    pde_left_SE_2_re = torch.zeros(pde_left_SE_2_le.shape[0], 1)
    pde_left_SE_3_re = torch.zeros(pde_left_SE_3_le.shape[0], 1)
    pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re = pde_left_SE_1_re.to(device), pde_left_SE_2_re.to(device), pde_left_SE_3_re.to(device)

    pde_left_SE_4_le = uC
    pde_left_SE_5_le = vC
    pde_left_SE_4_re = uC_bc_t
    pde_left_SE_5_re = vC_bc_t


    ### ----------right---------- ###  21
    _C = torch.eq(ind_bc, 21)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0 (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    uE, vE, pE = uC, vC, -pC

    # compute PDE matrix
    pde_right_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_1_re = torch.zeros(pde_right_1_le.shape[0], 1)
    pde_right_2_re = torch.zeros(pde_right_2_le.shape[0], 1)
    pde_right_3_re = torch.zeros(pde_right_3_le.shape[0], 1)
    pde_right_1_re, pde_right_2_re, pde_right_3_re = pde_right_1_re.to(device), pde_right_2_re.to(device), pde_right_3_re.to(device)
    pde_right_4_le = pC
    pde_right_4_re = pC_bc_t


    ### ----------right_W---------- ###  22
    _C = torch.eq(ind_bc, 22)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_W_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_W_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_W_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_W_1_re = torch.zeros(pde_right_W_1_le.shape[0], 1)
    pde_right_W_2_re = torch.zeros(pde_right_W_2_le.shape[0], 1)
    pde_right_W_3_re = torch.zeros(pde_right_W_3_le.shape[0], 1)
    pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re = pde_right_W_1_re.to(device), pde_right_W_2_re.to(device), pde_right_W_3_re.to(device)
    pde_right_W_4_le = pC
    pde_right_W_4_re = pC_bc_t


    ### ----------right_N---------- ###  23
    _C = torch.eq(ind_bc, 23)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uE, vE, pE = uC, vC, -pC
    uN, vN, pN = -uC, -vC, pC

    # compute PDE matrix
    pde_right_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_N_1_re = torch.zeros(pde_right_N_1_le.shape[0], 1)
    pde_right_N_2_re = torch.zeros(pde_right_N_2_le.shape[0], 1)
    pde_right_N_3_re = torch.zeros(pde_right_N_3_le.shape[0], 1)
    pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re = pde_right_N_1_re.to(device), pde_right_N_2_re.to(device), pde_right_N_3_re.to(device)
    pde_right_N_4_le = pC
    pde_right_N_4_re = pC_bc_t


    ### ----------right_NW---------- ###  24
    _C = torch.eq(ind_bc, 24)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_NW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_NW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_NW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_NW_1_re = torch.zeros(pde_right_NW_1_le.shape[0], 1)
    pde_right_NW_2_re = torch.zeros(pde_right_NW_2_le.shape[0], 1)
    pde_right_NW_3_re = torch.zeros(pde_right_NW_3_le.shape[0], 1)
    pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re = pde_right_NW_1_re.to(device), pde_right_NW_2_re.to(device), pde_right_NW_3_re.to(device)
    pde_right_NW_4_le = pC
    pde_right_NW_4_re = pC_bc_t


    ### ----------right_S---------- ###  25
    _C = torch.eq(ind_bc, 25)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uE, vE, pE = uC, vC, -pC
    uS, vS, pS = -uC, -vC, pC

    # compute PDE matrix
    pde_right_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_S_1_re = torch.zeros(pde_right_S_1_le.shape[0], 1)
    pde_right_S_2_re = torch.zeros(pde_right_S_2_le.shape[0], 1)
    pde_right_S_3_re = torch.zeros(pde_right_S_3_le.shape[0], 1)
    pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re = pde_right_S_1_re.to(device), pde_right_S_2_re.to(device), pde_right_S_3_re.to(device)
    pde_right_S_4_le = pC
    pde_right_S_4_re = pC_bc_t


    ### ----------right_SW---------- ###  26
    _C = torch.eq(ind_bc, 26)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uS, vS, pS = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_SW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_SW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_SW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_SW_1_re = torch.zeros(pde_right_SW_1_le.shape[0], 1)
    pde_right_SW_2_re = torch.zeros(pde_right_SW_2_le.shape[0], 1)
    pde_right_SW_3_re = torch.zeros(pde_right_SW_3_le.shape[0], 1)
    pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re = pde_right_SW_1_re.to(device), pde_right_SW_2_re.to(device), pde_right_SW_3_re.to(device)
    pde_right_SW_4_le = pC
    pde_right_SW_4_re = pC_bc_t


    # lamda_bc = 14
    # lamda_bc = 1

    # pde_M_normal_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le], axis=0)
    # pde_M_bc_le = lamda_bc*torch.cat([pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
    #                                   pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
    #                                   pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
    #                                   pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
    #                                   pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
    #                                   pde_left_1_le, pde_left_2_le, pde_left_3_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, \
    #                                   pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, \
    #                                   pde_right_1_le, pde_right_2_le, pde_right_3_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, \
    #                                   pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le], axis=0)
    # pde_M_le = torch.cat([pde_M_normal_le, pde_M_bc_le], axis=0)

    # pde_M_normal_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re], axis=0)
    # pde_M_bc_re = lamda_bc*torch.cat([pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
    #                                   pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
    #                                   pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
    #                                   pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
    #                                   pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
    #                                   pde_left_1_re, pde_left_2_re, pde_left_3_re, pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re, pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re, \
    #                                   pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re, pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re, pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re, \
    #                                   pde_right_1_re, pde_right_2_re, pde_right_3_re, pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re, pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re, \
    #                                   pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re, pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re, pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re], axis=0)
    # pde_M_re = torch.cat([pde_M_normal_re, pde_M_bc_re], axis=0)


    # pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
    #                       pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
    #                       pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
    #                       pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
    #                       pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
    #                       lamda_bc*pde_left_1_le, lamda_bc*pde_left_2_le, lamda_bc*pde_left_3_le, lamda_bc*pde_left_E_1_le, lamda_bc*pde_left_E_2_le, lamda_bc*pde_left_E_3_le, lamda_bc*pde_left_N_1_le, lamda_bc*pde_left_N_2_le, lamda_bc*pde_left_N_3_le, \
    #                       lamda_bc*pde_left_NE_1_le, lamda_bc*pde_left_NE_2_le, lamda_bc*pde_left_NE_3_le, lamda_bc*pde_left_S_1_le, lamda_bc*pde_left_S_2_le, lamda_bc*pde_left_S_3_le, lamda_bc*pde_left_SE_1_le, lamda_bc*pde_left_SE_2_le, lamda_bc*pde_left_SE_3_le, \
    #                       lamda_bc*pde_right_1_le, lamda_bc*pde_right_2_le, lamda_bc*pde_right_3_le, lamda_bc*pde_right_W_1_le, lamda_bc*pde_right_W_2_le, lamda_bc*pde_right_W_3_le, lamda_bc*pde_right_N_1_le, lamda_bc*pde_right_N_2_le, lamda_bc*pde_right_N_3_le, \
    #                       lamda_bc*pde_right_NW_1_le, lamda_bc*pde_right_NW_2_le, lamda_bc*pde_right_NW_3_le, lamda_bc*pde_right_S_1_le, lamda_bc*pde_right_S_2_le, lamda_bc*pde_right_S_3_le, lamda_bc*pde_right_SW_1_le, lamda_bc*pde_right_SW_2_le, lamda_bc*pde_right_SW_3_le], axis=0)


    # pde_M_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re, pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
    #                       pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
    #                       pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
    #                       pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
    #                       pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
    #                       lamda_bc*pde_left_1_re, lamda_bc*pde_left_2_re, lamda_bc*pde_left_3_re, lamda_bc*pde_left_E_1_re, lamda_bc*pde_left_E_2_re, lamda_bc*pde_left_E_3_re, lamda_bc*pde_left_N_1_re, lamda_bc*pde_left_N_2_re, lamda_bc*pde_left_N_3_re, \
    #                       lamda_bc*pde_left_NE_1_re, lamda_bc*pde_left_NE_2_re, lamda_bc*pde_left_NE_3_re, lamda_bc*pde_left_S_1_re, lamda_bc*pde_left_S_2_re, lamda_bc*pde_left_S_3_re, lamda_bc*pde_left_SE_1_re, lamda_bc*pde_left_SE_2_re, lamda_bc*pde_left_SE_3_re, \
    #                       lamda_bc*pde_right_1_re, lamda_bc*pde_right_2_re, lamda_bc*pde_right_3_re, lamda_bc*pde_right_W_1_re, lamda_bc*pde_right_W_2_re, lamda_bc*pde_right_W_3_re, lamda_bc*pde_right_N_1_re, lamda_bc*pde_right_N_2_re, lamda_bc*pde_right_N_3_re, \
    #                       lamda_bc*pde_right_NW_1_re, lamda_bc*pde_right_NW_2_re, lamda_bc*pde_right_NW_3_re, lamda_bc*pde_right_S_1_re, lamda_bc*pde_right_S_2_re, lamda_bc*pde_right_S_3_re, lamda_bc*pde_right_SW_1_re, lamda_bc*pde_right_SW_2_re, lamda_bc*pde_right_SW_3_re], axis=0)


    pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
                          pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
                          pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
                          pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
                          pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
                          pde_left_1_le, pde_left_2_le, pde_left_3_le, lamda_bc*pde_left_4_le, lamda_bc*pde_left_5_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, lamda_bc*pde_left_E_4_le, lamda_bc*pde_left_E_5_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, lamda_bc*pde_left_N_4_le, lamda_bc*pde_left_N_5_le, \
                          pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, lamda_bc*pde_left_NE_4_le, lamda_bc*pde_left_NE_5_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, lamda_bc*pde_left_S_4_le, lamda_bc*pde_left_S_5_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, lamda_bc*pde_left_SE_4_le, lamda_bc*pde_left_SE_5_le, \
                          pde_right_1_le, pde_right_2_le, pde_right_3_le, lamda_bc*pde_right_4_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, lamda_bc*pde_right_W_4_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, lamda_bc*pde_right_N_4_le, \
                          pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, lamda_bc*pde_right_NW_4_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, lamda_bc*pde_right_S_4_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le, lamda_bc*pde_right_SW_4_le], axis=0)


    pde_M_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re, pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
                          pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
                          pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
                          pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
                          pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
                          pde_left_1_re, pde_left_2_re, pde_left_3_re, lamda_bc*pde_left_4_re, lamda_bc*pde_left_5_re, pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re, lamda_bc*pde_left_E_4_re, lamda_bc*pde_left_E_5_re, pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re, lamda_bc*pde_left_N_4_re, lamda_bc*pde_left_N_5_re,  \
                          pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re, lamda_bc*pde_left_NE_4_re, lamda_bc*pde_left_NE_5_re, pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re, lamda_bc*pde_left_S_4_re, lamda_bc*pde_left_S_5_re, pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re, lamda_bc*pde_left_SE_4_re, lamda_bc*pde_left_SE_5_re,  \
                          pde_right_1_re, pde_right_2_re, pde_right_3_re, lamda_bc*pde_right_4_re, pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re, lamda_bc*pde_right_W_4_re, pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re, lamda_bc*pde_right_N_4_re, \
                          pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re, lamda_bc*pde_right_NW_4_re, pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re, lamda_bc*pde_right_S_4_re, pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re, lamda_bc*pde_right_SW_4_re], axis=0)
    
    pde_M_re = pde_M_re.double()  # change to 64 bits

    # start = time.time()
    kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re
    # end = time.time()
    # runtime = end - start

    # iteration to calculate the nonlinear term
    for i in np.arange(1):
        ### ------normal PDE------ ###  0
        _C = torch.eq(ind_bc, 0)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :] 
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters
        u0C_old, v0C_old = u0C, v0C

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_normal_1_le = u_x + v_y
        pde_normal_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_normal_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_normal_1_re = torch.zeros(pde_normal_1_le.shape[0], 1)
        # pde_normal_2_re = torch.zeros(pde_normal_2_le.shape[0], 1)
        # pde_normal_3_re = torch.zeros(pde_normal_3_le.shape[0], 1)
        # pde_normal_1_re, pde_normal_2_re, pde_normal_3_re = pde_normal_1_re.to(device), pde_normal_2_re.to(device), pde_normal_3_re.to(device)


        ### ----------W_1---------- ###  1
        _C = torch.eq(ind_bc, 1)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_W_1_1_le = u_x + v_y
        pde_W_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_W_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_W_1_1_re = torch.zeros(pde_W_1_1_le.shape[0], 1)
        # pde_W_1_2_re = torch.zeros(pde_W_1_2_le.shape[0], 1)
        # pde_W_1_3_re = torch.zeros(pde_W_1_3_le.shape[0], 1)
        # pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re = pde_W_1_1_re.to(device), pde_W_1_2_re.to(device), pde_W_1_3_re.to(device)


        ### ----------S_1---------- ###  2
        _C = torch.eq(ind_bc, 2)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_S_1_1_le = u_x + v_y
        pde_S_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_S_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_S_1_1_re = torch.zeros(pde_S_1_1_le.shape[0], 1)
        # pde_S_1_2_re = torch.zeros(pde_S_1_2_le.shape[0], 1)
        # pde_S_1_3_re = torch.zeros(pde_S_1_3_le.shape[0], 1)
        # pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re = pde_S_1_1_re.to(device), pde_S_1_2_re.to(device), pde_S_1_3_re.to(device)


        ### ----------E_1---------- ###  3
        _C = torch.eq(ind_bc, 3)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_E_1_1_le = u_x + v_y
        pde_E_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_E_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_E_1_1_re = torch.zeros(pde_E_1_1_le.shape[0], 1)
        # pde_E_1_2_re = torch.zeros(pde_E_1_2_le.shape[0], 1)
        # pde_E_1_3_re = torch.zeros(pde_E_1_3_le.shape[0], 1)
        # pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re = pde_E_1_1_re.to(device), pde_E_1_2_re.to(device), pde_E_1_3_re.to(device)


        ### ----------N_1---------- ###  4
        _C = torch.eq(ind_bc, 4)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_N_1_1_le = u_x + v_y
        pde_N_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_N_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_N_1_1_re = torch.zeros(pde_N_1_1_le.shape[0], 1)
        # pde_N_1_2_re = torch.zeros(pde_N_1_2_le.shape[0], 1)
        # pde_N_1_3_re = torch.zeros(pde_N_1_3_le.shape[0], 1)
        # pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re = pde_N_1_1_re.to(device), pde_N_1_2_re.to(device), pde_N_1_3_re.to(device)


        ### ----------WS_2---------- ###  5
        _C = torch.eq(ind_bc, 5)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WS_2_1_le = u_x + v_y
        pde_WS_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WS_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WS_2_1_re = torch.zeros(pde_WS_2_1_le.shape[0], 1)
        # pde_WS_2_2_re = torch.zeros(pde_WS_2_2_le.shape[0], 1)
        # pde_WS_2_3_re = torch.zeros(pde_WS_2_3_le.shape[0], 1)
        # pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re = pde_WS_2_1_re.to(device), pde_WS_2_2_re.to(device), pde_WS_2_3_re.to(device)


        ### ----------WE_2---------- ###  6
        _C = torch.eq(ind_bc, 6)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WE_2_1_le = u_x + v_y
        pde_WE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WE_2_1_re = torch.zeros(pde_WE_2_1_le.shape[0], 1)
        # pde_WE_2_2_re = torch.zeros(pde_WE_2_2_le.shape[0], 1)
        # pde_WE_2_3_re = torch.zeros(pde_WE_2_3_le.shape[0], 1)
        # pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re = pde_WE_2_1_re.to(device), pde_WE_2_2_re.to(device), pde_WE_2_3_re.to(device)


        ### ----------WN_2---------- ###  7
        _C = torch.eq(ind_bc, 7)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uW, vW, pW = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WN_2_1_le = u_x + v_y
        pde_WN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WN_2_1_re = torch.zeros(pde_WN_2_1_le.shape[0], 1)
        # pde_WN_2_2_re = torch.zeros(pde_WN_2_2_le.shape[0], 1)
        # pde_WN_2_3_re = torch.zeros(pde_WN_2_3_le.shape[0], 1)
        # pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re = pde_WN_2_1_re.to(device), pde_WN_2_2_re.to(device), pde_WN_2_3_re.to(device)


        ### ----------SE_2---------- ###  8
        _C = torch.eq(ind_bc, 8)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SE_2_1_le = u_x + v_y
        pde_SE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SE_2_1_re = torch.zeros(pde_SE_2_1_le.shape[0], 1)
        # pde_SE_2_2_re = torch.zeros(pde_SE_2_2_le.shape[0], 1)
        # pde_SE_2_3_re = torch.zeros(pde_SE_2_3_le.shape[0], 1)
        # pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re = pde_SE_2_1_re.to(device), pde_SE_2_2_re.to(device), pde_SE_2_3_re.to(device)


        ### ----------SN_2---------- ###  9
        _C = torch.eq(ind_bc, 9)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SN_2_1_le = u_x + v_y
        pde_SN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SN_2_1_re = torch.zeros(pde_SN_2_1_le.shape[0], 1)
        # pde_SN_2_2_re = torch.zeros(pde_SN_2_2_le.shape[0], 1)
        # pde_SN_2_3_re = torch.zeros(pde_SN_2_3_le.shape[0], 1)
        # pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re = pde_SN_2_1_re.to(device), pde_SN_2_2_re.to(device), pde_SN_2_3_re.to(device)


        ### ----------EN_2---------- ###  10
        _C = torch.eq(ind_bc, 10)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_EN_2_1_le = u_x + v_y
        pde_EN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_EN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_EN_2_1_re = torch.zeros(pde_EN_2_1_le.shape[0], 1)
        # pde_EN_2_2_re = torch.zeros(pde_EN_2_2_le.shape[0], 1)
        # pde_EN_2_3_re = torch.zeros(pde_EN_2_3_le.shape[0], 1)
        # pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re = pde_EN_2_1_re.to(device), pde_EN_2_2_re.to(device), pde_EN_2_3_re.to(device)


        ### ----------WSE_3---------- ###  11
        _C = torch.eq(ind_bc, 11)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WSE_3_1_le = u_x + v_y
        pde_WSE_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WSE_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WSE_3_1_re = torch.zeros(pde_WSE_3_1_le.shape[0], 1)
        # pde_WSE_3_2_re = torch.zeros(pde_WSE_3_2_le.shape[0], 1)
        # pde_WSE_3_3_re = torch.zeros(pde_WSE_3_3_le.shape[0], 1)
        # pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re = pde_WSE_3_1_re.to(device), pde_WSE_3_2_re.to(device), pde_WSE_3_3_re.to(device)


        ### ----------SEN_3---------- ###  12
        _C = torch.eq(ind_bc, 12)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SEN_3_1_le = u_x + v_y
        pde_SEN_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SEN_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SEN_3_1_re = torch.zeros(pde_SEN_3_1_le.shape[0], 1)
        # pde_SEN_3_2_re = torch.zeros(pde_SEN_3_2_le.shape[0], 1)
        # pde_SEN_3_3_re = torch.zeros(pde_SEN_3_3_le.shape[0], 1)
        # pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re = pde_SEN_3_1_re.to(device), pde_SEN_3_2_re.to(device), pde_SEN_3_3_re.to(device)


        ### ----------ENW_3---------- ###  13
        _C = torch.eq(ind_bc, 13)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_ENW_3_1_le = u_x + v_y
        pde_ENW_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_ENW_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_ENW_3_1_re = torch.zeros(pde_ENW_3_1_le.shape[0], 1)
        # pde_ENW_3_2_re = torch.zeros(pde_ENW_3_2_le.shape[0], 1)
        # pde_ENW_3_3_re = torch.zeros(pde_ENW_3_3_le.shape[0], 1)
        # pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re = pde_ENW_3_1_re.to(device), pde_ENW_3_2_re.to(device), pde_ENW_3_3_re.to(device)


        ### ----------NWS_3---------- ###  14
        _C = torch.eq(ind_bc, 14)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_NWS_3_1_le = u_x + v_y
        pde_NWS_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_NWS_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_NWS_3_1_re = torch.zeros(pde_NWS_3_1_le.shape[0], 1)
        # pde_NWS_3_2_re = torch.zeros(pde_NWS_3_2_le.shape[0], 1)
        # pde_NWS_3_3_re = torch.zeros(pde_NWS_3_3_le.shape[0], 1)
        # pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re = pde_NWS_3_1_re.to(device), pde_NWS_3_2_re.to(device), pde_NWS_3_3_re.to(device)


        ### ----------left---------- ###  15
        _C = torch.eq(ind_bc, 15)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        uW, vW, pW = uC, vC, pC

        # compute PDE matrix
        # pde_left_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_1_re = torch.zeros(pde_left_1_le.shape[0], 1)
        # pde_left_2_re = torch.zeros(pde_left_2_le.shape[0], 1)
        # pde_left_3_re = torch.zeros(pde_left_3_le.shape[0], 1)
        # pde_left_1_re, pde_left_2_re, pde_left_3_re = pde_left_1_re.to(device), pde_left_2_re.to(device), pde_left_3_re.to(device)

        # pde_left_4_le = uC
        # pde_left_5_le = vC
        # pde_left_4_re = uC_bc_t
        # pde_left_5_re = vC_bc_t


        ### ----------left_E---------- ###  16
        _C = torch.eq(ind_bc, 16)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_E_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_E_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_E_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_E_1_re = torch.zeros(pde_left_E_1_le.shape[0], 1)
        # pde_left_E_2_re = torch.zeros(pde_left_E_2_le.shape[0], 1)
        # pde_left_E_3_re = torch.zeros(pde_left_E_3_le.shape[0], 1)
        # pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re = pde_left_E_1_re.to(device), pde_left_E_2_re.to(device), pde_left_E_3_re.to(device)

        # pde_left_E_4_le = uC
        # pde_left_E_5_le = vC
        # pde_left_E_4_re = uC_bc_t
        # pde_left_E_5_re = vC_bc_t


        ### ----------left_N---------- ###  17
        _C = torch.eq(ind_bc, 17)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uW, vW, pW = uC, vC, pC
        uN, vN, pN = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_N_1_re = torch.zeros(pde_left_N_1_le.shape[0], 1)
        # pde_left_N_2_re = torch.zeros(pde_left_N_2_le.shape[0], 1)
        # pde_left_N_3_re = torch.zeros(pde_left_N_3_le.shape[0], 1)
        # pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re = pde_left_N_1_re.to(device), pde_left_N_2_re.to(device), pde_left_N_3_re.to(device)

        # pde_left_N_4_le = uC
        # pde_left_N_5_le = vC
        # pde_left_N_4_re = uC_bc_t
        # pde_left_N_5_re = vC_bc_t


        ### ----------left_NE---------- ###  18
        _C = torch.eq(ind_bc, 18)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uN, vN, pN = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_NE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_NE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_NE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_NE_1_re = torch.zeros(pde_left_NE_1_le.shape[0], 1)
        # pde_left_NE_2_re = torch.zeros(pde_left_NE_2_le.shape[0], 1)
        # pde_left_NE_3_re = torch.zeros(pde_left_NE_3_le.shape[0], 1)
        # pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re = pde_left_NE_1_re.to(device), pde_left_NE_2_re.to(device), pde_left_NE_3_re.to(device)

        # pde_left_NE_4_le = uC
        # pde_left_NE_5_le = vC
        # pde_left_NE_4_re = uC_bc_t
        # pde_left_NE_5_re = vC_bc_t


        ### ----------left_S---------- ###  19
        _C = torch.eq(ind_bc, 19)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uW, vW, pW = uC, vC, pC
        uS, vS, pS = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_S_1_re = torch.zeros(pde_left_S_1_le.shape[0], 1)
        # pde_left_S_2_re = torch.zeros(pde_left_S_2_le.shape[0], 1)
        # pde_left_S_3_re = torch.zeros(pde_left_S_3_le.shape[0], 1)
        # pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re = pde_left_S_1_re.to(device), pde_left_S_2_re.to(device), pde_left_S_3_re.to(device)

        # pde_left_S_4_le = uC
        # pde_left_S_5_le = vC
        # pde_left_S_4_re = uC_bc_t
        # pde_left_S_5_re = vC_bc_t


        ### ----------left_SE---------- ###  20
        _C = torch.eq(ind_bc, 20)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_SE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_SE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_SE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_SE_1_re = torch.zeros(pde_left_SE_1_le.shape[0], 1)
        # pde_left_SE_2_re = torch.zeros(pde_left_SE_2_le.shape[0], 1)
        # pde_left_SE_3_re = torch.zeros(pde_left_SE_3_le.shape[0], 1)
        # pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re = pde_left_SE_1_re.to(device), pde_left_SE_2_re.to(device), pde_left_SE_3_re.to(device)

        # pde_left_SE_4_le = uC
        # pde_left_SE_5_le = vC
        # pde_left_SE_4_re = uC_bc_t
        # pde_left_SE_5_re = vC_bc_t


        ### ----------right---------- ###  21
        _C = torch.eq(ind_bc, 21)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0 (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        uE, vE, pE = uC, vC, -pC

        # compute PDE matrix
        # pde_right_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_1_re = torch.zeros(pde_right_1_le.shape[0], 1)
        # pde_right_2_re = torch.zeros(pde_right_2_le.shape[0], 1)
        # pde_right_3_re = torch.zeros(pde_right_3_le.shape[0], 1)
        # pde_right_1_re, pde_right_2_re, pde_right_3_re = pde_right_1_re.to(device), pde_right_2_re.to(device), pde_right_3_re.to(device)
        # pde_right_4_le = pC
        # pde_right_4_re = pC_bc_t


        ### ----------right_W---------- ###  22
        _C = torch.eq(ind_bc, 22)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_W_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_W_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_W_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_W_1_re = torch.zeros(pde_right_W_1_le.shape[0], 1)
        # pde_right_W_2_re = torch.zeros(pde_right_W_2_le.shape[0], 1)
        # pde_right_W_3_re = torch.zeros(pde_right_W_3_le.shape[0], 1)
        # pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re = pde_right_W_1_re.to(device), pde_right_W_2_re.to(device), pde_right_W_3_re.to(device)
        # pde_right_W_4_le = pC
        # pde_right_W_4_re = pC_bc_t


        ### ----------right_N---------- ###  23
        _C = torch.eq(ind_bc, 23)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uE, vE, pE = uC, vC, -pC
        uN, vN, pN = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_N_1_re = torch.zeros(pde_right_N_1_le.shape[0], 1)
        # pde_right_N_2_re = torch.zeros(pde_right_N_2_le.shape[0], 1)
        # pde_right_N_3_re = torch.zeros(pde_right_N_3_le.shape[0], 1)
        # pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re = pde_right_N_1_re.to(device), pde_right_N_2_re.to(device), pde_right_N_3_re.to(device)
        # pde_right_N_4_le = pC
        # pde_right_N_4_re = pC_bc_t


        ### ----------right_NW---------- ###  24
        _C = torch.eq(ind_bc, 24)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_NW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_NW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_NW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_NW_1_re = torch.zeros(pde_right_NW_1_le.shape[0], 1)
        # pde_right_NW_2_re = torch.zeros(pde_right_NW_2_le.shape[0], 1)
        # pde_right_NW_3_re = torch.zeros(pde_right_NW_3_le.shape[0], 1)
        # pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re = pde_right_NW_1_re.to(device), pde_right_NW_2_re.to(device), pde_right_NW_3_re.to(device)
        # pde_right_NW_4_le = pC
        # pde_right_NW_4_re = pC_bc_t


        ### ----------right_S---------- ###  25
        _C = torch.eq(ind_bc, 25)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uE, vE, pE = uC, vC, -pC
        uS, vS, pS = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_S_1_re = torch.zeros(pde_right_S_1_le.shape[0], 1)
        # pde_right_S_2_re = torch.zeros(pde_right_S_2_le.shape[0], 1)
        # pde_right_S_3_re = torch.zeros(pde_right_S_3_le.shape[0], 1)
        # pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re = pde_right_S_1_re.to(device), pde_right_S_2_re.to(device), pde_right_S_3_re.to(device)
        # pde_right_S_4_le = pC
        # pde_right_S_4_re = pC_bc_t


        ### ----------right_SW---------- ###  26
        _C = torch.eq(ind_bc, 26)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uS, vS, pS = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_SW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_SW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_SW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_SW_1_re = torch.zeros(pde_right_SW_1_le.shape[0], 1)
        # pde_right_SW_2_re = torch.zeros(pde_right_SW_2_le.shape[0], 1)
        # pde_right_SW_3_re = torch.zeros(pde_right_SW_3_le.shape[0], 1)
        # pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re = pde_right_SW_1_re.to(device), pde_right_SW_2_re.to(device), pde_right_SW_3_re.to(device)
        # pde_right_SW_4_le = pC
        # pde_right_SW_4_re = pC_bc_t



        pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
                              pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
                              pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
                              pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
                              pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
                              pde_left_1_le, pde_left_2_le, pde_left_3_le, lamda_bc*pde_left_4_le, lamda_bc*pde_left_5_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, lamda_bc*pde_left_E_4_le, lamda_bc*pde_left_E_5_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, lamda_bc*pde_left_N_4_le, lamda_bc*pde_left_N_5_le, \
                              pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, lamda_bc*pde_left_NE_4_le, lamda_bc*pde_left_NE_5_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, lamda_bc*pde_left_S_4_le, lamda_bc*pde_left_S_5_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, lamda_bc*pde_left_SE_4_le, lamda_bc*pde_left_SE_5_le, \
                              pde_right_1_le, pde_right_2_le, pde_right_3_le, lamda_bc*pde_right_4_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, lamda_bc*pde_right_W_4_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, lamda_bc*pde_right_N_4_le, \
                              pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, lamda_bc*pde_right_NW_4_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, lamda_bc*pde_right_S_4_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le, lamda_bc*pde_right_SW_4_le], axis=0)

        kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re


        # test error
        _C = torch.eq(ind_bc, 0)
        _C = _C.reshape(_C.shape[0], -1)
        _C = _C.flatten()
        uC = M_u[_C, :]
        u0C_new = uC @ kernel_parameters
        error = torch.mean((u0C_new - u0C_old)**2)
        ttt = 1

   
    # pseudo inverse prediction results
    u, v, p = M_u @ kernel_parameters, M_v @ kernel_parameters, M_p @ kernel_parameters
    u, v, p = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1), p.reshape(inputs0.shape[0], -1)
    u, v, p = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), p.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    u, v, p = torch.unsqueeze(u, axis=1), torch.unsqueeze(v, axis=1), torch.unsqueeze(p, axis=1)
    outputs_pseudo_inverse = torch.concat([u, v, p], axis=1)
    outputs_pseudo_inverse = outputs_pseudo_inverse.float()
    # outputs_pseudo_inverse = outputs_pseudo_inverse.requires_grad_(True)

    # convolution 2d prediction results
    outputs_conv2d = conv2d_head(inputs_conv2d).float()

    ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    return outputs_pseudo_inverse, outputs_conv2d, ssr






def CNN_PINN_Pseudo_Inverse_2D_NS_DirichletBC_grad(inputs, inputs0, conv2d_head):

    inputs_conv2d = inputs

    n_kernel_size = 5
    n_p = int(np.floor(n_kernel_size/2))
    # kernel size: 1*1, 3*3, 5*5, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

    # computational boundary
    # x_l, x_u, y_l, y_u = 0, 2, 0, 1
    x_l, x_u, y_l, y_u = 0, 2, 0, 2

    # dx, dy
    # n_x, n_y = 64, 64
    n_x, n_y = 128, 128
    # n_x, n_y = 512, 512
    dx = (x_u - x_l) / n_x  #/ 2
    dy = (y_u - y_l) / n_y  #/ 2

    x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
    y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)
    x_grid, y_grid = np.meshgrid(x, y)

    # n_padding = int((inputs0.shape[2] - n_x)/2)

    device = inputs.device

    x_ind_bc = inputs0[0, 0, :, :]

    # pad BC indicator
    # x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=0)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)

    inputs_np = inputs.detach().cpu().numpy()

    M0 = torch.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_conv2d.shape[0]):
        for _x in np.arange(inputs_conv2d.shape[2]):
            for _y in np.arange(inputs_conv2d.shape[3]):
                M0_part = inputs[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                M0_part = M0_part.reshape(M0_part.shape[0], M0_part.shape[1], -1)
                M0_part = M0_part.reshape(M0_part.shape[0], -1)
                M0[ni:ni+1, :] = M0_part
                ni = ni + 1

    M_u = torch.hstack([M0, torch.zeros_like(M0), torch.zeros_like(M0)])
    M_v = torch.hstack([torch.zeros_like(M0), M0, torch.zeros_like(M0)])
    M_p = torch.hstack([torch.zeros_like(M0), torch.zeros_like(M0), M0])
    # M_u, M_v, M_p = torch.tensor(M_u).to(device), torch.tensor(M_v).to(device), torch.tensor(M_p).to(device)
    M_u, M_v, M_p = M_u.to(device), M_v.to(device), M_p.to(device)

    # get BC
    # M_u_bc, M_v_bc, M_p_bc = inputs0[:, 1, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 2, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 3, :, :].reshape(inputs0.shape[0], -1)
    # M_u_bc, M_v_bc, M_p_bc = M_u_bc.flatten(), M_v_bc.flatten(), M_p_bc.flatten()

    M_u_bc_t = M_v_bc_t = M_p_bc_t = torch.zeros((1, 128, 128))
    index_u = torch.eq(ind_bc, 15) | torch.eq(ind_bc, 16) | torch.eq(ind_bc, 17) | torch.eq(ind_bc, 18) | torch.eq(ind_bc, 19) | torch.eq(ind_bc, 20)
    M_u_bc_t[index_u] = 1
    M_u_bc_t, M_v_bc_t, M_p_bc_t = M_u_bc_t.reshape(1, -1), M_v_bc_t.reshape(1, -1), M_p_bc_t.reshape(1, -1)
    M_u_bc_t, M_v_bc_t, M_p_bc_t = M_u_bc_t.flatten().to(device), M_v_bc_t.flatten().to(device), M_p_bc_t.flatten().to(device)

    # first iteration
    reg = 1e-4
    # reg = 5e-4
    # reg = 1.83673469e+03  # kernel 5

    ### ------normal PDE------ ###  0
    _C = torch.eq(ind_bc, 0)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :] 
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    Re = 500
    # Re = 0.1
    pde_normal_1_le = u_x + v_y
    pde_normal_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_normal_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_normal_1_re = torch.zeros(pde_normal_1_le.shape[0], 1, requires_grad=True)
    pde_normal_2_re = torch.zeros(pde_normal_2_le.shape[0], 1, requires_grad=True)
    pde_normal_3_re = torch.zeros(pde_normal_3_le.shape[0], 1, requires_grad=True)
    pde_normal_1_re, pde_normal_2_re, pde_normal_3_re = pde_normal_1_re.to(device), pde_normal_2_re.to(device), pde_normal_3_re.to(device)


    ### ----------W_1---------- ###  1
    _C = torch.eq(ind_bc, 1)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_W_1_1_le = u_x + v_y
    pde_W_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_W_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_W_1_1_re = torch.zeros(pde_W_1_1_le.shape[0], 1)
    pde_W_1_2_re = torch.zeros(pde_W_1_2_le.shape[0], 1)
    pde_W_1_3_re = torch.zeros(pde_W_1_3_le.shape[0], 1)
    pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re = pde_W_1_1_re.to(device), pde_W_1_2_re.to(device), pde_W_1_3_re.to(device)


    ### ----------S_1---------- ###  2
    _C = torch.eq(ind_bc, 2)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_S_1_1_le = u_x + v_y
    pde_S_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_S_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_S_1_1_re = torch.zeros(pde_S_1_1_le.shape[0], 1)
    pde_S_1_2_re = torch.zeros(pde_S_1_2_le.shape[0], 1)
    pde_S_1_3_re = torch.zeros(pde_S_1_3_le.shape[0], 1)
    pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re = pde_S_1_1_re.to(device), pde_S_1_2_re.to(device), pde_S_1_3_re.to(device)


    ### ----------E_1---------- ###  3
    _C = torch.eq(ind_bc, 3)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_E_1_1_le = u_x + v_y
    pde_E_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_E_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_E_1_1_re = torch.zeros(pde_E_1_1_le.shape[0], 1)
    pde_E_1_2_re = torch.zeros(pde_E_1_2_le.shape[0], 1)
    pde_E_1_3_re = torch.zeros(pde_E_1_3_le.shape[0], 1)
    pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re = pde_E_1_1_re.to(device), pde_E_1_2_re.to(device), pde_E_1_3_re.to(device)


    ### ----------N_1---------- ###  4
    _C = torch.eq(ind_bc, 4)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_N_1_1_le = u_x + v_y
    pde_N_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_N_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_N_1_1_re = torch.zeros(pde_N_1_1_le.shape[0], 1)
    pde_N_1_2_re = torch.zeros(pde_N_1_2_le.shape[0], 1)
    pde_N_1_3_re = torch.zeros(pde_N_1_3_le.shape[0], 1)
    pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re = pde_N_1_1_re.to(device), pde_N_1_2_re.to(device), pde_N_1_3_re.to(device)


    ### ----------WS_2---------- ###  5
    _C = torch.eq(ind_bc, 5)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WS_2_1_le = u_x + v_y
    pde_WS_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WS_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WS_2_1_re = torch.zeros(pde_WS_2_1_le.shape[0], 1)
    pde_WS_2_2_re = torch.zeros(pde_WS_2_2_le.shape[0], 1)
    pde_WS_2_3_re = torch.zeros(pde_WS_2_3_le.shape[0], 1)
    pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re = pde_WS_2_1_re.to(device), pde_WS_2_2_re.to(device), pde_WS_2_3_re.to(device)


    ### ----------WE_2---------- ###  6
    _C = torch.eq(ind_bc, 6)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WE_2_1_le = u_x + v_y
    pde_WE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WE_2_1_re = torch.zeros(pde_WE_2_1_le.shape[0], 1)
    pde_WE_2_2_re = torch.zeros(pde_WE_2_2_le.shape[0], 1)
    pde_WE_2_3_re = torch.zeros(pde_WE_2_3_le.shape[0], 1)
    pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re = pde_WE_2_1_re.to(device), pde_WE_2_2_re.to(device), pde_WE_2_3_re.to(device)


    ### ----------WN_2---------- ###  7
    _C = torch.eq(ind_bc, 7)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uW, vW, pW = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WN_2_1_le = u_x + v_y
    pde_WN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WN_2_1_re = torch.zeros(pde_WN_2_1_le.shape[0], 1)
    pde_WN_2_2_re = torch.zeros(pde_WN_2_2_le.shape[0], 1)
    pde_WN_2_3_re = torch.zeros(pde_WN_2_3_le.shape[0], 1)
    pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re = pde_WN_2_1_re.to(device), pde_WN_2_2_re.to(device), pde_WN_2_3_re.to(device)


    ### ----------SE_2---------- ###  8
    _C = torch.eq(ind_bc, 8)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SE_2_1_le = u_x + v_y
    pde_SE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SE_2_1_re = torch.zeros(pde_SE_2_1_le.shape[0], 1)
    pde_SE_2_2_re = torch.zeros(pde_SE_2_2_le.shape[0], 1)
    pde_SE_2_3_re = torch.zeros(pde_SE_2_3_le.shape[0], 1)
    pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re = pde_SE_2_1_re.to(device), pde_SE_2_2_re.to(device), pde_SE_2_3_re.to(device)


    ### ----------SN_2---------- ###  9
    _C = torch.eq(ind_bc, 9)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SN_2_1_le = u_x + v_y
    pde_SN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SN_2_1_re = torch.zeros(pde_SN_2_1_le.shape[0], 1)
    pde_SN_2_2_re = torch.zeros(pde_SN_2_2_le.shape[0], 1)
    pde_SN_2_3_re = torch.zeros(pde_SN_2_3_le.shape[0], 1)
    pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re = pde_SN_2_1_re.to(device), pde_SN_2_2_re.to(device), pde_SN_2_3_re.to(device)


    ### ----------EN_2---------- ###  10
    _C = torch.eq(ind_bc, 10)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_EN_2_1_le = u_x + v_y
    pde_EN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_EN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_EN_2_1_re = torch.zeros(pde_EN_2_1_le.shape[0], 1)
    pde_EN_2_2_re = torch.zeros(pde_EN_2_2_le.shape[0], 1)
    pde_EN_2_3_re = torch.zeros(pde_EN_2_3_le.shape[0], 1)
    pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re = pde_EN_2_1_re.to(device), pde_EN_2_2_re.to(device), pde_EN_2_3_re.to(device)


    ### ----------WSE_3---------- ###  11
    _C = torch.eq(ind_bc, 11)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_WSE_3_1_le = u_x + v_y
    pde_WSE_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_WSE_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_WSE_3_1_re = torch.zeros(pde_WSE_3_1_le.shape[0], 1)
    pde_WSE_3_2_re = torch.zeros(pde_WSE_3_2_le.shape[0], 1)
    pde_WSE_3_3_re = torch.zeros(pde_WSE_3_3_le.shape[0], 1)
    pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re = pde_WSE_3_1_re.to(device), pde_WSE_3_2_re.to(device), pde_WSE_3_3_re.to(device)


    ### ----------SEN_3---------- ###  12
    _C = torch.eq(ind_bc, 12)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_SEN_3_1_le = u_x + v_y
    pde_SEN_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_SEN_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_SEN_3_1_re = torch.zeros(pde_SEN_3_1_le.shape[0], 1)
    pde_SEN_3_2_re = torch.zeros(pde_SEN_3_2_le.shape[0], 1)
    pde_SEN_3_3_re = torch.zeros(pde_SEN_3_3_le.shape[0], 1)
    pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re = pde_SEN_3_1_re.to(device), pde_SEN_3_2_re.to(device), pde_SEN_3_3_re.to(device)


    ### ----------ENW_3---------- ###  13
    _C = torch.eq(ind_bc, 13)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = -uC, -vC, pC
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_ENW_3_1_le = u_x + v_y
    pde_ENW_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_ENW_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_ENW_3_1_re = torch.zeros(pde_ENW_3_1_le.shape[0], 1)
    pde_ENW_3_2_re = torch.zeros(pde_ENW_3_2_le.shape[0], 1)
    pde_ENW_3_3_re = torch.zeros(pde_ENW_3_3_le.shape[0], 1)
    pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re = pde_ENW_3_1_re.to(device), pde_ENW_3_2_re.to(device), pde_ENW_3_3_re.to(device)


    ### ----------NWS_3---------- ###  14
    _C = torch.eq(ind_bc, 14)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC
    uS, vS, pS = -uC, -vC, pC

    # numerical differentiation PDE (n-pde)
    u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
    u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
    v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
    p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

    UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
    UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

    # compute PDE matrix
    pde_NWS_3_1_le = u_x + v_y
    pde_NWS_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
    pde_NWS_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
    pde_NWS_3_1_re = torch.zeros(pde_NWS_3_1_le.shape[0], 1)
    pde_NWS_3_2_re = torch.zeros(pde_NWS_3_2_le.shape[0], 1)
    pde_NWS_3_3_re = torch.zeros(pde_NWS_3_3_le.shape[0], 1)
    pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re = pde_NWS_3_1_re.to(device), pde_NWS_3_2_re.to(device), pde_NWS_3_3_re.to(device)


    ### ----------left---------- ###  15
    _C = torch.eq(ind_bc, 15)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    uW, vW, pW = uC, vC, pC

    # compute PDE matrix
    pde_left_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_1_re = torch.zeros(pde_left_1_le.shape[0], 1)
    pde_left_2_re = torch.zeros(pde_left_2_le.shape[0], 1)
    pde_left_3_re = torch.zeros(pde_left_3_le.shape[0], 1)
    pde_left_1_re, pde_left_2_re, pde_left_3_re = pde_left_1_re.to(device), pde_left_2_re.to(device), pde_left_3_re.to(device)

    pde_left_4_le = uC
    pde_left_5_le = vC
    pde_left_4_re = uC_bc_t
    pde_left_5_re = vC_bc_t


    ### ----------left_E---------- ###  16
    _C = torch.eq(ind_bc, 16)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_E_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_E_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_E_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_E_1_re = torch.zeros(pde_left_E_1_le.shape[0], 1)
    pde_left_E_2_re = torch.zeros(pde_left_E_2_le.shape[0], 1)
    pde_left_E_3_re = torch.zeros(pde_left_E_3_le.shape[0], 1)
    pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re = pde_left_E_1_re.to(device), pde_left_E_2_re.to(device), pde_left_E_3_re.to(device)

    pde_left_E_4_le = uC
    pde_left_E_5_le = vC
    pde_left_E_4_re = uC_bc_t
    pde_left_E_5_re = vC_bc_t


    ### ----------left_N---------- ###  17
    _C = torch.eq(ind_bc, 17)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uW, vW, pW = uC, vC, pC
    uN, vN, pN = -uC, -vC, pC

    # compute PDE matrix
    pde_left_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_N_1_re = torch.zeros(pde_left_N_1_le.shape[0], 1)
    pde_left_N_2_re = torch.zeros(pde_left_N_2_le.shape[0], 1)
    pde_left_N_3_re = torch.zeros(pde_left_N_3_le.shape[0], 1)
    pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re = pde_left_N_1_re.to(device), pde_left_N_2_re.to(device), pde_left_N_3_re.to(device)

    pde_left_N_4_le = uC
    pde_left_N_5_le = vC
    pde_left_N_4_re = uC_bc_t
    pde_left_N_5_re = vC_bc_t


    ### ----------left_NE---------- ###  18
    _C = torch.eq(ind_bc, 18)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uN, vN, pN = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_NE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_NE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_NE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_NE_1_re = torch.zeros(pde_left_NE_1_le.shape[0], 1)
    pde_left_NE_2_re = torch.zeros(pde_left_NE_2_le.shape[0], 1)
    pde_left_NE_3_re = torch.zeros(pde_left_NE_3_le.shape[0], 1)
    pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re = pde_left_NE_1_re.to(device), pde_left_NE_2_re.to(device), pde_left_NE_3_re.to(device)

    pde_left_NE_4_le = uC
    pde_left_NE_5_le = vC
    pde_left_NE_4_re = uC_bc_t
    pde_left_NE_5_re = vC_bc_t


    ### ----------left_S---------- ###  19
    _C = torch.eq(ind_bc, 19)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uW, vW, pW = uC, vC, pC
    uS, vS, pS = -uC, -vC, pC

    # compute PDE matrix
    pde_left_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_S_1_re = torch.zeros(pde_left_S_1_le.shape[0], 1)
    pde_left_S_2_re = torch.zeros(pde_left_S_2_le.shape[0], 1)
    pde_left_S_3_re = torch.zeros(pde_left_S_3_le.shape[0], 1)
    pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re = pde_left_S_1_re.to(device), pde_left_S_2_re.to(device), pde_left_S_3_re.to(device)

    pde_left_S_4_le = uC
    pde_left_S_5_le = vC
    pde_left_S_4_re = uC_bc_t
    pde_left_S_5_re = vC_bc_t


    ### ----------left_SE---------- ###  20
    _C = torch.eq(ind_bc, 20)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (uW + uC)/2 = 1, uW = 2 - uC (not working)
    # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
    uW, vW, pW = uC, vC, pC
    uS, vS, pS = -uC, -vC, pC
    uE, vE, pE = -uC, -vC, pC

    # compute PDE matrix
    pde_left_SE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_left_SE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_left_SE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_left_SE_1_re = torch.zeros(pde_left_SE_1_le.shape[0], 1)
    pde_left_SE_2_re = torch.zeros(pde_left_SE_2_le.shape[0], 1)
    pde_left_SE_3_re = torch.zeros(pde_left_SE_3_le.shape[0], 1)
    pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re = pde_left_SE_1_re.to(device), pde_left_SE_2_re.to(device), pde_left_SE_3_re.to(device)

    pde_left_SE_4_le = uC
    pde_left_SE_5_le = vC
    pde_left_SE_4_re = uC_bc_t
    pde_left_SE_5_re = vC_bc_t


    ### ----------right---------- ###  21
    _C = torch.eq(ind_bc, 21)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0 (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    uE, vE, pE = uC, vC, -pC

    # compute PDE matrix
    pde_right_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_1_re = torch.zeros(pde_right_1_le.shape[0], 1)
    pde_right_2_re = torch.zeros(pde_right_2_le.shape[0], 1)
    pde_right_3_re = torch.zeros(pde_right_3_le.shape[0], 1)
    pde_right_1_re, pde_right_2_re, pde_right_3_re = pde_right_1_re.to(device), pde_right_2_re.to(device), pde_right_3_re.to(device)
    pde_right_4_le = pC
    pde_right_4_re = pC_bc_t


    ### ----------right_W---------- ###  22
    _C = torch.eq(ind_bc, 22)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_W_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_W_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_W_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_W_1_re = torch.zeros(pde_right_W_1_le.shape[0], 1)
    pde_right_W_2_re = torch.zeros(pde_right_W_2_le.shape[0], 1)
    pde_right_W_3_re = torch.zeros(pde_right_W_3_le.shape[0], 1)
    pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re = pde_right_W_1_re.to(device), pde_right_W_2_re.to(device), pde_right_W_3_re.to(device)
    pde_right_W_4_le = pC
    pde_right_W_4_re = pC_bc_t


    ### ----------right_N---------- ###  23
    _C = torch.eq(ind_bc, 23)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    uE, vE, pE = uC, vC, -pC
    uN, vN, pN = -uC, -vC, pC

    # compute PDE matrix
    pde_right_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_N_1_re = torch.zeros(pde_right_N_1_le.shape[0], 1)
    pde_right_N_2_re = torch.zeros(pde_right_N_2_le.shape[0], 1)
    pde_right_N_3_re = torch.zeros(pde_right_N_3_le.shape[0], 1)
    pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re = pde_right_N_1_re.to(device), pde_right_N_2_re.to(device), pde_right_N_3_re.to(device)
    pde_right_N_4_le = pC
    pde_right_N_4_re = pC_bc_t


    ### ----------right_NW---------- ###  24
    _C = torch.eq(ind_bc, 24)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uN, vN, pN = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_NW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_NW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_NW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_NW_1_re = torch.zeros(pde_right_NW_1_le.shape[0], 1)
    pde_right_NW_2_re = torch.zeros(pde_right_NW_2_le.shape[0], 1)
    pde_right_NW_3_re = torch.zeros(pde_right_NW_3_le.shape[0], 1)
    pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re = pde_right_NW_1_re.to(device), pde_right_NW_2_re.to(device), pde_right_NW_3_re.to(device)
    pde_right_NW_4_le = pC
    pde_right_NW_4_re = pC_bc_t


    ### ----------right_S---------- ###  25
    _C = torch.eq(ind_bc, 25)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    uE, vE, pE = uC, vC, -pC
    uS, vS, pS = -uC, -vC, pC

    # compute PDE matrix
    pde_right_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_S_1_re = torch.zeros(pde_right_S_1_le.shape[0], 1)
    pde_right_S_2_re = torch.zeros(pde_right_S_2_le.shape[0], 1)
    pde_right_S_3_re = torch.zeros(pde_right_S_3_le.shape[0], 1)
    pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re = pde_right_S_1_re.to(device), pde_right_S_2_re.to(device), pde_right_S_3_re.to(device)
    pde_right_S_4_le = pC
    pde_right_S_4_re = pC_bc_t


    ### ----------right_SW---------- ###  26
    _C = torch.eq(ind_bc, 26)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
    # uE, vE = M_u[_E, :], M_v[_E, :]
    # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
    uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
    # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

    # u0C, v0C = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)
    u0C, v0C = torch.ones(uC.shape[0], 1).to(device), torch.zeros(vC.shape[0], 1).to(device)

    # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

    uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

    # (pE + pC)/2 = 0  (not working)
    # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
    # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
    # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
    uE, vE, pE = uC, vC, -pC
    uS, vS, pS = -uC, -vC, pC
    uW, vW, pW = -uC, -vC, pC

    # compute PDE matrix
    pde_right_SW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
    pde_right_SW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
    pde_right_SW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
    pde_right_SW_1_re = torch.zeros(pde_right_SW_1_le.shape[0], 1)
    pde_right_SW_2_re = torch.zeros(pde_right_SW_2_le.shape[0], 1)
    pde_right_SW_3_re = torch.zeros(pde_right_SW_3_le.shape[0], 1)
    pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re = pde_right_SW_1_re.to(device), pde_right_SW_2_re.to(device), pde_right_SW_3_re.to(device)
    pde_right_SW_4_le = pC
    pde_right_SW_4_re = pC_bc_t


    # lamda_bc = 1e2
    lamda_bc = 5

    # pde_M_normal_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le], axis=0)
    # pde_M_bc_le = lamda_bc*torch.cat([pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
    #                                   pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
    #                                   pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
    #                                   pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
    #                                   pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
    #                                   pde_left_1_le, pde_left_2_le, pde_left_3_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, \
    #                                   pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, \
    #                                   pde_right_1_le, pde_right_2_le, pde_right_3_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, \
    #                                   pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le], axis=0)
    # pde_M_le = torch.cat([pde_M_normal_le, pde_M_bc_le], axis=0)

    # pde_M_normal_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re], axis=0)
    # pde_M_bc_re = lamda_bc*torch.cat([pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
    #                                   pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
    #                                   pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
    #                                   pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
    #                                   pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
    #                                   pde_left_1_re, pde_left_2_re, pde_left_3_re, pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re, pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re, \
    #                                   pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re, pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re, pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re, \
    #                                   pde_right_1_re, pde_right_2_re, pde_right_3_re, pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re, pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re, \
    #                                   pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re, pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re, pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re], axis=0)
    # pde_M_re = torch.cat([pde_M_normal_re, pde_M_bc_re], axis=0)


    # pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
    #                       pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
    #                       pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
    #                       pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
    #                       pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
    #                       lamda_bc*pde_left_1_le, lamda_bc*pde_left_2_le, lamda_bc*pde_left_3_le, lamda_bc*pde_left_E_1_le, lamda_bc*pde_left_E_2_le, lamda_bc*pde_left_E_3_le, lamda_bc*pde_left_N_1_le, lamda_bc*pde_left_N_2_le, lamda_bc*pde_left_N_3_le, \
    #                       lamda_bc*pde_left_NE_1_le, lamda_bc*pde_left_NE_2_le, lamda_bc*pde_left_NE_3_le, lamda_bc*pde_left_S_1_le, lamda_bc*pde_left_S_2_le, lamda_bc*pde_left_S_3_le, lamda_bc*pde_left_SE_1_le, lamda_bc*pde_left_SE_2_le, lamda_bc*pde_left_SE_3_le, \
    #                       lamda_bc*pde_right_1_le, lamda_bc*pde_right_2_le, lamda_bc*pde_right_3_le, lamda_bc*pde_right_W_1_le, lamda_bc*pde_right_W_2_le, lamda_bc*pde_right_W_3_le, lamda_bc*pde_right_N_1_le, lamda_bc*pde_right_N_2_le, lamda_bc*pde_right_N_3_le, \
    #                       lamda_bc*pde_right_NW_1_le, lamda_bc*pde_right_NW_2_le, lamda_bc*pde_right_NW_3_le, lamda_bc*pde_right_S_1_le, lamda_bc*pde_right_S_2_le, lamda_bc*pde_right_S_3_le, lamda_bc*pde_right_SW_1_le, lamda_bc*pde_right_SW_2_le, lamda_bc*pde_right_SW_3_le], axis=0)


    # pde_M_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re, pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
    #                       pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
    #                       pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
    #                       pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
    #                       pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
    #                       lamda_bc*pde_left_1_re, lamda_bc*pde_left_2_re, lamda_bc*pde_left_3_re, lamda_bc*pde_left_E_1_re, lamda_bc*pde_left_E_2_re, lamda_bc*pde_left_E_3_re, lamda_bc*pde_left_N_1_re, lamda_bc*pde_left_N_2_re, lamda_bc*pde_left_N_3_re, \
    #                       lamda_bc*pde_left_NE_1_re, lamda_bc*pde_left_NE_2_re, lamda_bc*pde_left_NE_3_re, lamda_bc*pde_left_S_1_re, lamda_bc*pde_left_S_2_re, lamda_bc*pde_left_S_3_re, lamda_bc*pde_left_SE_1_re, lamda_bc*pde_left_SE_2_re, lamda_bc*pde_left_SE_3_re, \
    #                       lamda_bc*pde_right_1_re, lamda_bc*pde_right_2_re, lamda_bc*pde_right_3_re, lamda_bc*pde_right_W_1_re, lamda_bc*pde_right_W_2_re, lamda_bc*pde_right_W_3_re, lamda_bc*pde_right_N_1_re, lamda_bc*pde_right_N_2_re, lamda_bc*pde_right_N_3_re, \
    #                       lamda_bc*pde_right_NW_1_re, lamda_bc*pde_right_NW_2_re, lamda_bc*pde_right_NW_3_re, lamda_bc*pde_right_S_1_re, lamda_bc*pde_right_S_2_re, lamda_bc*pde_right_S_3_re, lamda_bc*pde_right_SW_1_re, lamda_bc*pde_right_SW_2_re, lamda_bc*pde_right_SW_3_re], axis=0)


    pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
                          pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
                          pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
                          pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
                          pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
                          pde_left_1_le, pde_left_2_le, pde_left_3_le, lamda_bc*pde_left_4_le, lamda_bc*pde_left_5_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, lamda_bc*pde_left_E_4_le, lamda_bc*pde_left_E_5_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, lamda_bc*pde_left_N_4_le, lamda_bc*pde_left_N_5_le, \
                          pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, lamda_bc*pde_left_NE_4_le, lamda_bc*pde_left_NE_5_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, lamda_bc*pde_left_S_4_le, lamda_bc*pde_left_S_5_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, lamda_bc*pde_left_SE_4_le, lamda_bc*pde_left_SE_5_le, \
                          pde_right_1_le, pde_right_2_le, pde_right_3_le, lamda_bc*pde_right_4_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, lamda_bc*pde_right_W_4_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, lamda_bc*pde_right_N_4_le, \
                          pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, lamda_bc*pde_right_NW_4_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, lamda_bc*pde_right_S_4_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le, lamda_bc*pde_right_SW_4_le], axis=0)


    pde_M_re = torch.cat([pde_normal_1_re, pde_normal_2_re, pde_normal_3_re, pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re, pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re, \
                          pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re, pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re, pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re, \
                          pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re, pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re, pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re, \
                          pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re, pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re, pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re, \
                          pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re, pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re, pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re, \
                          pde_left_1_re, pde_left_2_re, pde_left_3_re, lamda_bc*pde_left_4_re, lamda_bc*pde_left_5_re, pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re, lamda_bc*pde_left_E_4_re, lamda_bc*pde_left_E_5_re, pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re, lamda_bc*pde_left_N_4_re, lamda_bc*pde_left_N_5_re,  \
                          pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re, lamda_bc*pde_left_NE_4_re, lamda_bc*pde_left_NE_5_re, pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re, lamda_bc*pde_left_S_4_re, lamda_bc*pde_left_S_5_re, pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re, lamda_bc*pde_left_SE_4_re, lamda_bc*pde_left_SE_5_re,  \
                          pde_right_1_re, pde_right_2_re, pde_right_3_re, lamda_bc*pde_right_4_re, pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re, lamda_bc*pde_right_W_4_re, pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re, lamda_bc*pde_right_N_4_re, \
                          pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re, lamda_bc*pde_right_NW_4_re, pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re, lamda_bc*pde_right_S_4_re, pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re, lamda_bc*pde_right_SW_4_re], axis=0)
    
    pde_M_le = pde_M_le.double()  # change to 64 bits
    pde_M_re = pde_M_re.double()  # change to 64 bits

    # start = time.time()
    kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re
    # end = time.time()
    # runtime = end - start

    # iteration to calculate the nonlinear term
    for i in np.arange(0):
        ### ------normal PDE------ ###  0
        _C = torch.eq(ind_bc, 0)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :] 
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters
        u0C_old, v0C_old = u0C, v0C

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_normal_1_le = u_x + v_y
        pde_normal_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_normal_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_normal_1_re = torch.zeros(pde_normal_1_le.shape[0], 1)
        # pde_normal_2_re = torch.zeros(pde_normal_2_le.shape[0], 1)
        # pde_normal_3_re = torch.zeros(pde_normal_3_le.shape[0], 1)
        # pde_normal_1_re, pde_normal_2_re, pde_normal_3_re = pde_normal_1_re.to(device), pde_normal_2_re.to(device), pde_normal_3_re.to(device)


        ### ----------W_1---------- ###  1
        _C = torch.eq(ind_bc, 1)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_W_1_1_le = u_x + v_y
        pde_W_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_W_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_W_1_1_re = torch.zeros(pde_W_1_1_le.shape[0], 1)
        # pde_W_1_2_re = torch.zeros(pde_W_1_2_le.shape[0], 1)
        # pde_W_1_3_re = torch.zeros(pde_W_1_3_le.shape[0], 1)
        # pde_W_1_1_re, pde_W_1_2_re, pde_W_1_3_re = pde_W_1_1_re.to(device), pde_W_1_2_re.to(device), pde_W_1_3_re.to(device)


        ### ----------S_1---------- ###  2
        _C = torch.eq(ind_bc, 2)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_S_1_1_le = u_x + v_y
        pde_S_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_S_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_S_1_1_re = torch.zeros(pde_S_1_1_le.shape[0], 1)
        # pde_S_1_2_re = torch.zeros(pde_S_1_2_le.shape[0], 1)
        # pde_S_1_3_re = torch.zeros(pde_S_1_3_le.shape[0], 1)
        # pde_S_1_1_re, pde_S_1_2_re, pde_S_1_3_re = pde_S_1_1_re.to(device), pde_S_1_2_re.to(device), pde_S_1_3_re.to(device)


        ### ----------E_1---------- ###  3
        _C = torch.eq(ind_bc, 3)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_E_1_1_le = u_x + v_y
        pde_E_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_E_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_E_1_1_re = torch.zeros(pde_E_1_1_le.shape[0], 1)
        # pde_E_1_2_re = torch.zeros(pde_E_1_2_le.shape[0], 1)
        # pde_E_1_3_re = torch.zeros(pde_E_1_3_le.shape[0], 1)
        # pde_E_1_1_re, pde_E_1_2_re, pde_E_1_3_re = pde_E_1_1_re.to(device), pde_E_1_2_re.to(device), pde_E_1_3_re.to(device)


        ### ----------N_1---------- ###  4
        _C = torch.eq(ind_bc, 4)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_N_1_1_le = u_x + v_y
        pde_N_1_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_N_1_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_N_1_1_re = torch.zeros(pde_N_1_1_le.shape[0], 1)
        # pde_N_1_2_re = torch.zeros(pde_N_1_2_le.shape[0], 1)
        # pde_N_1_3_re = torch.zeros(pde_N_1_3_le.shape[0], 1)
        # pde_N_1_1_re, pde_N_1_2_re, pde_N_1_3_re = pde_N_1_1_re.to(device), pde_N_1_2_re.to(device), pde_N_1_3_re.to(device)


        ### ----------WS_2---------- ###  5
        _C = torch.eq(ind_bc, 5)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WS_2_1_le = u_x + v_y
        pde_WS_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WS_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WS_2_1_re = torch.zeros(pde_WS_2_1_le.shape[0], 1)
        # pde_WS_2_2_re = torch.zeros(pde_WS_2_2_le.shape[0], 1)
        # pde_WS_2_3_re = torch.zeros(pde_WS_2_3_le.shape[0], 1)
        # pde_WS_2_1_re, pde_WS_2_2_re, pde_WS_2_3_re = pde_WS_2_1_re.to(device), pde_WS_2_2_re.to(device), pde_WS_2_3_re.to(device)


        ### ----------WE_2---------- ###  6
        _C = torch.eq(ind_bc, 6)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WE_2_1_le = u_x + v_y
        pde_WE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WE_2_1_re = torch.zeros(pde_WE_2_1_le.shape[0], 1)
        # pde_WE_2_2_re = torch.zeros(pde_WE_2_2_le.shape[0], 1)
        # pde_WE_2_3_re = torch.zeros(pde_WE_2_3_le.shape[0], 1)
        # pde_WE_2_1_re, pde_WE_2_2_re, pde_WE_2_3_re = pde_WE_2_1_re.to(device), pde_WE_2_2_re.to(device), pde_WE_2_3_re.to(device)


        ### ----------WN_2---------- ###  7
        _C = torch.eq(ind_bc, 7)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uW, vW, pW = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WN_2_1_le = u_x + v_y
        pde_WN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WN_2_1_re = torch.zeros(pde_WN_2_1_le.shape[0], 1)
        # pde_WN_2_2_re = torch.zeros(pde_WN_2_2_le.shape[0], 1)
        # pde_WN_2_3_re = torch.zeros(pde_WN_2_3_le.shape[0], 1)
        # pde_WN_2_1_re, pde_WN_2_2_re, pde_WN_2_3_re = pde_WN_2_1_re.to(device), pde_WN_2_2_re.to(device), pde_WN_2_3_re.to(device)


        ### ----------SE_2---------- ###  8
        _C = torch.eq(ind_bc, 8)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SE_2_1_le = u_x + v_y
        pde_SE_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SE_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SE_2_1_re = torch.zeros(pde_SE_2_1_le.shape[0], 1)
        # pde_SE_2_2_re = torch.zeros(pde_SE_2_2_le.shape[0], 1)
        # pde_SE_2_3_re = torch.zeros(pde_SE_2_3_le.shape[0], 1)
        # pde_SE_2_1_re, pde_SE_2_2_re, pde_SE_2_3_re = pde_SE_2_1_re.to(device), pde_SE_2_2_re.to(device), pde_SE_2_3_re.to(device)


        ### ----------SN_2---------- ###  9
        _C = torch.eq(ind_bc, 9)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SN_2_1_le = u_x + v_y
        pde_SN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SN_2_1_re = torch.zeros(pde_SN_2_1_le.shape[0], 1)
        # pde_SN_2_2_re = torch.zeros(pde_SN_2_2_le.shape[0], 1)
        # pde_SN_2_3_re = torch.zeros(pde_SN_2_3_le.shape[0], 1)
        # pde_SN_2_1_re, pde_SN_2_2_re, pde_SN_2_3_re = pde_SN_2_1_re.to(device), pde_SN_2_2_re.to(device), pde_SN_2_3_re.to(device)


        ### ----------EN_2---------- ###  10
        _C = torch.eq(ind_bc, 10)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_EN_2_1_le = u_x + v_y
        pde_EN_2_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_EN_2_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_EN_2_1_re = torch.zeros(pde_EN_2_1_le.shape[0], 1)
        # pde_EN_2_2_re = torch.zeros(pde_EN_2_2_le.shape[0], 1)
        # pde_EN_2_3_re = torch.zeros(pde_EN_2_3_le.shape[0], 1)
        # pde_EN_2_1_re, pde_EN_2_2_re, pde_EN_2_3_re = pde_EN_2_1_re.to(device), pde_EN_2_2_re.to(device), pde_EN_2_3_re.to(device)


        ### ----------WSE_3---------- ###  11
        _C = torch.eq(ind_bc, 11)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_WSE_3_1_le = u_x + v_y
        pde_WSE_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_WSE_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_WSE_3_1_re = torch.zeros(pde_WSE_3_1_le.shape[0], 1)
        # pde_WSE_3_2_re = torch.zeros(pde_WSE_3_2_le.shape[0], 1)
        # pde_WSE_3_3_re = torch.zeros(pde_WSE_3_3_le.shape[0], 1)
        # pde_WSE_3_1_re, pde_WSE_3_2_re, pde_WSE_3_3_re = pde_WSE_3_1_re.to(device), pde_WSE_3_2_re.to(device), pde_WSE_3_3_re.to(device)


        ### ----------SEN_3---------- ###  12
        _C = torch.eq(ind_bc, 12)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_SEN_3_1_le = u_x + v_y
        pde_SEN_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_SEN_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_SEN_3_1_re = torch.zeros(pde_SEN_3_1_le.shape[0], 1)
        # pde_SEN_3_2_re = torch.zeros(pde_SEN_3_2_le.shape[0], 1)
        # pde_SEN_3_3_re = torch.zeros(pde_SEN_3_3_le.shape[0], 1)
        # pde_SEN_3_1_re, pde_SEN_3_2_re, pde_SEN_3_3_re = pde_SEN_3_1_re.to(device), pde_SEN_3_2_re.to(device), pde_SEN_3_3_re.to(device)


        ### ----------ENW_3---------- ###  13
        _C = torch.eq(ind_bc, 13)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = -uC, -vC, pC
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_ENW_3_1_le = u_x + v_y
        pde_ENW_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_ENW_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_ENW_3_1_re = torch.zeros(pde_ENW_3_1_le.shape[0], 1)
        # pde_ENW_3_2_re = torch.zeros(pde_ENW_3_2_le.shape[0], 1)
        # pde_ENW_3_3_re = torch.zeros(pde_ENW_3_3_le.shape[0], 1)
        # pde_ENW_3_1_re, pde_ENW_3_2_re, pde_ENW_3_3_re = pde_ENW_3_1_re.to(device), pde_ENW_3_2_re.to(device), pde_ENW_3_3_re.to(device)


        ### ----------NWS_3---------- ###  14
        _C = torch.eq(ind_bc, 14)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC
        uS, vS, pS = -uC, -vC, pC

        # numerical differentiation PDE (n-pde)
        u_x, v_y = (uE - uW)/(2*dx), (vN - vS)/(2*dy)
        u_xx, u_yy = (uE - 2*uC + uW)/(dx*dx), (uN - 2*uC + uS)/(dy*dy)
        v_xx, v_yy = (vE - 2*vC + vW)/(dx*dx), (vN - 2*vC + vS)/(dy*dy)
        p_x, p_y = (pE - pW)/(2*dx), (pN - pS)/(2*dy)

        UUx, VUy = (u0C*uE - u0C*uW)/(2*dx), (v0C*uN - v0C*uS)/(2*dy)
        UVx, VVy = (u0C*vE - u0C*vW)/(2*dx), (v0C*vN - v0C*vS)/(2*dy)

        # compute PDE matrix
        # pde_NWS_3_1_le = u_x + v_y
        pde_NWS_3_2_le = UUx + VUy + p_x - 1/Re*(u_xx + u_yy)
        pde_NWS_3_3_le = UVx + VVy + p_y - 1/Re*(v_xx + v_yy)
        # pde_NWS_3_1_re = torch.zeros(pde_NWS_3_1_le.shape[0], 1)
        # pde_NWS_3_2_re = torch.zeros(pde_NWS_3_2_le.shape[0], 1)
        # pde_NWS_3_3_re = torch.zeros(pde_NWS_3_3_le.shape[0], 1)
        # pde_NWS_3_1_re, pde_NWS_3_2_re, pde_NWS_3_3_re = pde_NWS_3_1_re.to(device), pde_NWS_3_2_re.to(device), pde_NWS_3_3_re.to(device)


        ### ----------left---------- ###  15
        _C = torch.eq(ind_bc, 15)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        uW, vW, pW = uC, vC, pC

        # compute PDE matrix
        # pde_left_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_1_re = torch.zeros(pde_left_1_le.shape[0], 1)
        # pde_left_2_re = torch.zeros(pde_left_2_le.shape[0], 1)
        # pde_left_3_re = torch.zeros(pde_left_3_le.shape[0], 1)
        # pde_left_1_re, pde_left_2_re, pde_left_3_re = pde_left_1_re.to(device), pde_left_2_re.to(device), pde_left_3_re.to(device)

        # pde_left_4_le = uC
        # pde_left_5_le = vC
        # pde_left_4_re = uC_bc_t
        # pde_left_5_re = vC_bc_t


        ### ----------left_E---------- ###  16
        _C = torch.eq(ind_bc, 16)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_E_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_E_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_E_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_E_1_re = torch.zeros(pde_left_E_1_le.shape[0], 1)
        # pde_left_E_2_re = torch.zeros(pde_left_E_2_le.shape[0], 1)
        # pde_left_E_3_re = torch.zeros(pde_left_E_3_le.shape[0], 1)
        # pde_left_E_1_re, pde_left_E_2_re, pde_left_E_3_re = pde_left_E_1_re.to(device), pde_left_E_2_re.to(device), pde_left_E_3_re.to(device)

        # pde_left_E_4_le = uC
        # pde_left_E_5_le = vC
        # pde_left_E_4_re = uC_bc_t
        # pde_left_E_5_re = vC_bc_t


        ### ----------left_N---------- ###  17
        _C = torch.eq(ind_bc, 17)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uW, vW, pW = uC, vC, pC
        uN, vN, pN = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_N_1_re = torch.zeros(pde_left_N_1_le.shape[0], 1)
        # pde_left_N_2_re = torch.zeros(pde_left_N_2_le.shape[0], 1)
        # pde_left_N_3_re = torch.zeros(pde_left_N_3_le.shape[0], 1)
        # pde_left_N_1_re, pde_left_N_2_re, pde_left_N_3_re = pde_left_N_1_re.to(device), pde_left_N_2_re.to(device), pde_left_N_3_re.to(device)

        # pde_left_N_4_le = uC
        # pde_left_N_5_le = vC
        # pde_left_N_4_re = uC_bc_t
        # pde_left_N_5_re = vC_bc_t


        ### ----------left_NE---------- ###  18
        _C = torch.eq(ind_bc, 18)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uN, vN, pN = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_NE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_NE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_NE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_NE_1_re = torch.zeros(pde_left_NE_1_le.shape[0], 1)
        # pde_left_NE_2_re = torch.zeros(pde_left_NE_2_le.shape[0], 1)
        # pde_left_NE_3_re = torch.zeros(pde_left_NE_3_le.shape[0], 1)
        # pde_left_NE_1_re, pde_left_NE_2_re, pde_left_NE_3_re = pde_left_NE_1_re.to(device), pde_left_NE_2_re.to(device), pde_left_NE_3_re.to(device)

        # pde_left_NE_4_le = uC
        # pde_left_NE_5_le = vC
        # pde_left_NE_4_re = uC_bc_t
        # pde_left_NE_5_re = vC_bc_t


        ### ----------left_S---------- ###  19
        _C = torch.eq(ind_bc, 19)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uW, vW, pW = uC, vC, pC
        uS, vS, pS = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_S_1_re = torch.zeros(pde_left_S_1_le.shape[0], 1)
        # pde_left_S_2_re = torch.zeros(pde_left_S_2_le.shape[0], 1)
        # pde_left_S_3_re = torch.zeros(pde_left_S_3_le.shape[0], 1)
        # pde_left_S_1_re, pde_left_S_2_re, pde_left_S_3_re = pde_left_S_1_re.to(device), pde_left_S_2_re.to(device), pde_left_S_3_re.to(device)

        # pde_left_S_4_le = uC
        # pde_left_S_5_le = vC
        # pde_left_S_4_re = uC_bc_t
        # pde_left_S_5_re = vC_bc_t


        ### ----------left_SE---------- ###  20
        _C = torch.eq(ind_bc, 20)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE, pE = M_u[_E, :], M_v[_E, :], M_p[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (uW + uC)/2 = 1, uW = 2 - uC (not working)
        # (uW + uC)/2 = uC_bc, (vW + vC)/2 = vC_bc, (pW + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uE + uC)/2 = 0, (vE + vC)/2 = 0, (pE - pC)/(dx) = 0
        uW, vW, pW = uC, vC, pC
        uS, vS, pS = -uC, -vC, pC
        uE, vE, pE = -uC, -vC, pC

        # compute PDE matrix
        # pde_left_SE_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_left_SE_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_left_SE_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_left_SE_1_re = torch.zeros(pde_left_SE_1_le.shape[0], 1)
        # pde_left_SE_2_re = torch.zeros(pde_left_SE_2_le.shape[0], 1)
        # pde_left_SE_3_re = torch.zeros(pde_left_SE_3_le.shape[0], 1)
        # pde_left_SE_1_re, pde_left_SE_2_re, pde_left_SE_3_re = pde_left_SE_1_re.to(device), pde_left_SE_2_re.to(device), pde_left_SE_3_re.to(device)

        # pde_left_SE_4_le = uC
        # pde_left_SE_5_le = vC
        # pde_left_SE_4_re = uC_bc_t
        # pde_left_SE_5_re = vC_bc_t


        ### ----------right---------- ###  21
        _C = torch.eq(ind_bc, 21)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0 (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        uE, vE, pE = uC, vC, -pC

        # compute PDE matrix
        # pde_right_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_1_re = torch.zeros(pde_right_1_le.shape[0], 1)
        # pde_right_2_re = torch.zeros(pde_right_2_le.shape[0], 1)
        # pde_right_3_re = torch.zeros(pde_right_3_le.shape[0], 1)
        # pde_right_1_re, pde_right_2_re, pde_right_3_re = pde_right_1_re.to(device), pde_right_2_re.to(device), pde_right_3_re.to(device)
        # pde_right_4_le = pC
        # pde_right_4_re = pC_bc_t


        ### ----------right_W---------- ###  22
        _C = torch.eq(ind_bc, 22)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_W_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_W_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_W_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_W_1_re = torch.zeros(pde_right_W_1_le.shape[0], 1)
        # pde_right_W_2_re = torch.zeros(pde_right_W_2_le.shape[0], 1)
        # pde_right_W_3_re = torch.zeros(pde_right_W_3_le.shape[0], 1)
        # pde_right_W_1_re, pde_right_W_2_re, pde_right_W_3_re = pde_right_W_1_re.to(device), pde_right_W_2_re.to(device), pde_right_W_3_re.to(device)
        # pde_right_W_4_le = pC
        # pde_right_W_4_re = pC_bc_t


        ### ----------right_N---------- ###  23
        _C = torch.eq(ind_bc, 23)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        uE, vE, pE = uC, vC, -pC
        uN, vN, pN = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_N_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_N_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_N_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_N_1_re = torch.zeros(pde_right_N_1_le.shape[0], 1)
        # pde_right_N_2_re = torch.zeros(pde_right_N_2_le.shape[0], 1)
        # pde_right_N_3_re = torch.zeros(pde_right_N_3_le.shape[0], 1)
        # pde_right_N_1_re, pde_right_N_2_re, pde_right_N_3_re = pde_right_N_1_re.to(device), pde_right_N_2_re.to(device), pde_right_N_3_re.to(device)
        # pde_right_N_4_le = pC
        # pde_right_N_4_re = pC_bc_t


        ### ----------right_NW---------- ###  24
        _C = torch.eq(ind_bc, 24)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        # uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uN + uC)/2 = 0, (vN + vC)/2 = 0, (pN - pC)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uN, vN, pN = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_NW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_NW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_NW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_NW_1_re = torch.zeros(pde_right_NW_1_le.shape[0], 1)
        # pde_right_NW_2_re = torch.zeros(pde_right_NW_2_le.shape[0], 1)
        # pde_right_NW_3_re = torch.zeros(pde_right_NW_3_le.shape[0], 1)
        # pde_right_NW_1_re, pde_right_NW_2_re, pde_right_NW_3_re = pde_right_NW_1_re.to(device), pde_right_NW_2_re.to(device), pde_right_NW_3_re.to(device)
        # pde_right_NW_4_le = pC
        # pde_right_NW_4_re = pC_bc_t


        ### ----------right_S---------- ###  25
        _C = torch.eq(ind_bc, 25)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        uE, vE, pE = uC, vC, -pC
        uS, vS, pS = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_S_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_S_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_S_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_S_1_re = torch.zeros(pde_right_S_1_le.shape[0], 1)
        # pde_right_S_2_re = torch.zeros(pde_right_S_2_le.shape[0], 1)
        # pde_right_S_3_re = torch.zeros(pde_right_S_3_le.shape[0], 1)
        # pde_right_S_1_re, pde_right_S_2_re, pde_right_S_3_re = pde_right_S_1_re.to(device), pde_right_S_2_re.to(device), pde_right_S_3_re.to(device)
        # pde_right_S_4_le = pC
        # pde_right_S_4_re = pC_bc_t


        ### ----------right_SW---------- ###  26
        _C = torch.eq(ind_bc, 26)
        _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
        _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
        _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
        _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

        # flatten the index
        _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
        _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

        uC, vC, pC = M_u[_C, :], M_v[_C, :], M_p[_C, :]
        # uE, vE = M_u[_E, :], M_v[_E, :]
        # uW, vW, pW = M_u[_W, :], M_v[_W, :], M_p[_W, :]
        uN, vN, pN = M_u[_N, :], M_v[_N, :], M_p[_N, :]
        # uS, vS, pS = M_u[_S, :], M_v[_S, :], M_p[_S, :]

        u0C, v0C = uC @ kernel_parameters, vC @ kernel_parameters

        # uC_bc, vC_bc, pC_bc = M_u_bc[_C].reshape(-1, 1), M_v_bc[_C].reshape(-1, 1), M_p_bc[_C].reshape(-1, 1)

        # uC_bc_t, vC_bc_t, pC_bc_t = M_u_bc_t[_C].reshape(-1, 1), M_v_bc_t[_C].reshape(-1, 1), M_p_bc_t[_C].reshape(-1, 1)

        # (pE + pC)/2 = 0  (not working)
        # (uE + uC)/2 = uC_bc, (vE + vC)/2 = vC_bc, (pE + pC)/2 = pC_bc
        # (uS + uC)/2 = 0, (vS + vC)/2 = 0, (pC - pS)/(dy) = 0
        # (uW + uC)/2 = 0, (vW + vC)/2 = 0, (pC - pW)/(dx) = 0
        uE, vE, pE = uC, vC, -pC
        uS, vS, pS = -uC, -vC, pC
        uW, vW, pW = -uC, -vC, pC

        # compute PDE matrix
        # pde_right_SW_1_le = (uE - uW)/(2*dx) + (vN - vS)/(2*dy)
        pde_right_SW_2_le = (u0C*uE - u0C*uW)/(2*dx) + (v0C*uN - v0C*uS)/(2*dy) + (pE - pW)/(2*dx) - 1/Re*((uE - 2*uC + uW)/(dx*dx) + (uN - 2*uC + uS)/(dy*dy))
        pde_right_SW_3_le = (u0C*vE - u0C*vW)/(2*dx) + (v0C*vN - v0C*vS)/(2*dy) + (pN - pS)/(2*dy) - 1/Re*((vE - 2*vC + vW)/(dx*dx) + (vN - 2*vC + vS)/(dy*dy))
        # pde_right_SW_1_re = torch.zeros(pde_right_SW_1_le.shape[0], 1)
        # pde_right_SW_2_re = torch.zeros(pde_right_SW_2_le.shape[0], 1)
        # pde_right_SW_3_re = torch.zeros(pde_right_SW_3_le.shape[0], 1)
        # pde_right_SW_1_re, pde_right_SW_2_re, pde_right_SW_3_re = pde_right_SW_1_re.to(device), pde_right_SW_2_re.to(device), pde_right_SW_3_re.to(device)
        # pde_right_SW_4_le = pC
        # pde_right_SW_4_re = pC_bc_t



        pde_M_le = torch.cat([pde_normal_1_le, pde_normal_2_le, pde_normal_3_le, pde_W_1_1_le, pde_W_1_2_le, pde_W_1_3_le, pde_S_1_1_le, pde_S_1_2_le, pde_S_1_3_le, \
                              pde_E_1_1_le, pde_E_1_2_le, pde_E_1_3_le, pde_N_1_1_le, pde_N_1_2_le, pde_N_1_3_le, pde_WS_2_1_le, pde_WS_2_2_le, pde_WS_2_3_le, \
                              pde_WE_2_1_le, pde_WE_2_2_le, pde_WE_2_3_le, pde_WN_2_1_le, pde_WN_2_2_le, pde_WN_2_3_le, pde_SE_2_1_le, pde_SE_2_2_le, pde_SE_2_3_le, \
                              pde_SN_2_1_le, pde_SN_2_2_le, pde_SN_2_3_le, pde_EN_2_1_le, pde_EN_2_2_le, pde_EN_2_3_le, pde_WSE_3_1_le, pde_WSE_3_2_le, pde_WSE_3_3_le, \
                              pde_SEN_3_1_le, pde_SEN_3_2_le, pde_SEN_3_3_le, pde_ENW_3_1_le, pde_ENW_3_2_le, pde_ENW_3_3_le, pde_NWS_3_1_le, pde_NWS_3_2_le, pde_NWS_3_3_le, \
                              pde_left_1_le, pde_left_2_le, pde_left_3_le, lamda_bc*pde_left_4_le, lamda_bc*pde_left_5_le, pde_left_E_1_le, pde_left_E_2_le, pde_left_E_3_le, lamda_bc*pde_left_E_4_le, lamda_bc*pde_left_E_5_le, pde_left_N_1_le, pde_left_N_2_le, pde_left_N_3_le, lamda_bc*pde_left_N_4_le, lamda_bc*pde_left_N_5_le, \
                              pde_left_NE_1_le, pde_left_NE_2_le, pde_left_NE_3_le, lamda_bc*pde_left_NE_4_le, lamda_bc*pde_left_NE_5_le, pde_left_S_1_le, pde_left_S_2_le, pde_left_S_3_le, lamda_bc*pde_left_S_4_le, lamda_bc*pde_left_S_5_le, pde_left_SE_1_le, pde_left_SE_2_le, pde_left_SE_3_le, lamda_bc*pde_left_SE_4_le, lamda_bc*pde_left_SE_5_le, \
                              pde_right_1_le, pde_right_2_le, pde_right_3_le, lamda_bc*pde_right_4_le, pde_right_W_1_le, pde_right_W_2_le, pde_right_W_3_le, lamda_bc*pde_right_W_4_le, pde_right_N_1_le, pde_right_N_2_le, pde_right_N_3_le, lamda_bc*pde_right_N_4_le, \
                              pde_right_NW_1_le, pde_right_NW_2_le, pde_right_NW_3_le, lamda_bc*pde_right_NW_4_le, pde_right_S_1_le, pde_right_S_2_le, pde_right_S_3_le, lamda_bc*pde_right_S_4_le, pde_right_SW_1_le, pde_right_SW_2_le, pde_right_SW_3_le, lamda_bc*pde_right_SW_4_le], axis=0)

        kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re


        # test error
        _C = torch.eq(ind_bc, 0)
        _C = _C.reshape(_C.shape[0], -1)
        _C = _C.flatten()
        uC = M_u[_C, :]
        u0C_new = uC @ kernel_parameters
        error = torch.mean((u0C_new - u0C_old)**2)
        ttt = 1

    M_u = M_u.double()
    M_v = M_v.double()
    M_p = M_p.double()
   
    # pseudo inverse prediction results
    u, v, p = M_u @ kernel_parameters, M_v @ kernel_parameters, M_p @ kernel_parameters
    u, v, p = u.reshape(inputs0.shape[0], -1), v.reshape(inputs0.shape[0], -1), p.reshape(inputs0.shape[0], -1)
    u, v, p = u.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), v.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3]), p.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    u, v, p = torch.unsqueeze(u, axis=1), torch.unsqueeze(v, axis=1), torch.unsqueeze(p, axis=1)
    outputs_pseudo_inverse = torch.concat([u, v, p], axis=1)
    outputs_pseudo_inverse = outputs_pseudo_inverse.float()
    # outputs_pseudo_inverse = outputs_pseudo_inverse.requires_grad_(True)

    # convolution 2d prediction results
    outputs_conv2d = conv2d_head(inputs_conv2d).float()

    ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # return outputs_pseudo_inverse, outputs_conv2d, ssr
    return outputs_pseudo_inverse






def CNN_PINN_Pseudo_Inverse_2D_NS_Heat_DirichletBC(inputs, inputs0, conv2d_head, targets):

    inputs_conv2d = inputs

    n_kernel_size = 7
    n_p = int(np.floor(n_kernel_size/2))
    # kernel size: 1*1, 3*3, 5*5, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

    # computational boundary
    # x_l, x_u, y_l, y_u = 0, 2, 0, 1
    x_l, x_u, y_l, y_u = 0, 2, 0, 2

    # dx, dy
    # n_x, n_y = 64, 64
    # n_x, n_y = 128, 128
    n_x, n_y = 256, 256
    # n_x, n_y = 512, 512
    dx = (x_u - x_l) / n_x  #/ 2
    dy = (y_u - y_l) / n_y  #/ 2

    x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
    y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)

    # n_padding = int((inputs0.shape[2] - n_x)/2)

    device = inputs.device

    x_ind_bc = inputs0[0, 0, :, :]

    # pad BC indicator
    # x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=0)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)

    inputs_np = inputs.detach().cpu().numpy()

    M0 = np.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_conv2d.shape[0]):
        for _x in np.arange(inputs_conv2d.shape[2]):
            for _y in np.arange(inputs_conv2d.shape[3]):
                M0_part = inputs_np[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                M0_part = M0_part.reshape(M0_part.shape[0], M0_part.shape[1], -1)
                M0_part = M0_part.reshape(M0_part.shape[0], -1)
                M0[ni:ni+1, :] = M0_part
                ni = ni + 1

    M_t = M0
    M_t = torch.tensor(M_t).to(device)


    # get u and v
    M_u_0, M_v_0 = inputs0[:, 1, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 2, :, :].reshape(inputs0.shape[0], -1)
    M_u_0, M_v_0 = M_u_0.flatten(), M_v_0.flatten()

    M_t_target = targets[:, 0, :, :].reshape(targets.shape[0], -1)
    M_t_target = M_t_target.flatten()

    # first iteration
    # reg = 1e9  # kernel 7
    # reg = 1e7  # kernel 9

    reg = 9.80552533e+09

    ### ------normal PDE------ ###  0
    _C = torch.eq(ind_bc, 0)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # numerical differentiation PDE (n-pde)
    UTx = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx)
    VTy = 0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy)
    t_xx, t_yy = (tE - 2*tC + tW)/(dx*dx), (tN - 2*tC + tS)/(dy*dy)

    # compute PDE matrix
    theta = 1/(500*4.3389)
    pde_normal_le = UTx + VTy - theta*(t_xx + t_yy)
    pde_normal_re = torch.zeros(pde_normal_le.shape[0], 1)
    pde_normal_re = pde_normal_re.to(device)


    ### ----------W_1---------- ###  1
    _C = torch.eq(ind_bc, 1)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1

    # compute PDE matrix
    pde_W_1_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                 0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                 theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_W_1_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)



    ### ----------S_1---------- ###  2
    _C = torch.eq(ind_bc, 2)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1

    # compute PDE matrix
    pde_S_1_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                 0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                 theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_S_1_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)



    ### ----------E_1---------- ###  3
    _C = torch.eq(ind_bc, 3)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1

    # compute PDE matrix
    pde_E_1_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                 0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                 theta*(-2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_E_1_re = -0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------N_1---------- ###  4
    _C = torch.eq(ind_bc, 4)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tN = 1

    # compute PDE matrix
    pde_N_1_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                 0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                 theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_N_1_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------WS_2---------- ###  5
    _C = torch.eq(ind_bc, 5)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tS = 1

    # compute PDE matrix
    pde_WS_2_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                  theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_WS_2_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) + 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------WE_2---------- ###  6
    _C = torch.eq(ind_bc, 6)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tE = 1

    # compute PDE matrix
    pde_WE_2_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                  theta*(-2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_WE_2_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------WN_2---------- ###  7
    _C = torch.eq(ind_bc, 7)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tN = 1

    # compute PDE matrix
    pde_WN_2_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                  0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                  theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_WN_2_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------SE_2---------- ###  8
    _C = torch.eq(ind_bc, 8)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tE = 1

    # compute PDE matrix
    pde_SE_2_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                  theta*(-2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_SE_2_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) - 0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------SN_2---------- ###  9
    _C = torch.eq(ind_bc, 9)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tN = 1

    # compute PDE matrix
    pde_SN_2_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                  0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                  theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_SN_2_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------EN_2---------- ###  10
    _C = torch.eq(ind_bc, 10)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1, tN = 1

    # compute PDE matrix
    pde_EN_2_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                  0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                  theta*(-2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_EN_2_re = -0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------WSE_3---------- ###  11
    _C = torch.eq(ind_bc, 11)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tS = 1, tE = 1

    # compute PDE matrix
    pde_WSE_3_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                   0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                   theta*(-2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_WSE_3_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) + 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) - 0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------SEN_3---------- ###  12
    _C = torch.eq(ind_bc, 12)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tE = 1, tN = 1

    # compute PDE matrix
    pde_SEN_3_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                   0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                   theta*(-2*tC + tW)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_SEN_3_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) - 0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------ENW_3---------- ###  13
    _C = torch.eq(ind_bc, 13)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1, tN = 1, tW = 1

    # compute PDE matrix
    pde_ENW_3_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                   0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                   theta*(-2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_ENW_3_re = -0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------NWS_3---------- ###  14
    _C = torch.eq(ind_bc, 14)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tN = 1, tW = 1, tS = 1

    # compute PDE matrix
    pde_NWS_3_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                   0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                   theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_NWS_3_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) + 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------left---------- ###  15
    _C = torch.eq(ind_bc, 15)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_le = tC
    pde_left_re = tC_target


    ### ----------left_E---------- ###  16
    _C = torch.eq(ind_bc, 16)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_E_le = tC
    pde_left_E_re = tC_target


    ### ----------left_N---------- ###  17
    _C = torch.eq(ind_bc, 17)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_N_le = tC
    pde_left_N_re = tC_target


    ### ----------left_NE---------- ###  18
    _C = torch.eq(ind_bc, 18)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_NE_le = tC
    pde_left_NE_re = tC_target


    ### ----------left_S---------- ###  19
    _C = torch.eq(ind_bc, 19)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_S_le = tC
    pde_left_S_re = tC_target


    ### ----------left_SE---------- ###  20
    _C = torch.eq(ind_bc, 20)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_SE_le = tC
    pde_left_SE_re = tC_target


    ### ----------right---------- ###  21
    _C = torch.eq(ind_bc, 21)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0
    tE = tC

    # compute PDE matrix
    pde_right_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                   0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                   theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_right_re = torch.zeros(pde_right_le.shape[0], 1).to(device)


    ### ----------right_W---------- ###  22
    _C = torch.eq(ind_bc, 22)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tW = 1
    tE = tC

    # compute PDE matrix
    pde_right_W_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                     0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                     theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_right_W_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------right_N---------- ###  23
    _C = torch.eq(ind_bc, 23)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tN = 1
    tE = tC

    # compute PDE matrix
    pde_right_N_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                     0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                     theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_right_N_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------right_NW---------- ###  24
    _C = torch.eq(ind_bc, 24)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tN = 1, tW = 1
    tE = tC

    # compute PDE matrix
    pde_right_NW_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                      0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                      theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_right_NW_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) 


    ### ----------right_S---------- ###  25
    _C = torch.eq(ind_bc, 25)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tS = 1
    tE = tC

    # compute PDE matrix
    pde_right_S_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                     0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                     theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_right_S_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------right_SW---------- ###  26
    _C = torch.eq(ind_bc, 26)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tS = 1, tW = 1
    tE = tC

    # compute PDE matrix
    pde_right_SW_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                      0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                      theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_right_SW_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) 


    # lamda_bc = 1e3  # kernel 7
    # lamda_bc = 1e5  # kernel 9
    # lamda_pde = 1e-6

    lamda_bc = 1.79061199e+04

    pde_M_le = torch.cat([pde_normal_le, pde_W_1_le, pde_S_1_le, \
                          pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
                          pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
                          pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
                          pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
                          lamda_bc*pde_left_le, lamda_bc*pde_left_E_le, lamda_bc*pde_left_N_le, \
                          lamda_bc*pde_left_NE_le, lamda_bc*pde_left_S_le, lamda_bc*pde_left_SE_le, \
                          lamda_bc*pde_right_le, lamda_bc*pde_right_W_le, lamda_bc*pde_right_N_le, \
                          lamda_bc*pde_right_NW_le, lamda_bc*pde_right_S_le, lamda_bc*pde_right_SW_le], axis=0)


    pde_M_re = torch.cat([pde_normal_re, pde_W_1_re, pde_S_1_re, \
                          pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
                          pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
                          pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
                          pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
                          lamda_bc*pde_left_re, lamda_bc*pde_left_E_re, lamda_bc*pde_left_N_re, \
                          lamda_bc*pde_left_NE_re, lamda_bc*pde_left_S_re, lamda_bc*pde_left_SE_re, \
                          lamda_bc*pde_right_re, lamda_bc*pde_right_W_re, lamda_bc*pde_right_N_re, \
                          lamda_bc*pde_right_NW_re, lamda_bc*pde_right_S_re, lamda_bc*pde_right_SW_re], axis=0)

    # pde_M_le = torch.cat([pde_normal_le, pde_W_1_le, pde_S_1_le, \
    #                       pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
    #                       pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
    #                       pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
    #                       pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
    #                       lamda_bc*pde_left_le, lamda_bc*pde_left_E_le, lamda_bc*pde_left_N_le, \
    #                       lamda_bc*pde_left_NE_le, lamda_bc*pde_left_S_le, lamda_bc*pde_left_SE_le, \
    #                       pde_right_le, pde_right_W_le, pde_right_N_le, \
    #                       pde_right_NW_le, pde_right_S_le, pde_right_SW_le], axis=0)


    # pde_M_re = torch.cat([pde_normal_re, pde_W_1_re, pde_S_1_re, \
    #                       pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
    #                       pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
    #                       pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
    #                       pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
    #                       lamda_bc*pde_left_re, lamda_bc*pde_left_E_re, lamda_bc*pde_left_N_re, \
    #                       lamda_bc*pde_left_NE_re, lamda_bc*pde_left_S_re, lamda_bc*pde_left_SE_re, \
    #                       pde_right_re, pde_right_W_re, pde_right_N_re, \
    #                       pde_right_NW_re, pde_right_S_re, pde_right_SW_re], axis=0)
    
    # pde_M_le = torch.cat([lamda_pde*pde_normal_le, pde_W_1_le, pde_S_1_le, \
    #                       pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
    #                       pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
    #                       pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
    #                       pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
    #                       pde_left_le, pde_left_E_le, pde_left_N_le, \
    #                       pde_left_NE_le, pde_left_S_le, pde_left_SE_le, \
    #                       pde_right_le, pde_right_W_le, pde_right_N_le, \
    #                       pde_right_NW_le, pde_right_S_le, pde_right_SW_le], axis=0)


    # pde_M_re = torch.cat([lamda_pde*pde_normal_re, pde_W_1_re, pde_S_1_re, \
    #                       pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
    #                       pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
    #                       pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
    #                       pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
    #                       pde_left_re, pde_left_E_re, pde_left_N_re, \
    #                       pde_left_NE_re, pde_left_S_re, pde_left_SE_re, \
    #                       pde_right_re, pde_right_W_re, pde_right_N_re, \
    #                       pde_right_NW_re, pde_right_S_re, pde_right_SW_re], axis=0)

    pde_M_re = pde_M_re.double()  # change to 64 bits

    # start = time.time()
    kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re
    # end = time.time()
    # runtime = end - start


    # pseudo inverse prediction results
    t = M_t @ kernel_parameters
    t = t.reshape(inputs0.shape[0], -1)
    t = t.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    t = torch.unsqueeze(t, axis=1)
    outputs_pseudo_inverse = t
    outputs_pseudo_inverse = outputs_pseudo_inverse.float()

    ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # convolution 2d prediction results
    outputs_conv2d = conv2d_head(inputs_conv2d).float()


    return outputs_pseudo_inverse, outputs_conv2d, ssr




def CNN_PINN_Pseudo_Inverse_2D_NS_Heat_DirichletBC1(inputs, inputs0, conv2d_head, targets):

    inputs_conv2d = inputs

    n_kernel_size = 9
    n_p = int(np.floor(n_kernel_size/2))
    # kernel size: 1*1, 3*3, 5*5, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

    # computational boundary
    # x_l, x_u, y_l, y_u = 0, 2, 0, 1
    x_l, x_u, y_l, y_u = 0, 2, 0, 2

    # dx, dy
    # n_x, n_y = 64, 64
    # n_x, n_y = 128, 128
    n_x, n_y = 256, 256
    # n_x, n_y = 512, 512
    dx = (x_u - x_l) / n_x  #/ 2
    dy = (y_u - y_l) / n_y  #/ 2

    x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
    y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)

    # n_padding = int((inputs0.shape[2] - n_x)/2)

    device = inputs.device

    x_ind_bc = inputs0[0, 0, :, :]

    # pad BC indicator
    # x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=0)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)

    inputs_np = inputs.detach().cpu().numpy()

    M0 = np.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_conv2d.shape[0]):
        for _x in np.arange(inputs_conv2d.shape[2]):
            for _y in np.arange(inputs_conv2d.shape[3]):
                M0_part = inputs_np[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                M0_part = M0_part.reshape(M0_part.shape[0], M0_part.shape[1], -1)
                M0_part = M0_part.reshape(M0_part.shape[0], -1)
                M0[ni:ni+1, :] = M0_part
                ni = ni + 1

    M_t = M0
    M_t = torch.tensor(M_t).to(device)


    # get u and v
    M_u_0, M_v_0 = inputs0[:, 1, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 2, :, :].reshape(inputs0.shape[0], -1)
    M_u_0, M_v_0 = M_u_0.flatten(), M_v_0.flatten()

    M_t_target = targets[:, 0, :, :].reshape(targets.shape[0], -1)
    M_t_target = M_t_target.flatten()

    # first iteration
    # reg = 1e0
    # reg = 1.83673469e+03  # kernel 5
    # reg = 6.32653065e+03
    reg = 4.03328307e-01

    ### ------normal PDE------ ###  0
    _C = torch.eq(ind_bc, 0)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # numerical differentiation PDE (n-pde)
    UTx = u0*((tE - tW)/(2*dx))
    VTy = v0*((tN - tS)/(2*dy))
    t_xx, t_yy = (tE - 2*tC + tW)/(dx*dx), (tN - 2*tC + tS)/(dy*dy)

    # compute PDE matrix
    theta = 1/(500*4.3389)
    pde_normal_le = UTx + VTy - theta*(t_xx + t_yy)
    pde_normal_re = torch.zeros(pde_normal_le.shape[0], 1)
    pde_normal_re = pde_normal_re.to(device)


    ### ----------W_1---------- ###  1
    _C = torch.eq(ind_bc, 1)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1

    # compute PDE matrix
    pde_W_1_le = u0*(tE/(2*dx)) + v0*((tN - tS)/(2*dy)) - theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_W_1_re = u0/(2*dx) + theta/(dx*dx)



    ### ----------S_1---------- ###  2
    _C = torch.eq(ind_bc, 2)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1

    # compute PDE matrix
    pde_S_1_le = u0*((tE - tW)/(2*dx)) + v0*(tN/(2*dy)) - theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_S_1_re = v0/(2*dy) + theta/(dy*dy)



    ### ----------E_1---------- ###  3
    _C = torch.eq(ind_bc, 3)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1

    # compute PDE matrix
    pde_E_1_le = u0*(-tW/(2*dx)) + v0*((tN - tS)/(2*dy)) - theta*(-2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_E_1_re = -u0/(2*dx) + theta/(dx*dx)


    ### ----------N_1---------- ###  4
    _C = torch.eq(ind_bc, 4)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tN = 1

    # compute PDE matrix
    pde_N_1_le = u0*((tE - tW)/(2*dx)) + v0*(-tS/(2*dy)) - theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_N_1_re = -v0/(2*dy) + theta/(dy*dy)


    ### ----------WS_2---------- ###  5
    _C = torch.eq(ind_bc, 5)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tS = 1

    # compute PDE matrix
    pde_WS_2_le = u0*(tE/(2*dx)) + v0*(tN/(2*dy)) - theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_WS_2_re = u0/(2*dx) + theta/(dx*dx) + v0/(2*dy) + theta/(dy*dy)


    ### ----------WE_2---------- ###  6
    _C = torch.eq(ind_bc, 6)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tE = 1

    # compute PDE matrix
    pde_WE_2_le = v0*((tN - tS)/(2*dy)) - theta*(-2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_WE_2_re = u0/(2*dx) + theta/(dx*dx) - u0/(2*dx) + theta/(dx*dx)


    ### ----------WN_2---------- ###  7
    _C = torch.eq(ind_bc, 7)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tN = 1

    # compute PDE matrix
    pde_WN_2_le = u0*(tE/(2*dx)) + v0*(-tS/(2*dy)) - theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_WN_2_re = u0/(2*dx) + theta/(dx*dx) - v0/(2*dy) + theta/(dy*dy)


    ### ----------SE_2---------- ###  8
    _C = torch.eq(ind_bc, 8)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tE = 1

    # compute PDE matrix
    pde_SE_2_le = u0*(-tW/(2*dx)) + v0*(tN/(2*dy)) - theta*(-2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_SE_2_re = v0/(2*dy) + theta/(dy*dy) - u0/(2*dx) + theta/(dx*dx)


    ### ----------SN_2---------- ###  9
    _C = torch.eq(ind_bc, 9)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tN = 1

    # compute PDE matrix
    pde_SN_2_le = u0*((tE - tW)/(2*dx)) - theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_SN_2_re = v0/(2*dy) + theta/(dy*dy) - v0/(2*dy) + theta/(dy*dy)


    ### ----------EN_2---------- ###  10
    _C = torch.eq(ind_bc, 10)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1, tN = 1

    # compute PDE matrix
    pde_EN_2_le = u0*(-tW/(2*dx)) + v0*(-tS/(2*dy)) - theta*(-2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_EN_2_re = -u0/(2*dx) + theta/(dx*dx) - v0/(2*dy) + theta/(dy*dy)


    ### ----------WSE_3---------- ###  11
    _C = torch.eq(ind_bc, 11)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tS = 1, tE = 1

    # compute PDE matrix
    pde_WSE_3_le = v0*(tN/(2*dy)) - theta*(-2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_WSE_3_re = u0/(2*dx) + theta/(dx*dx) + v0/(2*dy) + theta/(dy*dy) - u0/(2*dx) + theta/(dx*dx)


    ### ----------SEN_3---------- ###  12
    _C = torch.eq(ind_bc, 12)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tE = 1, tN = 1

    # compute PDE matrix
    pde_SEN_3_le = u0*(-tW/(2*dx)) - theta*(-2*tC + tW)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_SEN_3_re = v0/(2*dy) + theta/(dy*dy) - u0/(2*dx) + theta/(dx*dx) - v0/(2*dy) + theta/(dy*dy)


    ### ----------ENW_3---------- ###  13
    _C = torch.eq(ind_bc, 13)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1, tN = 1, tW = 1

    # compute PDE matrix
    pde_ENW_3_le = v0*(-tS/(2*dy)) - theta*(-2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_ENW_3_re = -u0/(2*dx) + theta/(dx*dx) - v0/(2*dy) + theta/(dy*dy) + u0/(2*dx) + theta/(dx*dx)


    ### ----------NWS_3---------- ###  14
    _C = torch.eq(ind_bc, 14)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tN = 1, tW = 1, tS = 1

    # compute PDE matrix
    pde_NWS_3_le = u0*(tE/(2*dx)) - theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_NWS_3_re = -v0/(2*dy) + theta/(dy*dy) + u0/(2*dx) + theta/(dx*dx) + v0/(2*dy) + theta/(dy*dy)


    ### ----------left---------- ###  15
    _C = torch.eq(ind_bc, 15)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_le = tC
    pde_left_re = tC_target


    ### ----------left_E---------- ###  16
    _C = torch.eq(ind_bc, 16)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_E_le = tC
    pde_left_E_re = tC_target


    ### ----------left_N---------- ###  17
    _C = torch.eq(ind_bc, 17)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_N_le = tC
    pde_left_N_re = tC_target


    ### ----------left_NE---------- ###  18
    _C = torch.eq(ind_bc, 18)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_NE_le = tC
    pde_left_NE_re = tC_target


    ### ----------left_S---------- ###  19
    _C = torch.eq(ind_bc, 19)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_S_le = tC
    pde_left_S_re = tC_target


    ### ----------left_SE---------- ###  20
    _C = torch.eq(ind_bc, 20)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_SE_le = tC
    pde_left_SE_re = tC_target


    ### ----------right---------- ###  21
    _C = torch.eq(ind_bc, 21)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0
    tE = tC

    # compute PDE matrix
    # pde_right_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
    #                0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
    #                theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    # pde_right_re = torch.zeros(pde_right_le.shape[0], 1).to(device)

    pde_right_le = tC
    pde_right_re = tC_target


    ### ----------right_W---------- ###  22
    _C = torch.eq(ind_bc, 22)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tW = 1
    tE = tC

    # compute PDE matrix
    # pde_right_W_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
    #                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
    #                  theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    # pde_right_W_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)

    pde_right_W_le = tC
    pde_right_W_re = tC_target


    ### ----------right_N---------- ###  23
    _C = torch.eq(ind_bc, 23)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tN = 1
    tE = tC

    # compute PDE matrix
    # pde_right_N_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
    #                  0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
    #                  theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    # pde_right_N_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)

    pde_right_N_le = tC
    pde_right_N_re = tC_target


    ### ----------right_NW---------- ###  24
    _C = torch.eq(ind_bc, 24)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tN = 1, tW = 1
    tE = tC

    # compute PDE matrix
    # pde_right_NW_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
    #                   0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
    #                   theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    # pde_right_NW_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)

    pde_right_NW_le = tC
    pde_right_NW_re = tC_target


    ### ----------right_S---------- ###  25
    _C = torch.eq(ind_bc, 25)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tS = 1
    tE = tC

    # compute PDE matrix
    # pde_right_S_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
    #                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
    #                  theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    # pde_right_S_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)

    pde_right_S_le = tC
    pde_right_S_re = tC_target


    ### ----------right_SW---------- ###  26
    _C = torch.eq(ind_bc, 26)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tS = 1, tW = 1
    tE = tC

    # compute PDE matrix
    # pde_right_SW_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
    #                   0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
    #                   theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    # pde_right_SW_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)

    pde_right_SW_le = tC
    pde_right_SW_re = tC_target


    # lamda_bc = 1e2
    # lamda_bc = 1.22457755e+04
    lamda_bc = 1.04900417e+01
    # lamda_pde = 1e-6

    pde_M_le = torch.cat([pde_normal_le, pde_W_1_le, pde_S_1_le, \
                          pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
                          pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
                          pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
                          pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
                          lamda_bc*pde_left_le, lamda_bc*pde_left_E_le, lamda_bc*pde_left_N_le, \
                          lamda_bc*pde_left_NE_le, lamda_bc*pde_left_S_le, lamda_bc*pde_left_SE_le, \
                          lamda_bc*pde_right_le, lamda_bc*pde_right_W_le, lamda_bc*pde_right_N_le, \
                          lamda_bc*pde_right_NW_le, lamda_bc*pde_right_S_le, lamda_bc*pde_right_SW_le], axis=0)


    pde_M_re = torch.cat([pde_normal_re, pde_W_1_re, pde_S_1_re, \
                          pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
                          pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
                          pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
                          pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
                          lamda_bc*pde_left_re, lamda_bc*pde_left_E_re, lamda_bc*pde_left_N_re, \
                          lamda_bc*pde_left_NE_re, lamda_bc*pde_left_S_re, lamda_bc*pde_left_SE_re, \
                          lamda_bc*pde_right_re, lamda_bc*pde_right_W_re, lamda_bc*pde_right_N_re, \
                          lamda_bc*pde_right_NW_re, lamda_bc*pde_right_S_re, lamda_bc*pde_right_SW_re], axis=0)
    
    # pde_M_le = torch.cat([lamda_pde*pde_normal_le, pde_W_1_le, pde_S_1_le, \
    #                       pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
    #                       pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
    #                       pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
    #                       pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
    #                       pde_left_le, pde_left_E_le, pde_left_N_le, \
    #                       pde_left_NE_le, pde_left_S_le, pde_left_SE_le, \
    #                       pde_right_le, pde_right_W_le, pde_right_N_le, \
    #                       pde_right_NW_le, pde_right_S_le, pde_right_SW_le], axis=0)


    # pde_M_re = torch.cat([lamda_pde*pde_normal_re, pde_W_1_re, pde_S_1_re, \
    #                       pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
    #                       pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
    #                       pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
    #                       pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
    #                       pde_left_re, pde_left_E_re, pde_left_N_re, \
    #                       pde_left_NE_re, pde_left_S_re, pde_left_SE_re, \
    #                       pde_right_re, pde_right_W_re, pde_right_N_re, \
    #                       pde_right_NW_re, pde_right_S_re, pde_right_SW_re], axis=0)

    pde_M_re = pde_M_re.double()  # change to 64 bits

    # start = time.time()
    kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re
    # end = time.time()
    # runtime = end - start


    # pseudo inverse prediction results
    t = M_t @ kernel_parameters
    t = t.reshape(inputs0.shape[0], -1)
    t = t.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    t = torch.unsqueeze(t, axis=1)
    outputs_pseudo_inverse = t
    outputs_pseudo_inverse = outputs_pseudo_inverse.float()

    ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # convolution 2d prediction results
    outputs_conv2d = conv2d_head(inputs_conv2d).float()


    return outputs_pseudo_inverse, outputs_conv2d, ssr



def CNN_PINN_Pseudo_Inverse_2D_NS_Heat_DirichletBC_reg(inputs, inputs0, conv2d_head, targets, lamda_bc, reg):

    inputs_conv2d = inputs

    n_kernel_size = 7
    n_p = int(np.floor(n_kernel_size/2))
    # kernel size: 1*1, 3*3, 5*5, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

    # computational boundary
    # x_l, x_u, y_l, y_u = 0, 2, 0, 1
    x_l, x_u, y_l, y_u = 0, 2, 0, 2

    # dx, dy
    # n_x, n_y = 64, 64
    # n_x, n_y = 128, 128
    n_x, n_y = 256, 256
    # n_x, n_y = 512, 512
    dx = (x_u - x_l) / n_x  #/ 2
    dy = (y_u - y_l) / n_y  #/ 2

    x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
    y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)

    # n_padding = int((inputs0.shape[2] - n_x)/2)

    device = inputs.device

    x_ind_bc = inputs0[0, 0, :, :]

    # pad BC indicator
    # x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=0)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)

    inputs_np = inputs.detach().cpu().numpy()

    M0 = np.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_conv2d.shape[0]):
        for _x in np.arange(inputs_conv2d.shape[2]):
            for _y in np.arange(inputs_conv2d.shape[3]):
                M0_part = inputs_np[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                M0_part = M0_part.reshape(M0_part.shape[0], M0_part.shape[1], -1)
                M0_part = M0_part.reshape(M0_part.shape[0], -1)
                M0[ni:ni+1, :] = M0_part
                ni = ni + 1

    M_t = M0
    M_t = torch.tensor(M_t).to(device)


    # get u and v
    M_u_0, M_v_0 = inputs0[:, 1, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 2, :, :].reshape(inputs0.shape[0], -1)
    M_u_0, M_v_0 = M_u_0.flatten(), M_v_0.flatten()

    M_t_target = targets[:, 0, :, :].reshape(targets.shape[0], -1)
    M_t_target = M_t_target.flatten()

    # first iteration
    # reg = 1e9
    # reg = 1.83673469e+03  # kernel 5

    ### ------normal PDE------ ###  0
    _C = torch.eq(ind_bc, 0)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # numerical differentiation PDE (n-pde)
    UTx = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx)
    VTy = 0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy)
    t_xx, t_yy = (tE - 2*tC + tW)/(dx*dx), (tN - 2*tC + tS)/(dy*dy)

    # compute PDE matrix
    theta = 1/(500*4.3389)
    pde_normal_le = UTx + VTy - theta*(t_xx + t_yy)
    pde_normal_re = torch.zeros(pde_normal_le.shape[0], 1)
    pde_normal_re = pde_normal_re.to(device)


    ### ----------W_1---------- ###  1
    _C = torch.eq(ind_bc, 1)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1

    # compute PDE matrix
    pde_W_1_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                 0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                 theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_W_1_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)



    ### ----------S_1---------- ###  2
    _C = torch.eq(ind_bc, 2)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1

    # compute PDE matrix
    pde_S_1_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                 0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                 theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_S_1_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)



    ### ----------E_1---------- ###  3
    _C = torch.eq(ind_bc, 3)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1

    # compute PDE matrix
    pde_E_1_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                 0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                 theta*(-2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_E_1_re = -0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------N_1---------- ###  4
    _C = torch.eq(ind_bc, 4)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tN = 1

    # compute PDE matrix
    pde_N_1_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                 0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                 theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_N_1_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------WS_2---------- ###  5
    _C = torch.eq(ind_bc, 5)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tS = 1

    # compute PDE matrix
    pde_WS_2_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                  theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_WS_2_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) + 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------WE_2---------- ###  6
    _C = torch.eq(ind_bc, 6)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tE = 1

    # compute PDE matrix
    pde_WE_2_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                  theta*(-2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_WE_2_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------WN_2---------- ###  7
    _C = torch.eq(ind_bc, 7)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tN = 1

    # compute PDE matrix
    pde_WN_2_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                  0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                  theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_WN_2_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------SE_2---------- ###  8
    _C = torch.eq(ind_bc, 8)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tE = 1

    # compute PDE matrix
    pde_SE_2_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                  theta*(-2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_SE_2_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) - 0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------SN_2---------- ###  9
    _C = torch.eq(ind_bc, 9)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tN = 1

    # compute PDE matrix
    pde_SN_2_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                  0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                  theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_SN_2_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------EN_2---------- ###  10
    _C = torch.eq(ind_bc, 10)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1, tN = 1

    # compute PDE matrix
    pde_EN_2_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                  0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                  theta*(-2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_EN_2_re = -0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------WSE_3---------- ###  11
    _C = torch.eq(ind_bc, 11)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tS = 1, tE = 1

    # compute PDE matrix
    pde_WSE_3_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                   0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                   theta*(-2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_WSE_3_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) + 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) - 0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------SEN_3---------- ###  12
    _C = torch.eq(ind_bc, 12)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tE = 1, tN = 1

    # compute PDE matrix
    pde_SEN_3_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                   0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                   theta*(-2*tC + tW)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_SEN_3_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) - 0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------ENW_3---------- ###  13
    _C = torch.eq(ind_bc, 13)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1, tN = 1, tW = 1

    # compute PDE matrix
    pde_ENW_3_le = 0.5*(u0 + torch.absolute(u0))*(-tC/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                   0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                   theta*(-2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_ENW_3_re = -0.5*(u0 + torch.absolute(u0))/dx + theta/(dx*dx) - 0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------NWS_3---------- ###  14
    _C = torch.eq(ind_bc, 14)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tN = 1, tW = 1, tS = 1

    # compute PDE matrix
    pde_NWS_3_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                   0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                   theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_NWS_3_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) + 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------left---------- ###  15
    _C = torch.eq(ind_bc, 15)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_le = tC
    pde_left_re = tC_target


    ### ----------left_E---------- ###  16
    _C = torch.eq(ind_bc, 16)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_E_le = tC
    pde_left_E_re = tC_target


    ### ----------left_N---------- ###  17
    _C = torch.eq(ind_bc, 17)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_N_le = tC
    pde_left_N_re = tC_target


    ### ----------left_NE---------- ###  18
    _C = torch.eq(ind_bc, 18)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_NE_le = tC
    pde_left_NE_re = tC_target


    ### ----------left_S---------- ###  19
    _C = torch.eq(ind_bc, 19)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_S_le = tC
    pde_left_S_re = tC_target


    ### ----------left_SE---------- ###  20
    _C = torch.eq(ind_bc, 20)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_SE_le = tC
    pde_left_SE_re = tC_target


    ### ----------right---------- ###  21
    _C = torch.eq(ind_bc, 21)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0
    tE = tC

    # compute PDE matrix
    pde_right_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                   0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                   theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_right_re = torch.zeros(pde_right_le.shape[0], 1).to(device)


    ### ----------right_W---------- ###  22
    _C = torch.eq(ind_bc, 22)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tW = 1
    tE = tC

    # compute PDE matrix
    pde_right_W_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                     0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                     theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_right_W_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)


    ### ----------right_N---------- ###  23
    _C = torch.eq(ind_bc, 23)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tN = 1
    tE = tC

    # compute PDE matrix
    pde_right_N_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                     0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                     theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_right_N_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------right_NW---------- ###  24
    _C = torch.eq(ind_bc, 24)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tN = 1, tW = 1
    tE = tC

    # compute PDE matrix
    pde_right_NW_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                      0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
                      theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_right_NW_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) 


    ### ----------right_S---------- ###  25
    _C = torch.eq(ind_bc, 25)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tS = 1
    tE = tC

    # compute PDE matrix
    pde_right_S_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
                     0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                     theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_right_S_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)


    ### ----------right_SW---------- ###  26
    _C = torch.eq(ind_bc, 26)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tS = 1, tW = 1
    tE = tC

    # compute PDE matrix
    pde_right_SW_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
                      0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
                      theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_right_SW_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx) 


    # lamda_bc = 1e3
    # lamda_pde = 1e-6

    pde_M_le = torch.cat([pde_normal_le, pde_W_1_le, pde_S_1_le, \
                          pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
                          pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
                          pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
                          pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
                          lamda_bc*pde_left_le, lamda_bc*pde_left_E_le, lamda_bc*pde_left_N_le, \
                          lamda_bc*pde_left_NE_le, lamda_bc*pde_left_S_le, lamda_bc*pde_left_SE_le, \
                          lamda_bc*pde_right_le, lamda_bc*pde_right_W_le, lamda_bc*pde_right_N_le, \
                          lamda_bc*pde_right_NW_le, lamda_bc*pde_right_S_le, lamda_bc*pde_right_SW_le], axis=0)


    pde_M_re = torch.cat([pde_normal_re, pde_W_1_re, pde_S_1_re, \
                          pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
                          pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
                          pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
                          pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
                          lamda_bc*pde_left_re, lamda_bc*pde_left_E_re, lamda_bc*pde_left_N_re, \
                          lamda_bc*pde_left_NE_re, lamda_bc*pde_left_S_re, lamda_bc*pde_left_SE_re, \
                          lamda_bc*pde_right_re, lamda_bc*pde_right_W_re, lamda_bc*pde_right_N_re, \
                          lamda_bc*pde_right_NW_re, lamda_bc*pde_right_S_re, lamda_bc*pde_right_SW_re], axis=0)

    # pde_M_le = torch.cat([pde_normal_le, pde_W_1_le, pde_S_1_le, \
    #                       pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
    #                       pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
    #                       pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
    #                       pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
    #                       lamda_bc*pde_left_le, lamda_bc*pde_left_E_le, lamda_bc*pde_left_N_le, \
    #                       lamda_bc*pde_left_NE_le, lamda_bc*pde_left_S_le, lamda_bc*pde_left_SE_le, \
    #                       pde_right_le, pde_right_W_le, pde_right_N_le, \
    #                       pde_right_NW_le, pde_right_S_le, pde_right_SW_le], axis=0)


    # pde_M_re = torch.cat([pde_normal_re, pde_W_1_re, pde_S_1_re, \
    #                       pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
    #                       pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
    #                       pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
    #                       pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
    #                       lamda_bc*pde_left_re, lamda_bc*pde_left_E_re, lamda_bc*pde_left_N_re, \
    #                       lamda_bc*pde_left_NE_re, lamda_bc*pde_left_S_re, lamda_bc*pde_left_SE_re, \
    #                       pde_right_re, pde_right_W_re, pde_right_N_re, \
    #                       pde_right_NW_re, pde_right_S_re, pde_right_SW_re], axis=0)
    
    # pde_M_le = torch.cat([lamda_pde*pde_normal_le, pde_W_1_le, pde_S_1_le, \
    #                       pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
    #                       pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
    #                       pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
    #                       pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
    #                       pde_left_le, pde_left_E_le, pde_left_N_le, \
    #                       pde_left_NE_le, pde_left_S_le, pde_left_SE_le, \
    #                       pde_right_le, pde_right_W_le, pde_right_N_le, \
    #                       pde_right_NW_le, pde_right_S_le, pde_right_SW_le], axis=0)


    # pde_M_re = torch.cat([lamda_pde*pde_normal_re, pde_W_1_re, pde_S_1_re, \
    #                       pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
    #                       pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
    #                       pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
    #                       pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
    #                       pde_left_re, pde_left_E_re, pde_left_N_re, \
    #                       pde_left_NE_re, pde_left_S_re, pde_left_SE_re, \
    #                       pde_right_re, pde_right_W_re, pde_right_N_re, \
    #                       pde_right_NW_re, pde_right_S_re, pde_right_SW_re], axis=0)

    pde_M_re = pde_M_re.double()  # change to 64 bits

    # start = time.time()
    kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re
    # end = time.time()
    # runtime = end - start


    # pseudo inverse prediction results
    t = M_t @ kernel_parameters
    t = t.reshape(inputs0.shape[0], -1)
    t = t.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    t = torch.unsqueeze(t, axis=1)
    outputs_pseudo_inverse = t
    outputs_pseudo_inverse = outputs_pseudo_inverse.float()

    ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # convolution 2d prediction results
    outputs_conv2d = conv2d_head(inputs_conv2d).float()


    return outputs_pseudo_inverse, outputs_conv2d, ssr



def CNN_PINN_Pseudo_Inverse_2D_NS_Heat_DirichletBC1_reg(inputs, inputs0, conv2d_head, targets, lamda_bc, reg):

    inputs_conv2d = inputs

    n_kernel_size = 9
    n_p = int(np.floor(n_kernel_size/2))
    # kernel size: 1*1, 3*3, 5*5, stride: 1
    inputs = nn.functional.pad(inputs[:, :, :, :], (n_p, 0, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, n_p, 0, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, n_p, 0, 0, 0, 0, 0))
    inputs = nn.functional.pad(inputs[:, :, :, :], (0, 0, 0, n_p, 0, 0, 0, 0))

    # computational boundary
    # x_l, x_u, y_l, y_u = 0, 2, 0, 1
    x_l, x_u, y_l, y_u = 0, 2, 0, 2

    # dx, dy
    # n_x, n_y = 64, 64
    # n_x, n_y = 128, 128
    n_x, n_y = 256, 256
    # n_x, n_y = 512, 512
    dx = (x_u - x_l) / n_x  #/ 2
    dy = (y_u - y_l) / n_y  #/ 2

    x = np.linspace(x_l + dx/2, x_u - dx/2, n_x)
    y = np.linspace(y_l + dy/2, y_u - dy/2, n_y)

    # n_padding = int((inputs0.shape[2] - n_x)/2)

    device = inputs.device

    x_ind_bc = inputs0[0, 0, :, :]

    # pad BC indicator
    # x_ind_bc = np.pad(x_ind_bc, (n_padding, n_padding), 'constant', constant_values=10)
    x_ind_bc = torch.unsqueeze(x_ind_bc, axis=0)
    ind_bc = x_ind_bc.repeat(inputs.shape[0], 1, 1)

    inputs_np = inputs.detach().cpu().numpy()

    M0 = np.zeros((inputs_conv2d.shape[0]*inputs_conv2d.shape[2]*inputs_conv2d.shape[3], inputs_conv2d.shape[1]*n_kernel_size*n_kernel_size))  # matrix to get u (n_channels*n_kernel_size*n_kernel_size*2)
    ni = 0
    for _t in np.arange(inputs_conv2d.shape[0]):
        for _x in np.arange(inputs_conv2d.shape[2]):
            for _y in np.arange(inputs_conv2d.shape[3]):
                M0_part = inputs_np[_t:_t+1, :, _x:_x+n_kernel_size, _y:_y+n_kernel_size]
                M0_part = M0_part.reshape(M0_part.shape[0], M0_part.shape[1], -1)
                M0_part = M0_part.reshape(M0_part.shape[0], -1)
                M0[ni:ni+1, :] = M0_part
                ni = ni + 1

    M_t = M0
    M_t = torch.tensor(M_t).to(device)


    # get u and v
    M_u_0, M_v_0 = inputs0[:, 1, :, :].reshape(inputs0.shape[0], -1), inputs0[:, 2, :, :].reshape(inputs0.shape[0], -1)
    M_u_0, M_v_0 = M_u_0.flatten(), M_v_0.flatten()

    M_t_target = targets[:, 0, :, :].reshape(targets.shape[0], -1)
    M_t_target = M_t_target.flatten()

    # first iteration
    # reg = 1e0
    # reg = 1.83673469e+03  # kernel 5

    ### ------normal PDE------ ###  0
    _C = torch.eq(ind_bc, 0)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # numerical differentiation PDE (n-pde)
    UTx = u0*((tE - tW)/(2*dx))
    VTy = v0*((tN - tS)/(2*dy))
    t_xx, t_yy = (tE - 2*tC + tW)/(dx*dx), (tN - 2*tC + tS)/(dy*dy)

    # compute PDE matrix
    theta = 1/(500*4.3389)
    pde_normal_le = UTx + VTy - theta*(t_xx + t_yy)
    pde_normal_re = torch.zeros(pde_normal_le.shape[0], 1)
    pde_normal_re = pde_normal_re.to(device)


    ### ----------W_1---------- ###  1
    _C = torch.eq(ind_bc, 1)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1

    # compute PDE matrix
    pde_W_1_le = u0*(tE/(2*dx)) + v0*((tN - tS)/(2*dy)) - theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_W_1_re = u0/(2*dx) + theta/(dx*dx)



    ### ----------S_1---------- ###  2
    _C = torch.eq(ind_bc, 2)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1

    # compute PDE matrix
    pde_S_1_le = u0*((tE - tW)/(2*dx)) + v0*(tN/(2*dy)) - theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_S_1_re = v0/(2*dy) + theta/(dy*dy)



    ### ----------E_1---------- ###  3
    _C = torch.eq(ind_bc, 3)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1

    # compute PDE matrix
    pde_E_1_le = u0*(-tW/(2*dx)) + v0*((tN - tS)/(2*dy)) - theta*(-2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_E_1_re = -u0/(2*dx) + theta/(dx*dx)


    ### ----------N_1---------- ###  4
    _C = torch.eq(ind_bc, 4)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tN = 1

    # compute PDE matrix
    pde_N_1_le = u0*((tE - tW)/(2*dx)) + v0*(-tS/(2*dy)) - theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_N_1_re = -v0/(2*dy) + theta/(dy*dy)


    ### ----------WS_2---------- ###  5
    _C = torch.eq(ind_bc, 5)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tS = 1

    # compute PDE matrix
    pde_WS_2_le = u0*(tE/(2*dx)) + v0*(tN/(2*dy)) - theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_WS_2_re = u0/(2*dx) + theta/(dx*dx) + v0/(2*dy) + theta/(dy*dy)


    ### ----------WE_2---------- ###  6
    _C = torch.eq(ind_bc, 6)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tE = 1

    # compute PDE matrix
    pde_WE_2_le = v0*((tN - tS)/(2*dy)) - theta*(-2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    pde_WE_2_re = u0/(2*dx) + theta/(dx*dx) - u0/(2*dx) + theta/(dx*dx)


    ### ----------WN_2---------- ###  7
    _C = torch.eq(ind_bc, 7)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tN = 1

    # compute PDE matrix
    pde_WN_2_le = u0*(tE/(2*dx)) + v0*(-tS/(2*dy)) - theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_WN_2_re = u0/(2*dx) + theta/(dx*dx) - v0/(2*dy) + theta/(dy*dy)


    ### ----------SE_2---------- ###  8
    _C = torch.eq(ind_bc, 8)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tE = 1

    # compute PDE matrix
    pde_SE_2_le = u0*(-tW/(2*dx)) + v0*(tN/(2*dy)) - theta*(-2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_SE_2_re = v0/(2*dy) + theta/(dy*dy) - u0/(2*dx) + theta/(dx*dx)


    ### ----------SN_2---------- ###  9
    _C = torch.eq(ind_bc, 9)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tN = 1

    # compute PDE matrix
    pde_SN_2_le = u0*((tE - tW)/(2*dx)) - theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_SN_2_re = v0/(2*dy) + theta/(dy*dy) - v0/(2*dy) + theta/(dy*dy)


    ### ----------EN_2---------- ###  10
    _C = torch.eq(ind_bc, 10)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1, tN = 1

    # compute PDE matrix
    pde_EN_2_le = u0*(-tW/(2*dx)) + v0*(-tS/(2*dy)) - theta*(-2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_EN_2_re = -u0/(2*dx) + theta/(dx*dx) - v0/(2*dy) + theta/(dy*dy)


    ### ----------WSE_3---------- ###  11
    _C = torch.eq(ind_bc, 11)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tW = 1, tS = 1, tE = 1

    # compute PDE matrix
    pde_WSE_3_le = v0*(tN/(2*dy)) - theta*(-2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    pde_WSE_3_re = u0/(2*dx) + theta/(dx*dx) + v0/(2*dy) + theta/(dy*dy) - u0/(2*dx) + theta/(dx*dx)


    ### ----------SEN_3---------- ###  12
    _C = torch.eq(ind_bc, 12)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tS = 1, tE = 1, tN = 1

    # compute PDE matrix
    pde_SEN_3_le = u0*(-tW/(2*dx)) - theta*(-2*tC + tW)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_SEN_3_re = v0/(2*dy) + theta/(dy*dy) - u0/(2*dx) + theta/(dx*dx) - v0/(2*dy) + theta/(dy*dy)


    ### ----------ENW_3---------- ###  13
    _C = torch.eq(ind_bc, 13)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tE = 1, tN = 1, tW = 1

    # compute PDE matrix
    pde_ENW_3_le = v0*(-tS/(2*dy)) - theta*(-2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    pde_ENW_3_re = -u0/(2*dx) + theta/(dx*dx) - v0/(2*dy) + theta/(dy*dy) + u0/(2*dx) + theta/(dx*dx)


    ### ----------NWS_3---------- ###  14
    _C = torch.eq(ind_bc, 14)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    # tN = 1, tW = 1, tS = 1

    # compute PDE matrix
    pde_NWS_3_le = u0*(tE/(2*dx)) - theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC)/(dy*dy)
    pde_NWS_3_re = -v0/(2*dy) + theta/(dy*dy) + u0/(2*dx) + theta/(dx*dx) + v0/(2*dy) + theta/(dy*dy)


    ### ----------left---------- ###  15
    _C = torch.eq(ind_bc, 15)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_le = tC
    pde_left_re = tC_target


    ### ----------left_E---------- ###  16
    _C = torch.eq(ind_bc, 16)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_E_le = tC
    pde_left_E_re = tC_target


    ### ----------left_N---------- ###  17
    _C = torch.eq(ind_bc, 17)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_N_le = tC
    pde_left_N_re = tC_target


    ### ----------left_NE---------- ###  18
    _C = torch.eq(ind_bc, 18)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_NE_le = tC
    pde_left_NE_re = tC_target


    ### ----------left_S---------- ###  19
    _C = torch.eq(ind_bc, 19)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_S_le = tC
    pde_left_S_re = tC_target


    ### ----------left_SE---------- ###  20
    _C = torch.eq(ind_bc, 20)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # compute PDE matrix
    pde_left_SE_le = tC
    pde_left_SE_re = tC_target


    ### ----------right---------- ###  21
    _C = torch.eq(ind_bc, 21)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0
    tE = tC

    # compute PDE matrix
    # pde_right_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
    #                0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
    #                theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    # pde_right_re = torch.zeros(pde_right_le.shape[0], 1).to(device)

    pde_right_le = tC
    pde_right_re = tC_target


    ### ----------right_W---------- ###  22
    _C = torch.eq(ind_bc, 22)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tW = 1
    tE = tC

    # compute PDE matrix
    # pde_right_W_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
    #                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
    #                  theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC + tS)/(dy*dy)
    # pde_right_W_re = 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)

    pde_right_W_le = tC
    pde_right_W_re = tC_target


    ### ----------right_N---------- ###  23
    _C = torch.eq(ind_bc, 23)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tN = 1
    tE = tC

    # compute PDE matrix
    # pde_right_N_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
    #                  0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
    #                  theta*(tE - 2*tC + tW)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    # pde_right_N_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy)

    pde_right_N_le = tC
    pde_right_N_re = tC_target


    ### ----------right_NW---------- ###  24
    _C = torch.eq(ind_bc, 24)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    # tN = M_t[_N, :]
    tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tN = 1, tW = 1
    tE = tC

    # compute PDE matrix
    # pde_right_NW_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
    #                   0.5*(v0 + torch.absolute(v0))*(-tC/dy) + 0.5*(v0 - torch.absolute(v0))*((tC - tS)/dy) - \
    #                   theta*(tE - 2*tC)/(dx*dx) - theta*(-2*tC + tS)/(dy*dy)
    # pde_right_NW_re = -0.5*(v0 + torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)

    pde_right_NW_le = tC
    pde_right_NW_re = tC_target


    ### ----------right_S---------- ###  25
    _C = torch.eq(ind_bc, 25)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tS = 1
    tE = tC

    # compute PDE matrix
    # pde_right_S_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*((tC - tW)/dx) + \
    #                  0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
    #                  theta*(tE - 2*tC + tW)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    # pde_right_S_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy)

    pde_right_S_le = tC
    pde_right_S_re = tC_target


    ### ----------right_SW---------- ###  26
    _C = torch.eq(ind_bc, 26)
    _E = nn.functional.pad(_C[:, :, :-1], (1, 0, 0, 0, 0, 0))
    _W = nn.functional.pad(_C[:, :, 1:], (0, 1, 0, 0, 0, 0))
    _N = nn.functional.pad(_C[:, :-1, :], (0, 0, 1, 0, 0, 0))
    _S = nn.functional.pad(_C[:, 1:, :], (0, 0, 0, 1, 0, 0))

    # flatten the index
    _C, _E, _W, _N, _S = _C.reshape(_C.shape[0], -1), _E.reshape(_E.shape[0], -1), _W.reshape(_W.shape[0], -1), _N.reshape(_N.shape[0], -1), _S.reshape(_S.shape[0], -1)
    _C, _E, _W, _N, _S = _C.flatten(), _E.flatten(), _W.flatten(), _N.flatten(), _S.flatten()

    tC = M_t[_C, :]
    # tE = M_t[_E, :]
    # tW = M_t[_W, :]
    tN = M_t[_N, :]
    # tS = M_t[_S, :]

    u0, v0 = M_u_0[_C].reshape(-1, 1), M_v_0[_C].reshape(-1, 1)

    tC_target = M_t_target[_C].reshape(-1, 1)

    # (tE - tC)/2 = 0, tS = 1, tW = 1
    tE = tC

    # compute PDE matrix
    # pde_right_SW_le = 0.5*(u0 + torch.absolute(u0))*((tE - tC)/dx) + 0.5*(u0 - torch.absolute(u0))*(tC/dx) + \
    #                   0.5*(v0 + torch.absolute(v0))*((tN - tC)/dy) + 0.5*(v0 - torch.absolute(v0))*(tC/dy) - \
    #                   theta*(tE - 2*tC)/(dx*dx) - theta*(tN - 2*tC)/(dy*dy)
    # pde_right_SW_re = 0.5*(v0 - torch.absolute(v0))/dy + theta/(dy*dy) + 0.5*(u0 - torch.absolute(u0))/dx + theta/(dx*dx)

    pde_right_SW_le = tC
    pde_right_SW_re = tC_target


    # lamda_bc = 1e2
    # lamda_pde = 1e-6

    pde_M_le = torch.cat([pde_normal_le, pde_W_1_le, pde_S_1_le, \
                          pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
                          pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
                          pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
                          pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
                          lamda_bc*pde_left_le, lamda_bc*pde_left_E_le, lamda_bc*pde_left_N_le, \
                          lamda_bc*pde_left_NE_le, lamda_bc*pde_left_S_le, lamda_bc*pde_left_SE_le, \
                          lamda_bc*pde_right_le, lamda_bc*pde_right_W_le, lamda_bc*pde_right_N_le, \
                          lamda_bc*pde_right_NW_le, lamda_bc*pde_right_S_le, lamda_bc*pde_right_SW_le], axis=0)


    pde_M_re = torch.cat([pde_normal_re, pde_W_1_re, pde_S_1_re, \
                          pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
                          pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
                          pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
                          pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
                          lamda_bc*pde_left_re, lamda_bc*pde_left_E_re, lamda_bc*pde_left_N_re, \
                          lamda_bc*pde_left_NE_re, lamda_bc*pde_left_S_re, lamda_bc*pde_left_SE_re, \
                          lamda_bc*pde_right_re, lamda_bc*pde_right_W_re, lamda_bc*pde_right_N_re, \
                          lamda_bc*pde_right_NW_re, lamda_bc*pde_right_S_re, lamda_bc*pde_right_SW_re], axis=0)
    
    # pde_M_le = torch.cat([lamda_pde*pde_normal_le, pde_W_1_le, pde_S_1_le, \
    #                       pde_E_1_le, pde_N_1_le, pde_WS_2_le, \
    #                       pde_WE_2_le, pde_WN_2_le, pde_SE_2_le, \
    #                       pde_SN_2_le, pde_EN_2_le, pde_WSE_3_le, \
    #                       pde_SEN_3_le, pde_ENW_3_le, pde_NWS_3_le, \
    #                       pde_left_le, pde_left_E_le, pde_left_N_le, \
    #                       pde_left_NE_le, pde_left_S_le, pde_left_SE_le, \
    #                       pde_right_le, pde_right_W_le, pde_right_N_le, \
    #                       pde_right_NW_le, pde_right_S_le, pde_right_SW_le], axis=0)


    # pde_M_re = torch.cat([lamda_pde*pde_normal_re, pde_W_1_re, pde_S_1_re, \
    #                       pde_E_1_re, pde_N_1_re, pde_WS_2_re, \
    #                       pde_WE_2_re, pde_WN_2_re, pde_SE_2_re, \
    #                       pde_SN_2_re, pde_EN_2_re, pde_WSE_3_re, \
    #                       pde_SEN_3_re, pde_ENW_3_re, pde_NWS_3_re, \
    #                       pde_left_re, pde_left_E_re, pde_left_N_re, \
    #                       pde_left_NE_re, pde_left_S_re, pde_left_SE_re, \
    #                       pde_right_re, pde_right_W_re, pde_right_N_re, \
    #                       pde_right_NW_re, pde_right_S_re, pde_right_SW_re], axis=0)

    pde_M_re = pde_M_re.double()  # change to 64 bits

    # start = time.time()
    kernel_parameters = torch.inverse(reg*torch.eye(pde_M_le.shape[1]).to(device) + (pde_M_le.T @ pde_M_le)) @ pde_M_le.T @ pde_M_re
    # end = time.time()
    # runtime = end - start


    # pseudo inverse prediction results
    t = M_t @ kernel_parameters
    t = t.reshape(inputs0.shape[0], -1)
    t = t.reshape(inputs0.shape[0], inputs0.shape[2], inputs0.shape[3])
    t = torch.unsqueeze(t, axis=1)
    outputs_pseudo_inverse = t
    outputs_pseudo_inverse = outputs_pseudo_inverse.float()

    ssr = torch.sum((pde_M_re - pde_M_le @ kernel_parameters)**2)

    # convolution 2d prediction results
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
