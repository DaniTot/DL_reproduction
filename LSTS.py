import torch
import torch.nn.functional as F
import numpy as np
import sys
import gc
import itertools

from collections import Counter
import tracemalloc
from MemProfile import display_top

class Com_LSTS():
    def __init__(self, In_Shape, N=5):
        # self.weight_SRFU = torch.rand(In_Shape)
        self.In_Shape = In_Shape
        self.N = N

        self.feature_quality = None
        self.F_mem = None

        # Location is of shape (N, 2)
        # self.Location = np.concatenate((np.random.uniform(0, In_Shape[2], size=(18, 32, self.N, 1)),
        #                                np.random.uniform(0, In_Shape[3], size=(18, 32, self.N, 1))),
        #                                axis=3)
        self.Location = self.loc_init()
        print("loc", self.Location.shape)
        print("shape", In_Shape)
        ## for testing purpose

        self.p_0 = None

        self.F_0_p = None
        self.s_p = torch.zeros((N, 18, 32))
        self.S_p = torch.zeros((N, 18, 32))

        self.F_0 = None
        self.F_1 = None

        self.F_0_embedded = None
        self.F_1_embedded = None

        # Initialize the low2high convolution network
        # code says: (1x1x256), (3x3x256), (3x3x1024)
        # paper says: (3x3x256), (3x3x512), (3x3x1024)
        self.low2high = torch.nn.Sequential(torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,
                                                            stride=1, padding=0, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                                            stride=1, padding=1, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3,
                                                            stride=1, padding=1, bias=True))

        # code says: (3x3x256), (3x3x16), (3x3x1)
        # paper says: (3x3x256), (1x1x16), (1x1x1)
        self.quality = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3),
            torch.nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, padding=1),
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding=1)
        )

        tracemalloc.start()

        print('Initialization done!')

    def loc_init(self, init_offset=2):
        loc = np.zeros((18, 32, self.N, 2))
        for i in range(18):
            for j in range(32):
                for n in range(self.N):
                    loc[i, j, n, 0] = i + np.clip(np.random.uniform(-1 * init_offset, init_offset), 0, 18-1)
                    loc[i, j, n, 1] = j + np.clip(np.random.uniform(-1 * init_offset, init_offset), 0, 32-1)
        return loc

    def get_input(self, feat1, feat2):
        self.F_0 = feat1
        self.F_1 = feat2

        self.F_0_embedded = self.f(self.F_0)
        self.F_1_embedded = self.f(self.F_1)

        self.F_0_p = None
        self.S_p = torch.zeros((self.N, 18, 32))
        self.feature_quality = None

        self.F_pred = torch.zeros(self.F_0.shape)
        self.grad = torch.zeros((18, 32, self.N, 2))

        if self.F_0_embedded.shape[2] == 18 and self.F_0_embedded.shape[3] == 32 and self.F_1_embedded.shape[2] == 18 and self.F_1_embedded.shape[3] == 32:
            return 1
        else:
            return 0

    def quality_network(self, feat1, feat2, key):
        feature_concat = torch.cat([feat1, feat2], dim=0)
        quality_res = self.quality(feature_concat)
        assert quality_res.shape[0] / 2 == 1, quality_res.shape
        quality_weights = torch.split(quality_res, 1, dim=0)

        #TODO: make sure dim=1 is correct. MXNet default is axis=-1.
        quality_weights = F.softmax(torch.cat([quality_weights[0], quality_weights[1]], dim=1))
        assert quality_weights.shape[1]/2 == 1, quality_weights.shape
        quality_weights_splice = torch.split(quality_weights, 1, dim=1)

        quality_weight1 = torch.tile(quality_weights_splice[0], (1, 1024, 1, 1))
        quality_weight2 = torch.tile(quality_weights_splice[1], (1, 1024, 1, 1))

        feat_result = torch.add(quality_weight1*feat1, quality_weight2*feat2)

        if key is True:
            self.F_mem = feat_result

        return feat_result

    def low2high_transform(self, nonkey):
        high_feat_current = self.low2high(nonkey)
        return high_feat_current

    def bilinear_sampling(self, input, offset):
        grid = offset
        return torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)

    def f(self, F_0):
        """Embedding function f"""
        embedding = torch.nn.Conv2d(F_0.shape[1], 256, kernel_size=(1, 1), stride=1, padding=0)
        fF_0 = embedding(F_0)
        return fF_0

    def BiLinKernel(self, q, p):
        x = torch.max(torch.as_tensor([0, 1 - torch.abs(q[0] - p[0])]))
        y = torch.max(torch.as_tensor([0, 1 - torch.abs(q[1] - p[1])]))
        c = x * y
        return c

    def BiLinKernel_vec(self, q, p):
        a = 1. - torch.abs(q[:, 0] - p[0])
        b = 1. - torch.abs(q[:, 1] - p[1])
        zero = torch.zeros(a.shape)

        x = torch.max(zero, a)
        y = torch.max(zero, b)
        c = x * y

        return c

    def Sum_q(self):
        """
        Populates/updates tensor matrices p_n and F_0_pn
        """
        self.F_0_p = None
        for n in range(self.N):
            sum = None
            for i in range(0, 18):
                for j in range(0, 32):
                    G = self.BiLinKernel_vec(torch.as_tensor(self.Location[self.p_0[0], self.p_0[1], :, :]).float(),
                                             torch.as_tensor((i, j)).float())
                    product = G[n] * self.F_0_embedded[:, :, i, j]
                    if sum is None:
                        sum = product
                    else:
                        sum += product
            if self.F_0_p is None:
                self.F_0_p = sum
            else:
                self.F_0_p = torch.cat((self.F_0_p, sum), 0)

    def dot_product(self):
        s_p = []
        for n in range(self.N):
            dot_product = torch.dot(self.F_0_p[n], self.F_1_embedded[0, :, self.p_0[0], self.p_0[1]])
            s_p.append(dot_product)

        # self.s_p = torch.tensor(s_p)
        return torch.tensor(s_p)

    def Normalize(self, s_p, absolutes=True):
        if absolutes:
            self.S_p[:, self.p_0[0], self.p_0[1]] = s_p / torch.sum(torch.abs(s_p))
        else:
            self.S_p[:, self.p_0[0], self.p_0[1]] = s_p/torch.sum(s_p)
        return


    def WeightGenerate(self):
        # print("Weigh Generate", self.p_0)
        self.Sum_q()
        s_p = self.dot_product()
        self.Normalize(s_p)
        return

    def DoImage(self, weight=True, aggregate=True, gradients=True, update=False):
        for pi in range(18):
            for pj in range(32):
                self.p_0 = [pi, pj]
                print(self.p_0)

                if weight:
                    self.WeightGenerate()

                if aggregate:
                    self.Aggregate(pi, pj)

                if gradients:
                    self.Gradients(pi, pj)
                    gc.collect()

                if update:
                    self.update()

        print("DONE!")
        return

    def Aggregate(self, pi, pj):
        # print("Aggregate", pi, pj)
        n_sum = 0
        for n in range(self.N):
            # Bilinear sampling of F_0 at p_n
            bil_sum = 0
            for qi in range(0, self.F_0.shape[2]):
                for qj in range(0, self.F_0.shape[3]):
                    G = self.BiLinKernel_vec(torch.as_tensor(self.Location[pi, pj, :, :]).float(),
                                             torch.as_tensor((qi, qj)).float())
                    product = G[n] * self.F_0[:, :, qi, qj]
                    bil_sum += product
            n_sum += self.S_p[n, pi, pj] * bil_sum
        self.F_pred[0, :, pi, pj] = n_sum
        return

    # TODO: backpropagation and update
    # TODO: We want high similarity, so S_p = 1 => p_n = p_n + d(S_p)/d(p_n) * lr

    # Figure 3 illustrates how to obtain the offsets. Firstly, RoIpooling (Eq. (5)) generates the pooled feature maps.
    # Fromthe maps, a fc layer generates thenormalizedoffsets∆̂pij,which are then transformed to the offsets∆pijin
    # Eq.(6) by element-wise product with the RoI’s width and height,as∆pij=γ·∆̂pij◦(w,h).
    # Hereγis a pre-defined scalarto modulate the magnitude of the offsets.  It is empiricallyset toγ=  0.1.
    # The offset normalization is necessary tomake the offset learning invariant to RoI size.
    # The fc layeris learned by back-propagation, as detailed in appendix A.
    def G_grad(self, p, q):
        if type(p) is torch.Tensor:
            p = p.detach().numpy()
        if type(q) is torch.Tensor:
            q = q.detach().numpy()
        if np.abs(p[1] - q[1]) < 1 and np.abs(p[0] - q[0]) < 1:
            if p[0] == q[0]:
                grad_x = 0
            else:
                grad_x = -(p[0] - q[0]) / np.abs(p[0] - q[0]) * (1 - np.abs(p[1] - q[1]))
            if p[1] == q[1]:
                grad_y = 0
            else:
                grad_y = -(p[1] - q[1]) / np.abs(p[1] - q[1]) * (1 - np.abs(p[0] - q[0]))

            assert not (np.isnan(grad_x) or np.isnan(grad_y))
        else:
            grad_x = 0
            grad_y = 0
        return [grad_x, grad_y]

    def Gradients(self, pi, pj):
        for n in range(self.N):
            # Grad x
            self.grad[pi, pj, n, 0] = 0
            # Grad y
            self.grad[pi, pj, n, 1] = 0
            for qi, qj in itertools.product(range(self.F_0_embedded.shape[2]), range(self.F_0_embedded.shape[3])):
                # print(type(self.Location[self.p_0[0], self.p_0[1], n, :]), type(np.array([qi, qj])))
                G_grad = self.G_grad(self.Location[self.p_0[0], self.p_0[1], n, :], np.array([qi, qj]))

                self.grad[pi, pj, n, 0] += torch.dot((G_grad[0] * self.F_0_embedded[0, :, qi, qj]),
                                                     self.F_1_embedded[0, :, self.p_0[0], self.p_0[1]])
                self.grad[pi, pj, n, 1] += torch.dot((G_grad[1] * self.F_0_embedded[0, :, qi, qj]),
                                                     self.F_1_embedded[0, :, self.p_0[0], self.p_0[1]])
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
        print()
        return

    def update(self, lr=0.1):
        if type(self.Location) is np.ndarray:
            self.Location = torch.from_numpy(self.Location) + self.grad*lr
        elif type(self.Location) is torch.Tensor:
            self.Location = self.Location + self.grad * lr
        return

    # TODO: offset p_n

class Agg_LSTS():
    def __init__(self, In_Size):
        self.weight_SRFU = torch.rand(In_Size)
        self.In_Size = In_Size
        self.N = In_Size*In_Size

        ## for testing purpose
        self.F_0 = torch.ones([In_Size, In_Size])
        self.pn = torch.ones([In_Size, In_Size])
        self.F_1_pred = torch.ones([In_Size, In_Size])

    def Aggregation(self, S_pn):
        for i in range(self.In_Size):
            for j in range(self.In_Size):
                self.F_1_pred = S_pn[i, j]*self.F_0[i, j]

    def Updating(self, S_pn):
        self.F_1 = self.Aggregation(S_pn)
        print("Updating")
        return(self.F_1)




# In_Size = 20
# Compare = Com_LSTS(In_Size)
# G = Compare.WeightGenerate()
# print(G)
# Aggreg = Agg_LSTS(In_Size)
# diff = Aggreg.Updating(G)
#
# print(diff)
