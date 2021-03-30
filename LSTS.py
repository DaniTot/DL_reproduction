import torch
import torch.nn.functional as F
import numpy as np

class Com_LSTS():
    def __init__(self, In_Shape, N=5):
        # self.weight_SRFU = torch.rand(In_Shape)
        self.In_Shape = In_Shape
        self.N = N

        self.feature_quality = None

        # Location is of shape (N, 2)
        # TODO: do we sample all the channels of the feature vecto at the same location?
        self.Location = np.array([np.random.randint(0, In_Shape[2]),
                                  np.random.randint(0, In_Shape[3])])
        for n in range(N-1):
            self.Location = np.vstack((self.Location,
                                       np.array([np.random.randint(0, In_Shape[2]),
                                                 np.random.randint(0, In_Shape[3])])
                                       ))

        ## for testing purpose

        self.F_0_p = None
        self.s_p = None
        self.S_p = np.zeros(N)

        self.F_0 = None
        self.F_1 = None

        # Initialize the low2high convolution network

        self.low2high = torch.nn.Sequential(torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1,
                                                            stride=1, padding=0, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                                            stride=1, padding=1, bias=True),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3,
                                                            stride=1, padding=1, bias=True))

        self.quality = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1024, out_channels =256, kernel_size = (1,1)),
            torch.nn.Conv2d(in_channels = 256, out_channels= 16, kernel_size= (3,3), padding=(1,1)),
            torch.nn.Conv2d(in_channels = 16, out_channels= 1, kernel_size= (3,3), padding=(1,1))
        )

        print('Initialization done!')

    def get_input(self, feat1, feat2):
        # TODO: which one is the old key, and which one is the noew key, and where does the non key frame go?
        self.F_0 = feat1
        self.F_1 = feat2

    def quality_network(self, feat1, feat2):
        feature_concat = torch.cat([feat1, feat2], dim=0)
        quality_res = self.quality(feature_concat)
        quality_weights = torch.split(quality_res, quality_res.shape[0]/2, dim=0)

        #TODO: make sure dim=1 is correct. MXNet default is axis=-1.
        quality_weights = F.softmax(torch.cat([quality_weights[0], quality_weights[1]], dim=1))
        quality_weights_splice = torch.split(quality_weights, quality_weights.shape[0]/2, dim=0)

        quality_weight1 = torch.tile(quality_weights_splice[0], (1, 1024, 1, 1))
        quality_weight2 = torch.tile(quality_weights_splice[1], (1, 1024, 1, 1))

        feat_result = torch.add(quality_weight1*feat1, quality_weight2*feat2)
        self.feature_quality = feat_result
        return feat_result

    def low2high_transform(self, nonkey):
        high_feat_current = self.low2high(nonkey)
        return high_feat_current

    def bilinear_sampling(self, input, offset):
        # TODO: how to get the grid from the offset?
        grid = offset
        return torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)

    def f(self, F_0):
        """Embedding function f"""
        embedding = torch.nn.Conv2d(F_0.shape[1], 256, kernel_size=(1, 1), stride=1, padding=0)
        fF_0 = embedding(F_0)
        return fF_0

    def BiLinKernel(self, q, p):
        # TODO: In bilinear kernel G(p_n, q), p_n has shape (N, 2), and q has shape of (2). How does that work?
        x = torch.max(torch.as_tensor([0, 1 - torch.abs(q[0] - p[0])]))
        y = torch.max(torch.as_tensor([0, 1 - torch.abs(q[1] - p[1])]))
        c = x * y
        return c

    def BiLinKernel_vec(self, q, p):
        # TODO: In bilinear kernel G(p_n, q), p_n has shape (N, 2), and q has shape of (2). How does that work?

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
        # TODO: What is q?
        #  Text says: q enumerates all integral spatial locations on the feature map f(Ft).
        #  What does this mean? Is it just all the [w, h] of embedded F_t?
        #  Do we sample F(F_t) at q in the summation?

        F_0_embedded = self.f(self.F_0)
        for n in range(self.N):
            sum = None
            for i in range(0, F_0_embedded.shape[2]):
                for j in range(0, F_0_embedded.shape[3]):
                    G = self.BiLinKernel_vec(torch.as_tensor(self.Location), torch.as_tensor((i, j)))
                    product = G[n] * F_0_embedded[:, :, i, j]
                    if sum is None:
                        sum = product
                    else:
                        sum += product

            if self.F_0_p is None:
                self.F_0_p = sum
            else:
                self.F_0_p = torch.cat((self.F_0_p, sum), 0)

    def dot_product(self):
        # TODO: We must take a sample of g(F_t+k), like self.f(self.F_1)[:, :, x, y].
        #  The text says g(Ft+k)p0 denote the features at location p0.
        #  What is p0? Is p0 the average location of p_n?

        # TODO: Some of the dot products (similarities) are negative, because F_0_p is sometimes negative.
        #  Is that normal?
        p_0 = np.around(np.sum(self.Location, axis=0)/self.N).astype(int)
        s_p = []
        for n in range(self.N):
            dot_product = torch.dot(self.F_0_p[n], self.f(self.F_1)[0, :, p_0[0], p_0[1]])
            s_p.append(dot_product)

        self.s_p = torch.tensor(s_p)

    def Normalize(self):
        # TODO: Do we normalize based on the sum, or the sum of absolutes?
        self.S_p = self.s_p/torch.sum(self.s_p)

    def WeightGenerate(self):
        self.Sum_q()
        self.dot_product()
        self.Normalize()
        assert False

    def AlignFeatures(self):
        # TODO: How to align conv_feat_oldkey to conv_feat_newkey?
        return

class Agg_LSTS():
    def __init__(self, In_Size):
        self.weight_SRFU = torch.rand(In_Size)
        self.In_Size = In_Size
        self.N = In_Size*In_Size

        ## for testing purpose
        self.Ft = torch.ones([In_Size, In_Size])
        self.pn = torch.ones([In_Size, In_Size])
        self.F_1 = torch.ones([In_Size, In_Size])

    def Aggregation(self, S_pn):
        for i in range(self.In_Size):
            for j in range(self.In_Size):
                self.F_1 = S_pn[i, j]*self.Ft[i, j]

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
