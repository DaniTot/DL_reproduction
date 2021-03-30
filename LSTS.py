import torch
import torch.nn.functional as F
import numpy as np

class Com_LSTS():
    def __init__(self, In_Shape):
        # self.weight_SRFU = torch.rand(In_Shape)
        # self.In_Size = In_Shape

        self.feature_quality = None

        #TODO: do we sample all the channels of the feature vecto at the same location?
        self.Location = [np.random.randint(0, In_Shape[2], 1), np.random.randint(0, In_Shape[3], 1)]

        ## for testing purpose

        self.F_0_pn = None
        self.tk = None
        self.s_pn = None
        self.S_pn = None

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
        # c is always 0. Is that normal?
        x = torch.max(torch.as_tensor([0, 1 - torch.abs(q[0] - p[0])]))
        y = torch.max(torch.as_tensor([0, 1 - torch.abs(q[1] - p[1])]))
        c = x * y
        return c

    def Sum_q(self):
        """
        Populates/updates tensor matrices p_n and F_0_pn
        """
        p_n = self.F_0[self.Location[0], self.Location[1]]

        for i in range(0, self.In_Size):
            for j in range(0, self.In_Size):
                self.F_0_pn = self.F_0_pn + (self.BiLinKernel(torch.as_tensor(p_n), torch.as_tensor((i, j))) * self.f(self.F_0)[i, j])

    def dot_product(self):
        self.s_pn = torch.tensordot(self.F_0_pn, self.f(self.F_1))

    def Normalize(self):
        self.S_pn = self.s_pn/torch.sum(self.s_pn)

    def WeightGenerate(self):
        self.Sum_q()
        self.dot_product()
        self.Normalize()

        return self.S_pn

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


In_Size = 20
Compare = Com_LSTS(In_Size)
G = Compare.WeightGenerate()
print(G)
Aggreg = Agg_LSTS(In_Size)
diff = Aggreg.Updating(G)

print(diff)
