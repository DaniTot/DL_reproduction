import torch
import numpy as np

class Com_LSTS():
    def __init__(self, In_Size):
        self.weight_SRFU = torch.rand(In_Size)
        self.In_Size = In_Size

        ## for testing purpose
        self.F_t = torch.ones([In_Size, In_Size]) * 2
        self.F_t_pn = torch.ones([In_Size, In_Size])
        self.tk = torch.ones([In_Size, In_Size])
        self.Location = np.random.randint(0, In_Size, 2)
        self.s_pn = torch.ones([In_Size, In_Size])
        self.S_pn = torch.ones([In_Size, In_Size])
        self.g_F_tk = torch.ones([In_Size, In_Size])

        print('Initialization done!')

    def BiLinKernel(self, a, b):
        x = torch.clamp(1 - torch.abs(a - b), 0)
        y = torch.clamp(1 - torch.abs(a - b), 0)
        c = x * y
        return c

    def Sum_q(self):
        p_n = self.F_t[self.Location[0], self.Location[1]]

        for i in range(0, self.In_Size):
            for j in range(0, self.In_Size):
                self.F_t_pn = self.F_t_pn + (self.BiLinKernel(p_n, self.F_t[i, j]) * self.F_t)

    def dot_product(self):
        self.s_pn = torch.tensordot(self.F_t_pn, self.g_F_tk)

    def Normalize(self):
        self.S_pn = self.s_pn/torch.sum(self.s_pn)

    def Run(self):
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
        self.F_tk = torch.ones([In_Size, In_Size])

    def Aggregation(self):
        for i in range(self.In_Size):
            for j in range(self.In_Size):
                self.F_tk = Compare.S_pn[i, j]*self.Ft[i, j]

    def Updating(self):
        self.F_tk = self.Aggregation()
        print("Updating")
        return(self.F_tk)


In_Size = 5
Compare = Com_LSTS(In_Size)
G = Compare.Run()
#Aggreg = Agg_LSTS(In_Size)
#diff = Aggreg.Updating()

print(diff)
