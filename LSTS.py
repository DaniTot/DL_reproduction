import torch
import numpy as np

class Com_LSTS():
    def __init__(self, In_Size, N=4, p_size=3):
        self.weight_SRFU = torch.rand(In_Size)
        self.In_Size = In_Size

        ## for testing purpose
        self.F_t = torch.ones([In_Size, In_Size]) * 2
        self.F_t_pn = torch.ones([N, In_Size, In_Size])
        self.tk = torch.ones([In_Size, In_Size])
        self.s_pn = torch.ones([In_Size, In_Size])
        self.S_pn = torch.ones([In_Size, In_Size])
        self.F_tk = torch.ones([In_Size, In_Size])

        self.p_size = p_size                                     # Sample matrix shape will be (p_size, p_size)
        self.N = N                                          # Number of sample matrices
        self.p_n = torch.empty((self.N, self.p_size, self.p_size))

        ###
        # The assserts below make sure that there is no overlap between the sampling matrices.
        ###
        # assert self.In_Size % self.p_size == 0
        # assert self.N * self.p_size < self.In_Size

        # The location matrix containts N number of x-y coordinate pairs,
        # defining the top left corner pixel of each sampling matrix.
        if self.In_Size % self.p_size == 0 and self.N * self.p_size < self.In_Size:
            self.Locations = np.random.randint(0, self.In_Size / self.p_size, (self.N, 2)) * self.p_size
        else:
            self.Locations = np.random.randint(0, self.In_Size - self.p_size, (self.N, 2))

        print('Initialization done!')

    def f(self, F_t):
        # TODO: What's the embedding function f?
        return F_t

    def g(self, F_tk):
        # TODO: What's the embedding function g?
        """Embedding function f"""
        return F_tk

    def BiLinKernel(self, q, p):
        # c is always 0. Is that normal?
        x = torch.max(torch.as_tensor([0, 1 - torch.abs(q[0] - p[0])]))
        y = torch.max(torch.as_tensor([0, 1 - torch.abs(q[1] - p[1])]))
        c = x * y
        return c

    def Sum_q(self):
        """
        Populates/updates tensor matrices p_n and F_t_pn
        """
        for n in range(self.Locations.shape[0]):
            coord_pair = self.Locations[n, :]
            self.p_n[n, :] = self.f(self.F_t)[coord_pair[0]:coord_pair[0]+self.p_size, coord_pair[1]:coord_pair[1]+self.p_size]

            for i in range(0, self.In_Size):
                for j in range(0, self.In_Size):
                    self.F_t_pn[n, :] = self.F_t_pn[n, :] + (self.BiLinKernel(torch.as_tensor(coord_pair), torch.as_tensor((i, j))) * self.f(self.F_t)[coord_pair[0], coord_pair[1]])

    def dot_product(self):
        self.s_pn = torch.tensordot(self.F_t_pn, self.g(self.F_tk))

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


In_Size = 20
Compare = Com_LSTS(In_Size)
G = Compare.Run()
#Aggreg = Agg_LSTS(In_Size)
#diff = Aggreg.Updating()

print(diff)
