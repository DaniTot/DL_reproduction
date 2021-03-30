import torch
import numpy as np

class Com_LSTS():
    def __init__(self, In_Shape):
        self.weight_SRFU = torch.rand(In_Shape)
        self.In_Size = In_Shape
        self.Location = np.random.randint(0, In_Size, 2)

        ## for testing purpose

        self.F_t_pn = None
        self.tk = None
        self.s_pn = None
        self.S_pn = None

        print('Initialization done!')

    def get_input(self, old_key, new_key):
        # TODO: which one is the old key, and which one is the noew key, and where does the non key frame go?
        self.F_t = None
        self.F_tk = None

    def sample_offset(self, N, K, C, H, W, lr):
        ###
        #   for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
        #       index < count;
        #       index += blockDim.x * gridDim.x * gridDim.y) {
        #
        #     int C1 = C / 8;
        #     const int w = index % W;
        #     const int h = (index / W) % H;
        #     const int c = (index / (H * W)) % C1;
        #     const int k = (index / (C1 * H * W)) % K;
        #     const int n = (index / (K* C1 * H * W));
        #
        #     for(int i=0;i<8;i++){
        #         DType x_real = bottom_offset[2 * k] + w;
        #         DType y_real = bottom_offset[2 * k + 1] + h;
        #
        #         const DType* data = bottom_data + n * (C * H * W) + (8*c+i) * (H * W);
        #
        #         int x1 = floor(x_real);
        #         int y1 = floor(y_real);
        #         int x2 = x1 + 1;
        #         int y2 = y1 + 1;
        #
        #         DType dist_x = static_cast<DType>(x_real - x1);
        #         DType dist_y = static_cast<DType>(y_real - y1);
        #
        #         DType value11 = 0;
        #         DType value12 = 0;
        #         DType value21 = 0;
        #         DType value22 = 0;
        #
        #         if (between(x1, 0, W-1) && between(y1, 0, H-1))
        #             value11 = *(data + y1*W + x1);
        #
        #         if (between(x1, 0, W-1) && between(y2, 0, H-1))
        #             value12 = *(data + y2*W + x1);
        #
        #         if (between(x2, 0, W-1) && between(y1, 0, H-1))
        #             value21 = *(data + y1*W + x2);
        #
        #         if (between(x2, 0, W-1) && between(y2, 0, H-1))
        #             value22 = *(data + y2*W + x2);
        #
        #         DType value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12 + dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
        #
        #         top_data[offset5d(n,k,8*c+i,h,w,N,K,C,H,W)] = value;
        #     }
        #
        # inline __device__ int offset5d(int n, int k, int c, int h, int w, int N, int K, int C, int H, int W) {
        #     return n*K*C*H*W + k*C*H*W + c*H*W + h*W + w;
        # }
        ###

        return n*K*C*H*W + k*C*H*W + c*H*W + h*W + w

    def bilinear_sampling(self, input, offset):
        # TODO: how to get the grid from the offset?
        grid = offset
        return torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)


    def f(self, F_t):
        """Embedding function f"""
        embedding = torch.nn.Conv2d(F_t.shape[1], 256, kernel_size=(1, 1), stride=1, padding=0)
        fF_t = embedding(F_t)
        return fF_t

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
        p_n = self.F_t[self.Location[0], self.Location[1]]

        for i in range(0, self.In_Size):
            for j in range(0, self.In_Size):
                self.F_t_pn = self.F_t_pn + (self.BiLinKernel(torch.as_tensor(p_n), torch.as_tensor((i, j))) * self.f(self.F_t)[i, j])

    def dot_product(self):
        self.s_pn = torch.tensordot(self.F_t_pn, self.f(self.F_tk))

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
