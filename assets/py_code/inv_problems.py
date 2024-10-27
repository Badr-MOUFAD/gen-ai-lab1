# Code adapted from
# https://github.com/bahjat-kawar/ddrm/blob/master/functions/svd_replacement.py
import torch


class BaseSVD:
    """Base class for a degradation operator defined by its SVD.

    The SVD decomposition reads::

        y = U @ Sigma @ V.T
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()

        # NOTE: Handle the case when H is rank deficit
        inv_singulars = 1 / singulars
        inv_singulars[singulars == 0.0] = 0.0

        temp[:, : singulars.shape[0]] = temp[:, : singulars.shape[0]] * inv_singulars
        return self.V(self.add_zeros(temp))


class Inpainting(BaseSVD):

    def __init__(self, channels, img_dim, missing_indices, device):
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.ones(
            channels * img_dim**2 - missing_indices.shape[0]
        ).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = (
            torch.Tensor(
                [i for i in range(channels * img_dim**2) if i not in missing_indices]
            )
            .to(device)
            .long()
        )

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, : self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0] :]
        return (
            out.reshape(vec.shape[0], -1, self.channels)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )

    def Vt(self, vec):
        temp = (
            vec.clone()
            .reshape(vec.shape[0], self.channels, -1)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )
        out = torch.zeros_like(temp)
        out[:, : self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0] :] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros(
            (vec.shape[0], self.channels * self.img_dim**2), device=vec.device
        )
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


class BoxInpainting(Inpainting):
    """Mask the center of an image.

    The Mask is a square at the center of the image with shape 50x50.
    """

    def __init__(self, n_channels=3, img_dim=256, device="cpu"):

        C, H, W = torch.meshgrid(
            torch.arange(n_channels),
            torch.arange(150, 200),
            torch.arange(100, 150),
            indexing="ij",
        )
        missing_indices = (H * 3 * img_dim + W * 3 + C).flatten()

        super().__init__(n_channels, img_dim, missing_indices, device)


class Colorization(BaseSVD):

    def __init__(self, img_dim=256, device="cpu"):
        self.channels = 3
        self.img_dim = img_dim
        # Do the SVD for the per-pixel matrix
        H = torch.Tensor([[0.3333, 0.3334, 0.3333]]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # get the needles
        needles = (
            vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)
        )  # shape: B, WH, C'
        # multiply each needle by the small V
        needles = torch.matmul(
            self.V_small, needles.reshape(-1, self.channels, 1)
        ).reshape(
            vec.shape[0], -1, self.channels
        )  # shape: B, WH, C
        # permute back to vector representation
        recon = needles.permute(0, 2, 1)  # shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        # get the needles
        needles = (
            vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)
        )  # shape: B, WH, C
        # multiply each needle by the small V transposed
        needles = torch.matmul(
            self.Vt_small, needles.reshape(-1, self.channels, 1)
        ).reshape(
            vec.shape[0], -1, self.channels
        )  # shape: B, WH, C'
        # reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], self.channels * self.img_dim**2), device=vec.device
        )
        temp[:, : self.img_dim**2] = reshaped
        return temp


class SuperResolution(BaseSVD):

    def __init__(self, channels, img_dim, ratio, device="cpu"):  # ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(
            vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device
        )
        patches[:, :, :, 0] = temp[:, : self.channels * self.y_dim**2].view(
            vec.shape[0], self.channels, -1
        )
        for idx in range(self.ratio**2 - 1):
            patches[:, :, :, idx + 1] = temp[
                :, (self.channels * self.y_dim**2 + idx) :: self.ratio**2 - 1
            ].view(vec.shape[0], self.channels, -1)
        # multiply each patch by the small V
        patches = torch.matmul(
            self.V_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # repatch the patches into an image
        patches_orig = patches.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio
        )
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim**2)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim, self.img_dim
        )
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )
        # multiply each by the small V transposed
        patches = torch.matmul(
            self.Vt_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(
            vec.shape[0], self.channels * self.img_dim**2, device=vec.device
        )
        recon[:, : self.channels * self.y_dim**2] = patches[:, :, :, 0].view(
            vec.shape[0], self.channels * self.y_dim**2
        )
        for idx in range(self.ratio**2 - 1):
            recon[:, (self.channels * self.y_dim**2 + idx) :: self.ratio**2 - 1] = (
                patches[:, :, :, idx + 1].view(
                    vec.shape[0], self.channels * self.y_dim**2
                )
            )
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device
        )
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


class SR4(SuperResolution):
    """Super Resolution with x4 factor."""

    def __init__(self, n_channels=3, img_dim=256, device="cpu"):
        super().__init__(n_channels, img_dim, ratio=4, device=device)
