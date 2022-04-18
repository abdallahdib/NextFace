import torch

class TextureLoss:
	def __init__(self, device):
		self.device = device

		self.RGB2XYZ = torch.tensor([[41.2390799265959, 35.7584339383878, 18.0480788401834],
									 [21.2639005871510, 71.5168678767756, 07.2192315360734],
									 [01.9330818715592, 11.9194779794626, 95.0532152249661]], dtype=torch.float).to(self.device)

	def regTextures(self, vTex, refTex, ws=3., wr=10.0, wc=10., wsm=0.01, wm=0.):
		'''
		regularize vTex with respect to refTex (more on this here: https://arxiv.org/abs/2101.05356)
		:param vTex: first texture [n, w, h, 3/1/]
		:param refTex: second texture [n, w, h, 3/1]
		:param ws: symmetry regularizer
		:param wr: rgb regularizer
		:param wc: consisntecy regularizer
		:param wsm: smoothness regularizer
		:param wm: mean regularizer
		:return: scalar loss
		'''
		symReg = (vTex - vTex.flip([2])).abs().mean()  # symmetry regularizer on vertical axis
		rgbReg = (vTex - refTex).abs().mean()  # rgb regularization with respect to reference texture
		loss = ws * symReg + wr * rgbReg

		loss += 1000.0 * torch.clamp(-vTex, min=0).mean()  # soft penalize < 0
		loss += 1000.0 * torch.clamp(vTex - 1.0, min=0).mean()  # soft penalize > 1

		loss += wsm * ((vTex[:, 1:] - vTex[:, :-1]).pow(2).sum())  # smooth on  y axis
		loss += wsm * ((vTex[:, :, 1:] - vTex[:, :, :-1]).pow(2).sum())  # smooth on x axis

		if wc > 0:  # regularize in xyz space
			refTex_XYZ = torch.matmul(self.RGB2XYZ, refTex[..., None])[..., 0]
			refTex_xyz = refTex_XYZ[..., :2] / (1.0 + refTex_XYZ.sum(dim=-1, keepdim=True))
			vTex_XYZ = torch.matmul(self.RGB2XYZ, vTex[..., None])[..., 0]
			vTex_xyz = vTex_XYZ[..., :2] / (1.0 + torch.clamp(vTex_XYZ, min=0.).sum(dim=-1, keepdim=True))
			xy_regularization = (refTex_xyz - vTex_xyz).abs().mean()
			loss += wc * xy_regularization

		if wm > 0:  # keep close to average (generally for specular map)
			loss += wm * ((vTex - vTex.mean(dim=-1, keepdim=True)).pow(2).sum())

		return loss
