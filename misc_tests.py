import model
import numpy as np

if __name__ == '__main__':
	mod = model.GitHubClassifier()
	# with open("espnet_all_data.txt", encoding='utf-8') as f:
	# 	lines = f.readlines()
	print(mod.classify(['jina', 'cards', 'cards', 'horovod']))

	# for p, (handle, ctx) in self._handles.items():
	# 	if self.training:
	# 		self.bleu = None
	# 	else:
	# 		ys_hat = pred_pad.argmax(dim=-1)
	# 		self.bleu = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
