import torch
import torch.nn as nn
import torchaudio


'''Reference: https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362?fbclid=IwAR3PfhU3Xs37sJ39-E9HiBcxth_V499I4kLU0lIMiYfG9pNKyce245sihfM'''
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Classifier(nn.Module):
	def __init__(self, config, d_model=100, n_spks=600, dropout=0.1):
		super().__init__()
		self.prenet = nn.Linear(40, d_model)
		self.conformer = torchaudio.models.Conformer(**config)
		self.pooling = SelfAttentionPooling(d_model)
		self.pred_layer = nn.Sequential(
			nn.PReLU(),
			nn.Linear(d_model, n_spks),
		)

	def forward(self, mels):
		out = self.prenet(mels)
		len = torch.full((out.shape[0], ), out.shape[1]).cuda()
		out, _ = self.conformer(out, len)
		stats = self.pooling(out)
		out = self.pred_layer(stats)
		w = torch.norm(self.pred_layer[-1].weight, p=2, dim=1)
		f = torch.norm(stats, p=2, dim=1)
		norm = torch.ger(f, w)
		cosine = out / norm
		return out, cosine
