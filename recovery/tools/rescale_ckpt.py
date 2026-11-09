"""Convert an rsl_rl checkpoint from action scale s_old to s_new so the POLICY IS FUNCTIONALLY IDENTICAL: output layer (and std) x s_old/s_new, and the input weights of the
last-action observation (last 37 of 123 inputs) x s_new/s_old for actor and critic. Usage: rescale_ckpt.py in.pt out.pt 0.5 1.0 [keep_std]
With keep_std the MEAN action is equivalent but sigma stays the same in action units, i.e. the joint-target exploration noise in radians scales by s_new/s_old."""
import sys, torch
src, dst, s_old, s_new = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); k = s_old / s_new
c = torch.load(src, map_location="cpu", weights_only=True)
a, cr = c["actor_state_dict"], c["critic_state_dict"]
keep_std = len(sys.argv) > 5 and sys.argv[5] == "keep_std"
a["mlp.6.weight"] *= k; a["mlp.6.bias"] *= k
if not keep_std: a["distribution.std_param"] *= k
a["mlp.0.weight"][:, -37:] /= k; cr["mlp.0.weight"][:, -37:] /= k
torch.save(c, dst); print("converted", src, "->", dst, "k=", k, "keep_std=", keep_std, "std mean", float(a["distribution.std_param"].mean()))
