import torch
import torch.nn as nn
import numpy as np
import pickle
import healpy as hp

from . import lie_tools
from . import utils
log = utils.log

class PoseTracker(nn.Module):
    def __init__(self, rots_np, trans_np=None, D=None, emb_type=None, deform=False, deform_emb_size=2, eulers_np=None, latents=None, batch_size=None, hp_order=2):
        super(PoseTracker, self).__init__()
        rots = torch.tensor(rots_np).float()
        trans = torch.tensor(trans_np).float() if trans_np is not None else None
        self.eulers = torch.tensor(eulers_np).float() if eulers_np is not None else None
        self.rots = rots
        self.trans = trans
        self.use_trans = trans_np is not None
        self.D = D
        self.emb_type = emb_type
        self.deform = deform
        self.deform_emb = None
        if emb_type is None:
            pass
        else:
            if self.use_trans:
                trans_emb = nn.Embedding(trans.shape[0], 2, sparse=True)
                trans_emb.weight.data.copy_(trans)
            if emb_type == 's2s2':
                rots_emb = nn.Embedding(rots.shape[0], 6, sparse=True)
                rots_emb.weight.data.copy_(lie_tools.SO3_to_s2s2(rots))
            elif emb_type == 'quat':
                rots_emb = nn.Embedding(rots.shape[0], 4, sparse=True)
                rots_emb.weight.data.copy_(lie_tools.SO3_to_quaternions(rots))
            else:
                raise RuntimeError('Embedding type {} not recognized'.format(emb_type))
            self.rots_emb = rots_emb
            self.trans_emb = trans_emb if self.use_trans else None
        if self.deform:
            self.deform_emb_size = deform_emb_size
            #deform_emb = nn.Embedding(rots.shape[0], deform_emb_size, sparse=True)
            #deform_emb = torch.zeros(rots.shape[0], deform_emb_size)
            #if encoding is not None:
            #    emb_data = encoding.repeat(rots.shape[0], 1)
            #    print(emb_data.shape)
            #else:
            #    emb_data = torch.randn(rots.shape[0], deform_emb_size)
            #deform_emb.weight.data.copy_(emb_data)
            #self.deform_emb = deform_emb
            # convert poses to healpix indicies
            print(eulers_np[:5, :])
            euler0 = eulers_np[:, 0]*np.pi/180 #(-180, 180)
            euler1 = eulers_np[:, 1]*np.pi/180 #(0, 180)
            self.hp_order = hp_order
            euler_pixs = hp.ang2pix(self.hp_order, euler1, euler0, nest=True)
            num_pixs   = self.hp_order**2*12
            self.poses_ind = [[] for i in range(num_pixs)]
            for i in range(len(euler_pixs)):
                assert euler_pixs[i] < num_pixs
                self.poses_ind[euler_pixs[i]].append(i)
            self.poses_ind = [torch.tensor(x) for x in self.poses_ind]
            self.euler_groups = euler_pixs
            if latents is not None:
                self.mu = latents
                #self.mu = latents["mu"]
                #self.nearest_poses = latents["nn"]
            else:
                self.mu = torch.randn(rots.shape[0], self.deform_emb_size)
            self.nearest_poses = [np.array([], dtype=np.int64) for i in range(len(euler_pixs))]
            self.batch_size = batch_size
            self.num_gpus = 4
            print("nn: ", len(self.nearest_poses), "batch_size: ", self.batch_size)
            self.ns = [(len(x) // self.batch_size)*self.batch_size for x in self.poses_ind]
            self.total_ns = sum(self.ns)
            print(self.ns)
            self.valid_poses = []
            for i in range(num_pixs):
                if self.ns[i] > 0:
                    self.valid_poses.append(i)
            #print("poses_ind: ", self.poses_ind)
            print(len(euler_pixs), len(eulers_np), self.valid_poses)

    def sample_neighbors(self, euler, inds, num_pose=8):
        cur_idx = self.euler_groups[inds[0]]
        euler0 = euler[0, 0]*np.pi/180
        euler1 = euler[0, 1]*np.pi/180
        cur_idx_ = hp.ang2pix(self.hp_order, euler1, euler0, nest=True)
        assert cur_idx == cur_idx_
        pose_sample = list(self.valid_poses)
        pose_sample.remove(cur_idx)
        perm = np.random.choice(pose_sample, size=num_pose, replace=False)
        total = sum([self.ns[i] for i in perm])
        sample_idices = []
        sample_mus = []
        #print(cur_idx, perm)
        #num_pose = min(len(pose_sample), num_pose)
        total_samples = self.batch_size*2*num_pose
        for i in range(len(perm)):
            #pose_idx = pose_sample[i] #
            pose_idx = perm[i]
            # sample from selected pose
            samples = np.random.choice(self.ns[pose_idx],
                    size=int(self.ns[pose_idx]/total*total_samples), replace=False)
            #print(samples)
            idx_ = self.poses_ind[pose_idx][samples]
            sample_idices.append(idx_)
            sample_mus.append(self.mu[idx_,:])
        #print(total)
        #sample_idices = np.concatenate(sample_idices, axis=0)
        # compare with current nearest neighbors
        mus = []
        top_indices = []
        top_mus = []
        #uniq_indices = np.array([], dtype=np.int64)
        num_samples = 320
        for i in range(len(inds)):
            global_i  = inds[i]
            #nearest_i = self.nearest_poses[global_i]
            #n_i = np.unique(np.concatenate([nearest_i, sample_idices], axis=0))
            mu_i = self.mu[global_i, :]
            #mu_ = self.mu[n_i, :]
            #diff = (mu_i - mu_).pow(2).sum(-1)
            #print("diff: ", diff.mean(), diff.std())
            n_i = []
            for i in range(len(perm)):
                pose_idx = perm[i]
                fraction = self.ns[pose_idx]/total
                # sample from selected pose
                #samples = np.random.choice(self.ns[pose_idx],
                #        size=int(fraction*total_samples), replace=False)
                #print(samples)
                idx_ = sample_idices[i]
                mu_ = sample_mus[i]
                diff = (mu_i - mu_).pow(2).sum(-1)
                top = torch.topk(diff, k=int(fraction*num_samples), largest=False, sorted=True)
                n_i.append(idx_[top.indices.detach().cpu()])
            n_i = np.concatenate(n_i, axis=0)
            self.nearest_poses[global_i] = n_i
            #sample_idices = np.concatenate(sample_idices, axis=0)

            #print(top.values.max())
            # convert top indices to global indices
            #self.nearest_poses[global_i] = n_i[top.indices.detach().cpu()]
            #mu_ = self.mu[self.nearest_poses[global_i], :]
            mu_ = self.mu[n_i, :]
            diff = (mu_i - mu_).pow(2).sum(-1)
            top = torch.topk(diff, k=5, largest=False, sorted=True)
            # gather output
            mus.append(mu_)
            top_indices.append(torch.from_numpy(n_i[top.indices]))
            top_mus.append(self.mu[n_i[top.indices], :])
            #uniq_indices = np.unique(np.concatenate([uniq_indices, self.nearest_poses[global_i]], axis=0))
            #print(mu_[:1, :])
            # update nn of top_indices
            #for j in range(self.batch_size):
            #    self.nearest_poses[self.nearest_poses[global_i][j]] = np.append(
            #        self.nearest_poses[self.nearest_poses[global_i][j]], global_i)
        #print("uniq_indices", len(uniq_indices))
        #uniq_mus = self.mu[uniq_indices, :].mean(dim=0)
        #print(uniq_mus.mean(dim=0))
        mus = torch.stack(mus, dim=0)
        top_indices = torch.stack(top_indices, dim=0) #(B*k)
        top_mus = torch.stack(top_mus, dim=0)
        #print(mus.shape, top_indices.shape, top_mus.shape)
        return mus, top_indices.view(-1), top_mus

    def set_emb(self, encodings, ind):
        self.mu[ind] = encodings.detach().cpu()

    def save_emb(self, filename):
        torch.save({"mu": self.mu, "nn": self.nearest_poses}, filename)

    @classmethod
    def load(cls, infile, Nimg, D, emb_type=None, ind=None, deform=False, deform_emb_size=2, latents=None, batch_size=None, hp_order=1):
        '''
        Return an instance of PoseTracker

        Inputs:
            infile (str or list):   One or two files, with format options of:
                                    single file with pose pickle
                                    two files with rot and trans pickle
                                    single file with rot pickle
            Nimg:               Number of particles
            D:                  Box size (pixels)
            emb_type:           SO(3) embedding type if refining poses
            ind:                Index array if poses are being filtered
        '''
        # load pickle
        if type(infile) is str: infile = [infile]
        assert len(infile) in (1,2)
        if len(infile) == 2: # rotation pickle, translation pickle
            poses = (utils.load_pkl(infile[0]), utils.load_pkl(infile[1]))
        else: # rotation pickle or poses pickle
            poses = utils.load_pkl(infile[0])
            if type(poses) != tuple: poses = (poses,)

        # rotations
        rots = poses[0]
        if ind is not None:
            if len(rots) > Nimg: # HACK
                rots = rots[ind]
        assert rots.shape == (Nimg,3,3), f"Input rotations have shape {rots.shape} but expected ({Nimg},3,3)"

        # translations if they exist
        if len(poses) == 2:
            trans = poses[1]
            if ind is not None:
                if len(trans) > Nimg: # HACK
                    trans = trans[ind]
            assert trans.shape == (Nimg,2), f"Input translations have shape {trans.shape} but expected ({Nimg},2)"
            assert np.all(trans <= 1), "ERROR: Old pose format detected. Translations must be in units of fraction of box."
            trans *= D # convert from fraction to pixels
        elif len(poses) == 3:
            trans = poses[1]
            if ind is not None:
                if len(trans) > Nimg: # HACK
                    trans = trans[ind]
            assert trans.shape == (Nimg,2), f"Input translations have shape {trans.shape} but expected ({Nimg},2)"
            assert np.all(trans <= 1), "ERROR: Old pose format detected. Translations must be in units of fraction of box."
            trans *= D # convert from fraction to pixels
            log("loaded eulers")
            eulers = poses[2]
            if ind is not None:
                if len(trans) > Nimg: # HACK
                    eulers = eulers[ind]
            assert eulers.shape == (Nimg,3), f"Input translations have shape {trans.shape} but expected ({Nimg},2)"
        else:
            log('WARNING: No translations provided')
            trans = None
            eulers = None

        if latents is not None:
            latents = torch.load(latents)
        return cls(rots, trans, D, emb_type, deform, deform_emb_size, eulers, latents, batch_size, hp_order)

    def save(self, out_pkl):
        if self.emb_type == 'quat':
            r = lie_tools.quaternions_to_SO3(self.rots_emb.weight.data).cpu().numpy()
        elif self.emb_type == 's2s2':
            r = lie_tools.s2s2_to_SO3(self.rots_emb.weight.data).cpu().numpy()
        else:
            r = self.rots.cpu().numpy()

        if self.use_trans:
            if self.emb_type is None:
                t = self.trans.cpu().numpy()
            else:
                t = self.trans_emb.weight.data.cpu().numpy()
            t /= self.D # convert from pixels to extent
            if self.eulers is not None:
                e = self.eulers.cpu().numpy()
                poses = (r,t,e)
            else:
                poses = (r,t)
        else:
            poses = (r,)

        pickle.dump(poses, open(out_pkl,'wb'))

    def get_euler(self, ind):
        if self.emb_type is None:
            euler = self.eulers[ind]
            return euler

    def get_pose(self, ind):
        if self.emb_type is None:
            rot = self.rots[ind]
            tran = self.trans[ind] if self.use_trans else None
        else:
            if self.emb_type == 's2s2':
                rot = lie_tools.s2s2_to_SO3(self.rots_emb(ind))
            elif self.emb_type == 'quat':
                rot = lie_tools.quaternions_to_SO3(self.rots_emb(ind))
            else:
                raise RuntimeError # should not reach here
            tran = self.trans_emb(ind) if self.use_trans else None
        #if self.deform:
        #    defo = self.deform_emb(ind)
        #    return rot, tran, defo
        return rot, tran
