import torch
import torch.nn.functional as F

from .partial_base_model import PartialBaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from utils.dpfm_util import cache_sample_idx
import os

@MODEL_REGISTRY.register()
class DPFM_Model(PartialBaseModel):
    def __init__(self, opt):
        super(DPFM_Model, self).__init__(opt)

    def feed_data(self, data):
        # quirk for dpfm attention
        cross_sampling_ratio = self.opt["networks"]['dpfm_net']['cfg']['attention']['cross_sampling_ratio']
        cache_sample_idx(data, cross_sampling_ratio=cross_sampling_ratio, cache_dir=os.path.join(self.data_root, 'sample_idx')) # for caching opereation can use 1.0 and afterwards will be flexible for any ratio

        # get data pair
        data = to_device(data, self.device)
        data_x, data_y = data['first'], data['second']

        # get spectral operators
        evecs_x, evecs_y = data_x['evecs'][0], data_y['evecs'][0]
        evals_x, evals_y = data_x['evals'][0], data_y['evals'][0]
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'][0], data_y['evecs_trans'][0]

        # extract features
        feat_x = self.networks['dpfm_net'].feature_extractor(data=data_x)
        feat_y = self.networks['dpfm_net'].feature_extractor(data=data_y)

        # attention overlap prediction
        ref_feat_x, ref_feat_y, overlap_score12, overlap_score21 = self.networks['dpfm_net'].feat_refiner(data_x['xyz'][0], data_y['xyz'][0], feat_x, feat_y, data)
        use_feat_x, use_feat_y = (ref_feat_x, ref_feat_y) if self.networks['dpfm_net'].robust else (feat_x, feat_y)

        # predict fmap
        Cxy = self.networks['dpfm_net'].fmreg_net(use_feat_x, use_feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        # gt partiality mask
        gt_partiality_mask12 = data['first'].get('partiality_mask', torch.ones((evecs_x.shape[0])).long().to(self.device)).squeeze()
        gt_partiality_mask21 = data['second'].get('partiality_mask', torch.ones((evecs_y.shape[0])).long().to(self.device)).squeeze()

        # gt correspondence
        corr_x = data['first']['corr'][0]
        corr_y = data['second']['corr'][0]

        # gt functional map
        P = torch.zeros((evecs_y.shape[0], evecs_x.shape[0])).to(self.device) # [Ny, Nx]
        P[corr_y, corr_x] = 1
        C_gt = evecs_trans_y @ P @ evecs_x
        C_gt = C_gt.unsqueeze(0)

        # loss
        fmap_loss, acc_loss, nce_loss = self.losses["dpfm_loss"](C_gt, Cxy, corr_x, corr_y, use_feat_x, use_feat_y,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

        self.loss_metrics = {'fmap_loss': fmap_loss, 'acc_loss': acc_loss, 'nce_loss': nce_loss}

    def validate_single(self, data, timer):
        # start record
        timer.start()
        
        # quirk for dpfm attention
        cross_sampling_ratio = self.opt["networks"]['dpfm_net']['cfg']['attention']['cross_sampling_ratio']
        cache_sample_idx(data, cross_sampling_ratio=cross_sampling_ratio, cache_dir=os.path.join(self.data_root, 'sample_idx')) # for caching opereation can use 1.0 and afterwards will be flexible for any ratio

        # get data pair
        data = to_device(data, self.device)
        data_x, data_y = data['first'], data['second']

        # get spectral operators
        evecs_x, evecs_y = data_x['evecs'][0], data_y['evecs'][0]
        evals_x, evals_y = data_x['evals'][0], data_y['evals'][0]
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'][0], data_y['evecs_trans'][0]

        # extract features
        feat_x = self.networks['dpfm_net'].feature_extractor(data=data_x)
        feat_y = self.networks['dpfm_net'].feature_extractor(data=data_y)

        # attention overlap prediction
        ref_feat_x, ref_feat_y, overlap_score12, overlap_score21 = self.networks['dpfm_net'].feat_refiner(data_x['xyz'][0], data_y['xyz'][0], feat_x, feat_y, data)
        use_feat_x, use_feat_y = (ref_feat_x, ref_feat_y) if self.networks['dpfm_net'].robust else (feat_x, feat_y)

        # predict fmap
        Cxy = self.networks['dpfm_net'].fmreg_net(use_feat_x, use_feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        Cxy = Cxy.squeeze()

        # convert functional map to point-to-point map
        p2p_fmap = fmap2pointmap(Cxy, evecs_x, evecs_y)

        # direct point-to-point map
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        p2p_point = nn_query(feat_x, feat_y)

        # finish record
        timer.record()

        return {
            'p2p': p2p_fmap,
            'overlap_score12': overlap_score12,
            'overlap_score21': overlap_score21,
            'p2p_variants': {
                'fmap': p2p_fmap,
                'point': p2p_point
            }
        }




    