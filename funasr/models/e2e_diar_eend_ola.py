# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as  nn
from typeguard import check_argument_types

from funasr.models.frontend.wav_frontend import WavFrontendMel23
from funasr.modules.eend_ola.encoder import EENDOLATransformerEncoder
from funasr.modules.eend_ola.encoder_decoder_attractor import EncoderDecoderAttractor
from funasr.modules.eend_ola.utils.losses import batch_pit_n_speaker_loss, standard_loss, cal_power_loss
from funasr.modules.eend_ola.utils.power import create_powerlabel
from funasr.modules.eend_ola.utils.power import generate_mapping_dict
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    pass
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def pad_tensor(tensor, out_size):
    tensor_padded = []
    for i, t in enumerate(tensor):
        if t.shape[1] < out_size:
            # padding
            tensor_padded.append(
                torch.cat([t, torch.zeros(t.shape[0], out_size - t.shape[1]).to(torch.float32).to(t.device)], dim=1))
        else:
            tensor_padded.append(t)
    return tensor_padded


def pad_attractor(att, max_n_speakers):
    C, D = att.shape
    if C < max_n_speakers:
        att = torch.cat([att, torch.zeros(max_n_speakers - C, D).to(torch.float32).to(att.device)], dim=0)
    return att


class DiarEENDOLAModel(AbsESPnetModel):
    """EEND-OLA diarization model"""

    def __init__(
            self,
            frontend: Union[WavFrontendMel23, None],
            encoder: EENDOLATransformerEncoder,
            encoder_decoder_attractor: EncoderDecoderAttractor,
            n_units: int = 256,
            max_n_speaker: int = 8,
            attractor_loss_weight: float = 1.0,
            mapping_dict=None,
            **kwargs,
    ):
        assert check_argument_types()

        super().__init__()
        self.frontend = frontend
        self.enc = encoder
        self.eda = encoder_decoder_attractor
        self.attractor_loss_weight = attractor_loss_weight
        self.max_n_speaker = max_n_speaker
        if mapping_dict is None:
            mapping_dict = generate_mapping_dict(max_speaker_num=self.max_n_speaker)
            self.mapping_dict = mapping_dict
        self.postnet = nn.LSTM(self.max_n_speaker, n_units, 1, batch_first=True)
        self.output_layer = nn.Linear(n_units, mapping_dict['oov'] + 1)

    def forward_encoder(self, xs, ilens):
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=-1)
        pad_shape = xs.shape
        xs_mask = [torch.ones(ilen).to(xs.device) for ilen in ilens]
        xs_mask = torch.nn.utils.rnn.pad_sequence(xs_mask, batch_first=True, padding_value=0).unsqueeze(-2)
        emb = self.enc(xs, xs_mask)
        emb = torch.split(emb.view(pad_shape[0], pad_shape[1], -1), 1, dim=0)
        emb = [e[0][:ilen] for e, ilen in zip(emb, ilens)]
        return emb

    def forward_post_net(self, logits, ilens):
        maxlen = torch.max(ilens)
        logits = nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=-1)
        logits = nn.utils.rnn.pack_padded_sequence(logits, ilens, batch_first=True,
                                                   enforce_sorted=False)
        outputs, (_, _) = self.postnet(logits)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=-1, total_length=maxlen)[0]
        outputs = [output[:ilens[i].to(torch.int).item()] for i, output in enumerate(outputs)]
        outputs = [self.output_layer(output) for output in outputs]
        return outputs

    def forward(
            self,
            xs=None,
            ts=None,
            orders=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        n_speakers = [t.shape[1] for t in ts]
        ilens = [x.shape[0] for x in xs]
        emb = self.forward_encoder(xs, ilens)
        batch_size = len(xs)
        attractor_loss, attractors = self.eda([e[order] for e, order in zip(emb, orders)], n_speakers)

        # PIT
        ys = [torch.matmul(e, att.permute(1, 0)) for e, att in zip(emb, attractors)]
        max_n_speakers = max(n_speakers)
        ts_padded = pad_tensor(ts, max_n_speakers)
        ys_padded = pad_tensor(ys, max_n_speakers)
        _, labels = batch_pit_n_speaker_loss(ys_padded, ts_padded, n_speakers)
        pit_loss = standard_loss(ys, labels)

        # PSE
        with torch.no_grad():
            power_ts = [create_powerlabel(label.cpu().numpy(), self.mapping_dict, self.max_n_speaker).
                            to(emb[0].device, non_blocking=True) for label in labels]
        pad_attractors = [pad_attractor(att, self.max_n_speaker) for att in attractors]
        pse_ys = [torch.matmul(e, pad_att.permute(1, 0)) for e, pad_att in zip(emb, pad_attractors)]
        pse_logits = self.forward_post_net(pse_ys, torch.tensor(ilens))
        pse_loss = cal_power_loss(pse_logits, power_ts)
        loss = pse_loss + pit_loss + attractor_loss * self.attractor_loss_ratio

        stats = dict()
        stats["pse_loss"] = pse_loss.detach()
        stats["attractor_loss"] = attractor_loss.detach()
        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def estimate_sequential(self,
                            speech: torch.Tensor,
                            speech_lengths: torch.Tensor,
                            n_speakers: int = None,
                            shuffle: bool = True,
                            threshold: float = 0.5,
                            **kwargs):
        speech = [s[:s_len] for s, s_len in zip(speech, speech_lengths)]
        emb = self.forward_encoder(speech, speech_lengths)
        if shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractors, probs = self.eda.estimate(
                [e[torch.from_numpy(order).to(torch.long).to(speech[0].device)] for e, order in zip(emb, orders)])
        else:
            attractors, probs = self.eda.estimate(emb)
        attractors_active = []
        for p, att, e in zip(probs, attractors, emb):
            if n_speakers and n_speakers >= 0:
                att = att[:n_speakers, ]
                attractors_active.append(att)
            elif threshold is not None:
                silence = torch.nonzero(p < threshold)[0]
                n_spk = silence[0] if silence.size else None
                att = att[:n_spk, ]
                attractors_active.append(att)
            else:
                NotImplementedError('n_speakers or threshold has to be given.')
        raw_n_speakers = [att.shape[0] for att in attractors_active]
        attractors = [
            pad_attractor(att, self.max_n_speaker) if att.shape[0] <= self.max_n_speaker else att[:self.max_n_speaker]
            for att in attractors_active]
        ys = [torch.matmul(e, att.permute(1, 0)) for e, att in zip(emb, attractors)]
        logits = self.forward_post_net(ys, speech_lengths.cpu().to(torch.int64))
        ys = [self.recover_y_from_powerlabel(logit, raw_n_speaker) for logit, raw_n_speaker in
              zip(logits, raw_n_speakers)]

        return ys, emb, attractors, raw_n_speakers

    def recover_y_from_powerlabel(self, logit, n_speaker):
        pred = torch.argmax(torch.softmax(logit, dim=-1), dim=-1)
        oov_index = torch.where(pred == self.mapping_dict['oov'])[0]
        for i in oov_index:
            if i > 0:
                pred[i] = pred[i - 1]
            else:
                pred[i] = 0
        pred = [self.inv_mapping_func(i) for i in pred]
        decisions = [bin(num)[2:].zfill(self.max_n_speaker)[::-1] for num in pred]
        decisions = torch.from_numpy(
            np.stack([np.array([int(i) for i in dec]) for dec in decisions], axis=0)).to(logit.device).to(
            torch.float32)
        decisions = decisions[:, :n_speaker]
        return decisions

    def inv_mapping_func(self, label):

        if not isinstance(label, int):
            label = int(label)
        if label in self.mapping_dict['label2dec'].keys():
            num = self.mapping_dict['label2dec'][label]
        else:
            num = -1
        return num

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass
