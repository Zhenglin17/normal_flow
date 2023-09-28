import numpy as np
import torch

def normalize_event_image(event_image, clamp_val=2., normalize_events=True):
    if not normalize_events:
        return event_image
    else:
        return torch.clamp(event_image, 0, clamp_val) / clamp_val  # + 1.) / 2.


def gen_event_images(event_volume, prefix, device="cuda", clamp_val=2., normalize_events=True):
    n_bins = int(event_volume.shape[1] / 2)
    time_range = torch.tensor(np.linspace(0.1, 1, n_bins), dtype=torch.float32)
    time_range = torch.reshape(time_range, (1, n_bins, 1, 1))

    pos_event_image = torch.sum(
        event_volume[:, :n_bins, ...] * time_range /
        (torch.sum(event_volume[:, :n_bins, ...], dim=1, keepdim=True) + 1e-5),
        dim=1, keepdim=True)
    neg_event_image = torch.sum(
        event_volume[:, n_bins:, ...] * time_range /
        (torch.sum(event_volume[:, n_bins:, ...], dim=1, keepdim=True) + 1e-5),
        dim=1, keepdim=True)

    outputs = {
        '{}_event_time_image'.format(prefix): (pos_event_image + neg_event_image) / 2.,
        '{}_event_image'.format(prefix): normalize_event_image(
            torch.sum(event_volume, dim=1, keepdim=True)),
        '{}_event_image_x'.format(prefix): normalize_event_image(
            torch.sum(event_volume.permute((0, 2, 1, 3)), dim=1, keepdim=True),
            normalize_events=normalize_events),
        '{}_event_image_y'.format(prefix): normalize_event_image(
            torch.sum(event_volume.permute(0, 3, 1, 2), dim=1, keepdim=True),
            normalize_events=normalize_events)
    }

    return outputs


def calc_floor_ceil_delta(x):
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]


def create_update(x, y, t, dt, p, vol_size):
    # assert (x >= 0).byte().all() and (x < vol_size[2]).byte().all()
    # assert (y >= 0).byte().all() and (y < vol_size[1]).byte().all()
    # assert (t >= 0).byte().all() and (t < vol_size[0] // 2).byte().all()

    vol_mul = torch.where(p < 0,
                          torch.ones(p.shape, dtype=torch.long) * vol_size[0] // 2,
                          torch.zeros(p.shape, dtype=torch.long))

    inds = (vol_size[1] * vol_size[2]) * (t + vol_mul) \
           + (vol_size[2]) * y \
           + x

    vals = dt

    return inds, vals


def gen_discretized_event_volume(events, vol_size):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size)

    x = events[:, 1].long()
    y = events[:, 0].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t - t_min) * ((vol_size[0] // 2 - 1) / (t_max - t_min))

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())

    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     events[:, 3],
                                     vol_size)

    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     events[:, 3],
                                     vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)

    return volume
