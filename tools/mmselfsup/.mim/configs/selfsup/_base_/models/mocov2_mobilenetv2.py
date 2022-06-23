# model settings
custom_imports=dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    type='MoCo',
    queue_len=4096,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        # use "scope.type" to import module from mmcls
        type='mmcls.MobileNetV2',
        out_indices=[7],
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=1280,
        hid_channels=1280,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
