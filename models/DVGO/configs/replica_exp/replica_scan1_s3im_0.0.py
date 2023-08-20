_base_ = './tt_default.py'

expname = 'dvgo_replica_scan1_s3im_0.0'

basedir = './logs/replica_exp/dvgo'

data = dict(
    inverse_y=False,
    dataset_type='replica',
    datadir='./data/Replica/scan1',
)

fine_train = dict(
    s3im_weight=0.0,
    s3im_kernel=4,
    s3im_stride=4,
    s3im_repeat_time=10,
    s3im_patch_height=64,
    s3im_patch_width=64,
)