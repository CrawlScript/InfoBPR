# coding=utf-8

try:
    from .th_losses import th_info_bpr
except Exception as e:
    print(e)

try:
    from .tf_losses import tf_info_bpr
except Exception as e:
    print(e)
