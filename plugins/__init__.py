import mitsuba as mi

from plugins.bsdfs.rednermat import RednerMat
# from plugins.bsdfs.mitsubaBSDF import MitsubaBSDF
# BSDFs
mi.register_bsdf("rednermat", lambda props: RednerMat(props))
# mi.register_bsdf("mitsubaBSDF", lambda props: MitsubaBSDF(props))
