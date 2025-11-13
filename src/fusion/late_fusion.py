#This is where the actual late fusion code will go
#AKA: final = w1*access + w2*quality
def fuse(accessibility, quality, w_access=0.6, w_quality=0.4):
    return w_access * accessibility + w_quality*quality