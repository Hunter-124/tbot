import cv2, numpy as np, sys
sys.path.insert(0,'.')
from fx_edges import edge
FT=0.3048
# (snaps, ft, feature, S_m, xl, xr, band)
F = [
 ([2,3,4],28,"lie",0.225,459.8,578.0,(445,475)), ([5,6,7],20,"lie",0.225,451.7,603.3,(440,490)),
 ([8,9],15,"lie",0.225,445,650,(450,510)), ([11,12],10,"lie",0.225,415,720,(505,535)),
 ([2,3,4],28,"upL",0.12,425,487,(510,580)), ([5,6,7],20,"upL",0.12,415,495,(560,640)),
 ([8,9],15,"upL",0.12,360,481,(600,700)), ([11,12],10,"upL",0.12,310,475,(640,760)),
 ([2,3,4],28,"upR",0.12,551.7,611.7,(508,578)), ([5,6,7],20,"upR",0.12,580,659.3,(563,621)),
 ([8,9],15,"upR",0.12,589,696.5,(605,694)), ([11,12],10,"upR",0.12,630,795,(593,638)),
 ([16,17],35,"upR",0.12,540,586,(512,543)),
]
out=[]
for shots,ft,feat,S,xl,xr,band in F:
    ws=[]
    for i in shots:
        im=cv2.imread(f"snaps/snap_{i:04d}.png",0)
        win=max(3,0.06*(xr-xl))
        L=edge(im,band,xl,win); R=edge(im,band,xr,win); ws.append(R[0]-L[0])
        if i==shots[0]:
            pad=12; x0=int(xl-pad); x1=int(xr+pad); y0=band[0]-30; y1=band[1]+30
            c=cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_GRAY2BGR); s=max(1,int(400/(x1-x0)))
            c=cv2.resize(c,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
            for X in (L[0],R[0]): cv2.line(c,(int((X-x0)*s),0),(int((X-x0)*s),c.shape[0]),(0,0,255),1)
            for Y in band: cv2.line(c,(0,(Y-y0)*s),(c.shape[1],(Y-y0)*s),(0,255,0),1)
            cv2.imwrite(f"ov_{feat}_{ft}.png",c)
    w=float(np.mean(ws)); f=w*ft*FT/S
    out.append((ft,feat,S,w,f)); print(f"{ft:>3} ft {feat}: width {w:7.2f} px (spread {max(ws)-min(ws):.2f})  f={f:6.0f}")
np.save("fx_feat.npy",np.array([(a,0 if b=='lie' else 1 if b=='upL' else 2,c,d,e) for a,b,c,d,e in out]))
