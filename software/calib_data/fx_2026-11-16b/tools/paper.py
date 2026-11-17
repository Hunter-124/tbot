# usage: python paper.py X0,Y0,X1,Y1 img.png ...  (rough paper bbox at 1280x800; 50% edge crossings on all 4 sides)
import sys, cv2, numpy as np, json
# usage: paper.py img... ; finds the bright paper rectangle and measures it with 50% crossings
def cross(p, x0, n):
    a=np.median(p[:n]); b=np.median(p[-n:]); h=(a+b)/2
    k=np.where(np.diff(np.sign(p-h))!=0)[0]; k=k[np.argmin(np.abs(k-len(p)/2))]
    return x0+k+(h-p[k])/(p[k+1]-p[k]), a, b
out={}
x0h,y0h,x1h,y1h=map(int,sys.argv[1].split(','))
for f in sys.argv[2:]:
    im=cv2.imread(f,0).astype(float)
    sm=cv2.GaussianBlur(im,(0,0),2)
    # paper = largest bright compact blob near frame centre region
    thr=np.percentile(sm,99.0) if False else None
    m=(sm>sm.max()*0.0)  # placeholder
    # find by Otsu inside a central window
    y0,y1,x0,x1=y0h-40,y1h+40,x0h-40,x1h+40
    hw=sm[y0h:y1h,x0h:x1h]; ww=sm[y0:y1,x0:x1]; t=0.5*(np.percentile(hw,50)+np.percentile(ww,30)); b=(ww>t).astype(np.uint8)*255
    n,lab,st,cen=cv2.connectedComponentsWithStats(b)
    best=None
    for i in range(1,n):
        X,Y,W,H,A=st[i]
        if W<15 or H<15: continue
        fill=A/(W*H); ar=W/H
        if fill>0.85 and 0.5<ar<2.2 and (best is None or A>best[4]): best=(X+x0,Y+y0,W,H,A)
    if best is None: print(f,"no paper"); continue
    X,Y,W,H,_=best; mY=(Y+int(0.25*H),Y+int(0.75*H)); mX=(X+int(0.25*W),X+int(0.75*W)); r=max(5,int(0.15*min(W,H)))
    L=cross(im[mY[0]:mY[1],X-r:X+r].mean(0),X-r,max(3,r//2))
    R=cross(im[mY[0]:mY[1],X+W-r:X+W+r].mean(0),X+W-r,max(3,r//2))
    T=cross(im[Y-r:Y+r,mX[0]:mX[1]].mean(1),Y-r,max(3,r//2))
    B=cross(im[Y+H-r:Y+H+r,mX[0]:mX[1]].mean(1),Y+H-r,max(3,r//2))
    w=R[0]-L[0]; h=B[0]-T[0]
    out[f]=dict(w=w,h=h,cx=(L[0]+R[0])/2,cy=(T[0]+B[0])/2,contrast=[round(L[2]-L[1]),round(T[2]-T[1])])
    print(f"{f}: paper w {w:.2f}  h {h:.2f}  ratio {w/h:.4f} (letter 1.2941)  centre ({out[f]['cx']:.1f},{out[f]['cy']:.1f})  edge contrast L {L[1]:.0f}->{L[2]:.0f}")
json.dump(out,open('paper_meas.json','a'))
